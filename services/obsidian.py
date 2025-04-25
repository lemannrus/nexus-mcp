import json
import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from config import OBSIDIAN_VAULT_PATH, SEMANTIC_SEARCH_ENABLED, EMBEDDINGS_PATH
from services.logger import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2TokenizerFast

cached_embeddings: Optional[np.ndarray] = None
cached_paths: Optional[List[str]] = None
model: Optional[SentenceTransformer] = None

MODEL_TOKEN_LIMIT = 10000
CHUNK_SIZE_TOKENS = 512

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def tokenize_text(text: str) -> List[int]:
    """
    Tokenizes text using the preloaded tokenizer.

    Args:
        text: response text to tokenize

    Returns:
        tokenized text as a list of integers
    """
    logger.debug(f"Tokenizing text of length {len(text)}")
    return tokenizer.encode(text)


def split_text_into_chunks(text: str, max_tokens: int) -> List[str]:
    """
    Splits text into chunks based on word boundaries and a maximum token count.

    Args:
        text: text to split into chunks
        max_tokens: maximum number of tokens per chunk

    Returns:
        chunks of text as a list of strings
    """
    logger.debug(f"Splitting text into chunks with max_tokens={max_tokens}")
    words = text.split()
    chunks, current = [], []

    for word in words:
        current.append(word)
        if len(tokenizer.encode(" ".join(current))) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))

    logger.debug(f"Split into {len(chunks)} chunks")
    return chunks


def summarize_chunk(text_chunk: str) -> str:
    """
    Creates a simple summary of a text chunk by taking the first 200 characters.

    Args:
        text_chunk: text chunk to summarize

    Returns:
        summary of the text chunk as a string
    """
    logger.debug(f"Summarizing chunk with {len(text_chunk)} characters")
    return f"Summary: {text_chunk[:200]}..."


def get_note_path(title: str, folder_name: str = None) -> Optional[Path]:
    """
    Constructs the full path to a note file based on title and optional folder.

    If a folder_name is provided, it checks for the note within that specific folder first.
    If not found or no folder_name is given, it searches the entire vault recursively.
    If the note doesn't exist, it returns the potential path for a new note in the vault root.

    Args:
        title: The title of the note (filename without extension).
        folder_name: Optional name of the folder containing the note.

    Returns:
        A Path object to the note file, or None if an error occurred during search.
        If the note doesn't exist, returns the intended path for creation.
    """
    logger.debug(f"Getting path for note: title='{title}', folder_name='{folder_name}'")
    target_name = f"{title}.md"
    if folder_name:
        path = Path(OBSIDIAN_VAULT_PATH / folder_name / target_name)
        if path.exists():
            return path
    try:
        for path in OBSIDIAN_VAULT_PATH.rglob("*.md"):
            if path.name == target_name:
                return path
        logger.warning(f"Note with title '{title}' not found. Creating new note.")
        return OBSIDIAN_VAULT_PATH / target_name
    except Exception as e:
        logger.error(f"Failed to search for note '{title}': {e}")
        return None


def create_note(
    title: str, folder_name: Optional[str] = None, content: Optional[str] = ""
) -> str:
    """
    Creates a new markdown note in the specified folder or vault root.

    Args:
        title: The title for the new note (used as filename).
        folder_name: Optional subfolder name within the vault.
        content: Optional initial content for the note. Defaults to empty string.

    Returns:
        A status message indicating success or failure.
    """
    logger.debug(f"Creating note: title='{title}', folder_name='{folder_name}'")
    try:
        note_path = get_note_path(title, folder_name)
        if note_path.exists():
            return f"Note '{title}' already exists."
        note_path.write_text(content, encoding="utf-8")
        return f"Note '{title}' created."
    except Exception as e:
        logger.error(f"Failed to create note: {e}")
        return f"Failed to create note. {e}"


def read_note(title: str) -> str:
    """
    Reads the content of an existing note.

    If the note's token count exceeds MODEL_TOKEN_LIMIT, it returns a summarized version.

    Args:
        title: The title of the note to read.

    Returns:
        The full content of the note, a summarized version if too large, or an error message.
    """
    logger.debug(f"Reading note with title: '{title}'")
    note_path = get_note_path(title)
    if note_path is None or not note_path.exists():
        return f"Note '{title}' not found."

    try:
        full_text = note_path.read_text(encoding="utf-8")
        num_tokens = len(tokenize_text(full_text))

        if num_tokens <= MODEL_TOKEN_LIMIT:
            return full_text
        else:
            chunks = split_text_into_chunks(full_text, CHUNK_SIZE_TOKENS)
            logger.info(
                f"Note too large: {num_tokens} tokens, split into {len(chunks)} chunks."
            )
            summarized_chunks = [summarize_chunk(chunk) for chunk in chunks]
            full_summary = "\n\n".join(summarized_chunks)
            return f"Note is too large, providing summarized version:\n\n{full_summary}"

    except Exception as e:
        logger.error(f"Failed to read or summarize note: {e}")
        return "Failed to read note."


def update_note(title: str, new_content: str, folder_name: Optional[str] = None) -> str:
    """
    Updates the content of an existing note.

    Args:
        title: The title of the note to update.
        new_content: The new content to write to the note.
        folder_name: Optional subfolder name containing the note.

    Returns:
        A status message indicating success or failure.
    """
    logger.debug(f"Updating note: title='{title}', folder_name='{folder_name}'")
    try:
        note_path = get_note_path(title, folder_name)
        if not note_path.exists():
            return f"Note '{title}' not found."
        note_path.write_text(new_content, encoding="utf-8")
        return f"Note '{title}' updated."
    except Exception as e:
        logger.error(f"Failed to update note: {e}")
        return "Failed to update note."


def delete_note(title: str, folder_name: Optional[str] = None) -> str:
    """
    Deletes an existing note.

    Args:
        title: The title of the note to delete.
        folder_name: Optional subfolder name containing the note.

    Returns:
        A status message indicating success or failure.
    """
    logger.debug(f"Deleting note: title='{title}', folder_name='{folder_name}'")
    try:
        note_path = get_note_path(title, folder_name)
        if not note_path.exists():
            return f"Note '{title}' not found."
        note_path.unlink()
        return f"Note '{title}' deleted."
    except Exception as e:
        logger.error(f"Failed to delete note: {e}")
        return "Failed to delete note."


def load_vectors(json_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Loads embeddings and corresponding file paths from a JSON file.

    Args:
        json_path: Path to the JSON file containing embedding data.

    Returns:
        A tuple containing a numpy array of embeddings and a list of file paths.
    """
    logger.debug(f"Loading vectors from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data['vectors'])} vectors from {json_path}")
    embeddings = []
    paths = []
    for item in data["vectors"]:
        embeddings.append(item["embedding"])
        paths.append(item["path"])
    return np.array(embeddings, dtype=np.float32), paths


def initialize_semantic_search(
    vector_json_path: str, model_name: str = "nomic-ai/nomic-embed-text-v1.5"
):
    """
    Initializes the semantic search components (model, vectors, paths).

    Loads the sentence transformer model and the precomputed embeddings from disk.
    Avoids reloading if components are already in memory.

    Args:
        vector_json_path: Path to the JSON file containing embeddings.
        model_name: The name of the sentence transformer model to use.
    """
    global cached_embeddings, cached_paths, model
    logger.debug(f"Initializing semantic search with model '{model_name}'")
    if cached_embeddings is not None and cached_paths is not None and model is not None:
        return
    try:
        cached_embeddings, cached_paths = load_vectors(vector_json_path)
        model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info(
            f"Semantic search initialized: {cached_embeddings.shape[0]} embeddings loaded."
        )
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        cached_embeddings, cached_paths, model = None, None, None


def semantic_search(
    query: str, embeddings: np.ndarray, paths: List[str], top_k: int = 5
) -> List[str]:
    """
    Performs semantic search using cosine similarity between query and note embeddings.

    Falls back to recursive filename search if no strong semantic matches are found.

    Args:
        query: The search query string.
        embeddings: Precomputed embeddings for the notes.
        paths: List of file paths corresponding to the embeddings.
        top_k: The maximum number of results to return.

    Returns:
        A list of file paths matching the query, sorted by relevance.
    """
    logger.debug(f"Performing semantic search for query: '{query}'")
    global cached_embeddings, cached_paths, model

    if cached_embeddings is None or cached_paths is None or model is None:
        logger.warning(
            "Semantic search not initialized. Call initialize_semantic_search first."
        )
        return []

    query_vec = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_vec, embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]
    top_scores = similarities[top_indices]
    logger.info(f"Top {top_k} matches: {top_scores}")

    if np.all(top_scores < 1):
        logger.info(
            "No strong semantic matches found. Falling back to recursive filename search."
        )
        matched_paths = recursive_filename_search(query, OBSIDIAN_VAULT_PATH)
        return matched_paths[:top_k] if matched_paths else []

    return [paths[i] for i in top_indices]


def recursive_filename_search(query: str, root_dir: str) -> List[str]:
    """
    Recursively searches for files within a directory whose names contain the query string.

    Args:
        query: The string to search for in filenames (case-insensitive).
        root_dir: The directory to start the search from.

    Returns:
        A list of full file paths matching the query.
    """
    logger.debug(
        f"Recursively searching filenames for query: '{query}' in root_dir: '{root_dir}'"
    )
    matches = []
    query_lower = query.lower()
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if query_lower in filename.lower():
                full_path = os.path.join(dirpath, filename)
                matches.append(full_path)
    return matches


def search_notes_by_semantics(
    query: str, vector_json_path: str, top_k: int = 5
) -> List[str]:
    """
    Wrapper function to perform semantic search on notes using precomputed vectors.

    Args:
        query: The search query string.
        vector_json_path: Path to the JSON file containing embeddings.
        top_k: The maximum number of results to return.

    Returns:
        A list of file paths matching the query, sorted by relevance, or empty list on error.
    """
    logger.debug(f"Searching notes by semantics for query: '{query}'")
    try:
        embeddings, paths = load_vectors(vector_json_path)
        return semantic_search(query, embeddings, paths, top_k)
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


def simple_search_by_keyword(keyword: str) -> List[str]:
    """
    Performs a simple keyword search through note content and filenames.

    Args:
        keyword: The keyword to search for (case-insensitive).

    Returns:
        A list of relative file paths (from vault root) for matching notes.
    """
    logger.debug(f"Simple keyword search for: '{keyword}'")
    matching_notes = []
    try:
        for md_file in OBSIDIAN_VAULT_PATH.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                if (
                    keyword.lower() in content.lower()
                    or keyword.lower() in md_file.name.lower()
                ):
                    matching_notes.append(str(md_file.relative_to(OBSIDIAN_VAULT_PATH)))
            except Exception as e:
                logger.warning(f"Failed to read file {md_file}: {e}")
    except Exception as e:
        logger.error(f"Failed to search notes: {e}")
    return matching_notes


def search_notes_by_content(keyword: str) -> List[str]:
    """
    Searches notes by content, using either semantic search or simple keyword search.

    The search method depends on the SEMANTIC_SEARCH_ENABLED configuration flag.

    Args:
        keyword: The keyword or query string to search for.

    Returns:
        A list of file paths for matching notes.
    """
    logger.debug(f"Searching notes by content with keyword: '{keyword}'")
    if SEMANTIC_SEARCH_ENABLED:
        initialize_semantic_search(OBSIDIAN_VAULT_PATH / EMBEDDINGS_PATH)
        logger.info("Semantic search enabled")
        return search_notes_by_semantics(keyword, OBSIDIAN_VAULT_PATH / EMBEDDINGS_PATH)
    else:
        logger.info("Semantic search disabled")
        return simple_search_by_keyword(keyword)


def create_folder(folder_name: str) -> str:
    """
    Creates a new folder within the Obsidian vault.

    Args:
        folder_name: The name of the folder to create. Can include subdirectories (e.g., "path/to/folder").

    Returns:
        A status message indicating success or failure.
    """
    logger.debug(f"Creating folder: '{folder_name}'")
    folder_path = OBSIDIAN_VAULT_PATH / folder_name
    try:
        if folder_path.exists():
            return f"Folder '{folder_name}' already exists."
        folder_path.mkdir(parents=True, exist_ok=True)
        return f"Folder '{folder_name}' created."
    except Exception as e:
        logger.error(f"Failed to create folder: {e}")
        return "Failed to create folder."


def delete_folder(folder_name: str) -> str:
    """
    Deletes an empty folder from the Obsidian vault.

    Args:
        folder_name: The name of the folder to delete.

    Returns:
        A status message indicating success, failure, or if the folder is not empty.
    """
    logger.debug(f"Deleting folder: '{folder_name}'")
    folder_path = OBSIDIAN_VAULT_PATH / folder_name
    try:
        if not folder_path.exists():
            return f"Folder '{folder_name}' not found."
        folder_path.rmdir()
        return f"Folder '{folder_name}' deleted."
    except OSError:
        return f"Folder '{folder_name}' is not empty. Delete files first."
    except Exception as e:
        logger.error(f"Failed to delete folder: {e}")
        return "Failed to delete folder."


def search_folders(keyword: str) -> List[str]:
    """
    Searches for folders within the vault whose names contain the keyword.

    Args:
        keyword: The string to search for in folder names (case-insensitive).

    Returns:
        A list of relative folder paths (from vault root) matching the keyword.
    """
    logger.debug(f"Searching folders with keyword: '{keyword}'")
    try:
        return [
            str(folder.relative_to(OBSIDIAN_VAULT_PATH))
            for folder in OBSIDIAN_VAULT_PATH.rglob("*")
            if folder.is_dir() and keyword.lower() in folder.name.lower()
        ]
    except Exception as e:
        logger.error(f"Failed to search folders: {e}")
        return []


def list_folders() -> List[str]:
    """
    Lists all folders within the Obsidian vault.

    Returns:
        A list of relative folder paths (from vault root).
    """
    logger.debug("Listing all folders")
    try:
        return [
            str(folder.relative_to(OBSIDIAN_VAULT_PATH))
            for folder in OBSIDIAN_VAULT_PATH.rglob("*")
            if folder.is_dir()
        ]
    except Exception as e:
        logger.error(f"Failed to list folders: {e}")
        return []

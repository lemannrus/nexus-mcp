import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from config import OBSIDIAN_VAULT_PATH, SEMANTIC_SEARCH_ENABLED, EMBEDDINGS_PATH
from services.logger import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

cached_embeddings: Optional[np.ndarray] = None
cached_paths: Optional[List[str]] = None
model: Optional[SentenceTransformer] = None

def get_note_path(title: str, folder_name: Optional[str] = None) -> Path:
    """
    Create the path to specific note

    Args:
        title (str): The title of the note (used as filename).
        folder_name (Optional[str]): Optional subfolder within the vault.

    Returns:
        str: path to note
    """
    if folder_name is None:
        note_path = OBSIDIAN_VAULT_PATH / f"{title}.md"
    else:
        note_path = OBSIDIAN_VAULT_PATH / folder_name / f"{title}.md"
    return note_path


def create_note(
    title: str, folder_name: Optional[str] = None, content: Optional[str] = ""
) -> str:
    """
    Create a new markdown note in the specified folder.

    Args:
        title (str): The title of the note (used as filename).
        folder_name (Optional[str]): Optional subfolder within the vault.
        content (Optional[str]): Initial content of the note.

    Returns:
        str: Status message indicating success or failure.
    """
    try:
        note_path = get_note_path(title, folder_name)
        if note_path.exists():
            return f"Note '{title}' already exists."
        note_path.write_text(content, encoding="utf-8")
        return f"Note '{title}' created."
    except Exception as e:
        logger.error(f"Failed to create note: {e}")
        return f"Failed to create note. {e}"


def read_note(title: str, folder_name: Optional[str] = None) -> str:
    """
    Read the contents of an existing note.

    Args:
        title (str): The title of the note.
        folder_name (Optional[str]): Optional subfolder containing the note.

    Returns:
        str: The content of the note, or an error message if it doesn't exist.
    """
    try:
        note_path = get_note_path(title, folder_name)
        if not note_path.exists():
            return f"Note '{title}' not found."
        return note_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read note: {e}")
        return "Failed to read note."


def update_note(title: str, new_content: str, folder_name: Optional[str] = None) -> str:
    """
    Update the content of an existing note.

    Args:
        title (str): The title of the note.
        new_content (str): The new content to be written.
        folder_name (Optional[str]): Optional subfolder containing the note.

    Returns:
        str: Status message indicating success or failure.
    """
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
    Delete an existing note.

    Args:
        title (str): The title of the note.
        folder_name (Optional[str]): Optional subfolder containing the note.

    Returns:
        str: Status message indicating success or failure.
    """
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
    Load vector embeddings and associated note paths from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing vector data.

    Returns:
        Tuple[np.ndarray, List[str]]: A tuple with:
            - Numpy array of embeddings.
            - List of file paths corresponding to embeddings.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data['vectors'])} vectors from {json_path}")
    embeddings = []
    paths = []
    for item in data["vectors"]:
        embeddings.append(item["embedding"])
        paths.append(item["path"])

    return np.array(embeddings, dtype=np.float32), paths


def initialize_semantic_search(vector_json_path: str, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
    """
    Initialize global variables for semantic search:
    model, embeddings, and their associated note paths.

    Args:
        vector_json_path (str): Path to the JSON file with precomputed vectors.
        model_name (str): Name of the embedding model to use.
    """
    global cached_embeddings, cached_paths, model
    if cached_embeddings is not None and cached_paths is not None and model is not None:
        return  # Already initialized

    try:
        cached_embeddings, cached_paths = load_vectors(vector_json_path)
        model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info(f"Semantic search initialized: {cached_embeddings.shape[0]} embeddings loaded.")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        cached_embeddings, cached_paths, model = None, None, None


def semantic_search(query: str, embeddings: np.ndarray, paths: List[str], top_k: int = 5) -> List[str]:
    """
    Perform semantic search over precomputed embeddings.

    Args:
        query (str): Text query to search for.
        embeddings (np.ndarray): Array of note embeddings.
        paths (List[str]): List of paths corresponding to embeddings.
        top_k (int): Number of top results to return.

    Returns:
        List[str]: Top matching note paths.
    """
    global cached_embeddings, cached_paths, model
    if cached_embeddings is None or cached_paths is None or model is None:
        logger.warning("Semantic search not initialized. Call initialize_semantic_search first.")
        return []
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    query_vec = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_vec, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [paths[i] for i in top_indices]


def search_notes_by_semantics(query: str, vector_json_path: str, top_k: int = 5) -> List[str]:
    """
    Perform semantic search using vector data stored in a file.

    Args:
        query (str): Text query to search for.
        vector_json_path (str): Path to JSON file containing vectors.
        top_k (int): Number of top matches to return.

    Returns:
        List[str]: Paths of notes that best match the query.
    """
    try:
        embeddings, paths = load_vectors(vector_json_path)
        return semantic_search(query, embeddings, paths, top_k)
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


def simple_search_by_keyword(keyword: str) -> List[str]:
    """
    Search for notes containing a specific keyword in content or filename.

    Args:
        keyword (str): The keyword to search for.

    Returns:
        List[str]: Relative paths to matching notes.
    """
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
    Dispatch content search using either semantic or keyword search.

    Args:
        keyword (str): Search keyword or phrase.

    Returns:
        List[str]: Paths to notes that match the search.
    """
    if SEMANTIC_SEARCH_ENABLED:
        initialize_semantic_search(OBSIDIAN_VAULT_PATH / EMBEDDINGS_PATH)
        logger.info("Semantic search enabled")
        return search_notes_by_semantics(keyword, OBSIDIAN_VAULT_PATH / EMBEDDINGS_PATH)
    else:
        logger.info("Semantic search disabled")
        return simple_search_by_keyword(keyword)



def create_folder(folder_name: str) -> str:
    """
    Create a new folder inside the vault.

    Args:
        folder_name (str): Name of the folder to create.

    Returns:
        str: Status message indicating success or failure.
    """
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
    Delete a folder from the vault. Folder must be empty.

    Args:
        folder_name (str): Name of the folder to delete.

    Returns:
        str: Status message indicating success or failure.
    """
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
    Search for folders with names containing the given keyword.

    Args:
        keyword (str): Keyword to match in folder names.

    Returns:
        List[str]: Relative paths to matching folders.
    """
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
    List all folders within the vault.

    Returns:
        List[str]: Relative paths to all folders.
    """
    try:
        return [
            str(folder.relative_to(OBSIDIAN_VAULT_PATH))
            for folder in OBSIDIAN_VAULT_PATH.rglob("*")
            if folder.is_dir()
        ]
    except Exception as e:
        logger.error(f"Failed to list folders: {e}")
        return []

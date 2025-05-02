import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from obsidiantools.api import Vault

from config import OBSIDIAN_VAULT_PATH, SEMANTIC_SEARCH_ENABLED, EMBEDDINGS_PATH, OBSIDIAN_DEFAULT_FOLDER, \
    SIMILARITY_THRESHOLD, MODEL_TOKEN_LIMIT, CHUNK_SIZE_TOKENS
from services.logger import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2TokenizerFast

_cached_vault: Optional[Vault] = None

_cached_embeddings: Optional[np.ndarray] = None
_cached_paths: Optional[List[str]] = None
_semantic_model: Optional[SentenceTransformer] = None
_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") # Keep tokenizer global

def _get_vault() -> Vault:
    """
    Initializes and returns a cached ObsidianVault instance.

    Reads OBSIDIAN_VAULT_PATH from config.

    Returns:
        An initialized ObsidianVault instance.

    Raises:
        FileNotFoundError: If the vault path is not a valid directory.
        ImportError: If obsidiantools is not installed.
    """
    global _cached_vault
    try:
        if _cached_vault is None:
            if not OBSIDIAN_VAULT_PATH:
                logger.error("OBSIDIAN_VAULT_PATH is not set in config.")
                raise ValueError("OBSIDIAN_VAULT_PATH is not configured.")

            vault_path = Path(OBSIDIAN_VAULT_PATH)
            if not vault_path.is_dir():
                logger.error(f"Obsidian vault path is not a valid directory: {vault_path}")
                raise FileNotFoundError(f"Obsidian vault not found at {vault_path}")

            try:
                logger.info(f"Initializing ObsidianVault at: {vault_path}")
                # safe_mode=True can prevent execution of dataview/js, often safer for API usage
                _cached_vault = Vault(vault_path)
                logger.info(f"ObsidianVault initialized.")
            except ImportError:
                logger.error("obsidiantools library not found. Please install it: pip install obsidiantools")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize ObsidianVault: {e}", exc_info=True)
                raise RuntimeError(f"Failed to initialize ObsidianVault: {e}") from e
        return _cached_vault
    except Exception as e:
        logger.error(f"Failed to get ObsidianVault: {e}", exc_info=True)
        raise RuntimeError(f"Failed to get ObsidianVault: {e}") from e

def resolve_note_path(
    title: str,
    folder_name: Optional[str] = None,
    ensure_exists: bool = False,
) -> tuple[Path, str] | tuple[None, str] | None:
    """
    Resolves the absolute path for a note and its display folder name.

    Handles finding existing notes or determining the path for creation.
    Uses vault.markdown_notes for efficient searching of existing notes.

    Args:
        title: The title of the note (filename without extension).
        folder_name: Optional subfolder name within the vault.
        ensure_exists: If True, only returns a path if the note *file* already exists.
                       If False, returns the intended path even if it doesn't exist.

    Returns:
        Tuple (note_path, display_folder):
        - note_path: Path object to the note file, or None if not found when ensure_exists=True.
        - display_folder: String name of the folder for display purposes.

    Raises:
        ValueError: If title is empty or vault is invalid.
    """
    try:
        vault = _get_vault()

        if not title:
            raise ValueError("Note title cannot be empty.")
        if not isinstance(vault, Vault):
             raise TypeError("Invalid vault object provided.")

        filename = f"{title}.md"
        target_dir: Path
        display_folder: str

        logger.debug(f"Resolving path for title='{title}', folder='{folder_name}', ensure_exists={ensure_exists}")

        if folder_name:
            relative_folder_path = Path(folder_name.strip('/\\'))
            target_dir = vault.dirpath / relative_folder_path
            display_folder = relative_folder_path.as_posix()
        else:
            target_dir = vault.dirpath
            display_folder = OBSIDIAN_DEFAULT_FOLDER

        intended_path = target_dir / filename

        if intended_path.exists() and intended_path.is_file():
            logger.debug(f"Note found at specific path: {intended_path}")
            note_exists_in_index = title in vault.md_file_index
            indexed_path = vault.md_file_index.get(title)

            if note_exists_in_index and indexed_path == intended_path:
                 return intended_path, display_folder
            else:
                 logger.warning(f"Path {intended_path} exists but not in vault.markdown_notes. Treating as found.")
                 return intended_path, display_folder

        if folder_name and ensure_exists:
             logger.debug(f"Note not found at specific path '{intended_path}' and ensure_exists=True.")
             return None, display_folder

        logger.debug(f"Note not at '{intended_path}', searching known markdown notes globally.")
        found_globally: Optional[Path] = None

        for note_path in vault.md_file_index.values():
            if note_path.name == filename:
                found_globally = note_path
                logger.debug(f"Found matching filename in vault.markdown_notes: {found_globally}")
                break

        if found_globally:
            actual_path = found_globally
            try:
                actual_relative_path = actual_path.parent.relative_to(vault.dirpath)
                actual_display_folder = actual_relative_path.as_posix()
                if actual_display_folder == '.': actual_display_folder = "vault root"
            except ValueError:
                logger.warning(f"Found path {actual_path} is outside vault root {vault.dirpath}?")
                actual_display_folder = "[external?]"
            logger.debug(f"Note found globally at: {actual_path} (display folder: '{actual_display_folder}')")
            return actual_path, actual_display_folder
        else:
            logger.debug(f"Note '{filename}' not found anywhere in the vault's known markdown notes.")
            if ensure_exists:
                return None, display_folder
            else:
                logger.debug(f"Returning intended path for creation: {intended_path}")
                return intended_path, display_folder
    except Exception as e:
        logger.error(f"Unexpected error resolving note path: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error resolving note path: {e}") from e

def tokenize_text(text: str) -> List[int]:
    """Tokenizes text using the preloaded tokenizer."""
    logger.debug(f"Tokenizing text of length {len(text)}")
    return _tokenizer.encode(text)

def split_text_into_chunks(text: str, max_tokens: int) -> List[str]:
    """Splits text into chunks based on word boundaries and a maximum token count."""
    logger.debug(f"Splitting text into chunks with max_tokens={max_tokens}")
    words = text.split()
    chunks, current_chunk_words = [], []
    current_token_count = 0

    for word in words:
        word_token_count = len(_tokenizer.encode(word))
        if current_token_count + word_token_count + (1 if current_chunk_words else 0) > max_tokens:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
            current_chunk_words = [word]
            current_token_count = word_token_count
            if word_token_count > max_tokens:
                 logger.warning(f"Word '{word[:30]}...' exceeds max_tokens ({max_tokens}), chunk may be oversized.")
                 chunks.append(" ".join(current_chunk_words))
                 current_chunk_words = []
                 current_token_count = 0
        else:
            current_chunk_words.append(word)
            current_token_count += word_token_count + (1 if len(current_chunk_words) > 1 else 0)

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    logger.debug(f"Split into {len(chunks)} chunks")
    return chunks


def summarize_chunk(text_chunk: str) -> str:
    """Creates a simple summary of a text chunk (first 200 chars)."""
    logger.debug(f"Summarizing chunk with {len(text_chunk)} characters")
    return f"Summary: {text_chunk[:200]}..."

def create_note(
    title: str, folder_name: Optional[str] = None, content: Optional[str] = None
) -> str:
    """Creates a new markdown note."""
    note_content = content if content is not None else ""
    logger.debug(f"Creating note: title='{title}', folder_name='{folder_name}'")
    try:
        note_path, display_folder = resolve_note_path(title, folder_name, ensure_exists=False)

        if note_path.exists():
            logger.warning(f"Note '{note_path}' already exists.")
            return f"Note '{title}' already exists in '{display_folder}'."

        logger.debug(f"Ensuring directory exists: '{note_path.parent}'")
        note_path.parent.mkdir(parents=True, exist_ok=True)

        note_path.write_text(note_content, encoding="utf-8")
        logger.info(f"Successfully created note at '{note_path}'")
        return f"Note '{title}' created successfully in '{display_folder}'."

    except (FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.error(f"Failed to create note '{title}': {e}", exc_info=isinstance(e, (RuntimeError, OSError)))
        return f"Failed to create note '{title}'. Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error creating note '{title}': {e}", exc_info=True)
        return f"Failed to create note '{title}'. Unexpected error: {e}"


def read_note(title: str, folder_name: Optional[str] = None) -> str:
    """Reads note content, summarizes if too large based on MODEL_TOKEN_LIMIT."""
    logger.debug(f"Reading note: title='{title}', folder_name='{folder_name}'")
    try:
        note_path, display_folder = resolve_note_path(title, folder_name, ensure_exists=True)

        if note_path is None:
            folder_info = f" in folder '{display_folder}'" if folder_name else ""
            logger.warning(f"Note '{title}'{folder_info} not found.")
            return f"Note '{title}'{folder_info} not found."

        logger.debug(f"Reading content from: {note_path}")
        full_text = note_path.read_text(encoding="utf-8")
        num_tokens = len(tokenize_text(full_text))

        if num_tokens <= MODEL_TOKEN_LIMIT:
            return full_text
        else:
            chunks = split_text_into_chunks(full_text, CHUNK_SIZE_TOKENS)
            logger.info(
                f"Note '{title}' is too large ({num_tokens} tokens > {MODEL_TOKEN_LIMIT}). Returning summary."
            )
            summarized_chunks = [summarize_chunk(chunk) for chunk in chunks]
            full_summary = "\n\n".join(summarized_chunks)
            return f"Note '{title}' in '{display_folder}' is too large, providing summarized version:\n\n{full_summary}"

    except (FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.error(f"Failed to read note '{title}': {e}", exc_info=isinstance(e, (RuntimeError, OSError)))
        return f"Failed to read note '{title}'. Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error reading note '{title}': {e}", exc_info=True)
        return f"Failed to read note '{title}'. Unexpected error: {e}"


def update_note(title: str, new_content: str, folder_name: Optional[str] = None) -> str:
    """Updates the content of an existing note."""
    logger.debug(f"Updating note: title='{title}', folder_name='{folder_name}'")
    try:
        note_path, display_folder = resolve_note_path(title, folder_name, ensure_exists=True)

        if note_path is None:
            folder_info = f" in folder '{display_folder}'" if folder_name else ""
            logger.warning(f"Note '{title}'{folder_info} not found for update.")
            return f"Note '{title}'{folder_info} not found."

        logger.debug(f"Writing update to: {note_path}")
        note_path.write_text(new_content, encoding="utf-8")
        return f"Note '{title}' in '{display_folder}' updated."

    except (FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.error(f"Failed to update note '{title}': {e}", exc_info=isinstance(e, (RuntimeError, OSError)))
        return f"Failed to update note '{title}'. Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error updating note '{title}': {e}", exc_info=True)
        return f"Failed to update note '{title}'. Unexpected error: {e}"


def delete_note(title: str, folder_name: Optional[str] = None) -> str:
    """Deletes an existing note."""
    logger.debug(f"Deleting note: title='{title}', folder_name='{folder_name}'")
    try:
        note_path, display_folder = resolve_note_path(title, folder_name, ensure_exists=True)

        if note_path is None:
            folder_info = f" in folder '{display_folder}'" if folder_name else ""
            logger.warning(f"Note '{title}'{folder_info} not found for deletion.")
            return f"Note '{title}'{folder_info} not found."

        logger.debug(f"Deleting file: {note_path}")
        note_path.unlink()
        return f"Note '{title}' in '{display_folder}' deleted."

    except (FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.error(f"Failed to delete note '{title}': {e}", exc_info=isinstance(e, (RuntimeError, OSError)))
        return f"Failed to delete note '{title}'. Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error deleting note '{title}': {e}", exc_info=True)
        return f"Failed to delete note '{title}'. Unexpected error: {e}"

def load_vectors(json_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Loads embeddings and paths from JSON."""
    logger.debug(f"Loading vectors from: {json_path}")
    if not json_path.is_file():
        logger.error(f"Embeddings JSON file not found: {json_path}")
        raise FileNotFoundError(f"Embeddings JSON file not found: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "vectors" not in data or not isinstance(data["vectors"], list):
             raise ValueError("Invalid format in embeddings JSON file: 'vectors' key missing or not a list.")

        embeddings = []
        paths = []
        for item in data["vectors"]:
            if "embedding" in item and "path" in item:
                embeddings.append(item["embedding"])
                paths.append(item["path"])
            else:
                logger.warning("Skipping invalid item in embeddings JSON.")

        if not embeddings:
             raise ValueError("No valid embeddings found in JSON file.")

        logger.info(f"Loaded {len(embeddings)} vectors from {json_path}")
        return np.array(embeddings, dtype=np.float32), paths
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {json_path}: {e}")
        raise ValueError(f"Invalid JSON in embeddings file: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load vectors from {json_path}: {e}", exc_info=True)
        raise


def initialize_semantic_search(
    model_name: str = "nomic-ai/nomic-embed-text-v1.5", # Or get from config
    force_reload: bool = False
):
    """Initializes semantic search model and loads vectors."""
    global _cached_embeddings, _cached_paths, _semantic_model
    logger.debug(f"Initializing semantic search with model '{model_name}'")

    vector_json_path = OBSIDIAN_VAULT_PATH / EMBEDDINGS_PATH
    logger.info(f"Embeddings path: {vector_json_path}")

    if not force_reload and _cached_embeddings is not None and _cached_paths is not None and _semantic_model is not None:
        logger.info("Semantic search components already initialized.")
        return True

    try:
        logger.info("Loading embeddings...")
        _cached_embeddings, _cached_paths = load_vectors(vector_json_path)

        logger.info(f"Loading sentence transformer model: {model_name}...")
        _semantic_model = SentenceTransformer(model_name, trust_remote_code=True)

        logger.info(
            f"Semantic search initialized: {_cached_embeddings.shape[0]} embeddings loaded."
        )
        return True
    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error(f"Semantic search initialization failed: {e}")
        _cached_embeddings, _cached_paths, _semantic_model = None, None, None
        return False
    except Exception as e:
        logger.error(f"Unexpected error during semantic search initialization: {e}", exc_info=True)
        _cached_embeddings, _cached_paths, _semantic_model = None, None, None
        return False

def recursive_filename_search(query: str, vault: Vault) -> List[str]:
    """Searches markdown note filenames within the vault."""
    logger.debug(f"Recursively searching filenames for query: '{query}' using obsidiantools")
    matches = []
    query_lower = query.lower()
    try:
        for note_path in vault.dirpath.rglob("*.md"):
            if query_lower in note_path.name.lower():
                matches.append(str(note_path))
    except Exception as e:
         logger.error(f"Error during recursive filename search: {e}", exc_info=True)
    return matches


def semantic_search(query: str, top_k: int = 5) -> List[str]:
    """Performs semantic search using loaded model and embeddings."""
    global _cached_embeddings, _cached_paths, _semantic_model
    logger.debug(f"Performing semantic search for query: '{query}'")

    if not SEMANTIC_SEARCH_ENABLED:
         logger.warning("Semantic search is disabled in config.")
         return []

    # Ensure initialization
    if _cached_embeddings is None or _cached_paths is None or _semantic_model is None:
        logger.warning("Semantic search not initialized. Attempting initialization...")
        if not initialize_semantic_search():
            logger.error("Cannot perform semantic search: Initialization failed.")
            return []

    try:
        vault = _get_vault()

        query_vec = _semantic_model.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_vec, _cached_embeddings)[0]

        effective_top_k = min(top_k, len(_cached_paths))
        if effective_top_k <= 0: return []

        top_indices = np.argsort(similarities)[-effective_top_k:][::-1]
        top_scores = similarities[top_indices]
        logger.info(f"Top {effective_top_k} semantic matches scores: {top_scores}")

        strong_match_indices = [idx for idx, score in zip(top_indices, top_scores) if score >= SIMILARITY_THRESHOLD]

        if not strong_match_indices:
            logger.info(
                f"No strong semantic matches found (threshold={SIMILARITY_THRESHOLD}). Falling back to filename search."
            )
            matched_paths = recursive_filename_search(query, vault)
            return matched_paths[:top_k]
        else:
            logger.info(f"Found {len(strong_match_indices)} strong semantic matches.")
            return [_cached_paths[i] for i in strong_match_indices]

    except Exception as e:
        logger.error(f"Error during semantic search query: {e}", exc_info=True)
        return []

def simple_search_by_keyword(keyword: str) -> List[str]:
    """Performs keyword search in markdown note filenames and content using obsidiantools."""
    logger.debug(f"Simple keyword search for: '{keyword}'")
    matching_notes_relative: List[str] = []
    try:
        vault = _get_vault()
        keyword_lower = keyword.lower()

        for note_path in vault.dirpath.rglob("*.md"):
            try:
                match_found = False
                if keyword_lower in note_path.name.lower():
                    match_found = True

                if not match_found:
                    content = note_path.read_text(encoding="utf-8")
                    if keyword_lower in content.lower():
                        match_found = True

                if match_found:
                    relative_path = str(note_path.relative_to(vault.dirpath))
                    matching_notes_relative.append(relative_path)
                    logger.debug(f"Keyword found in: {relative_path}")

            except Exception as e:
                logger.warning(f"Failed to process file {note_path} during keyword search: {e}")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
         logger.error(f"Failed to perform keyword search: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during keyword search: {e}", exc_info=True)

    return sorted(list(set(matching_notes_relative)))

def search_notes_by_content(query: str, top_k: int = 5) -> List[str]:
    """Searches notes by content/keyword. Uses semantic or simple search based on config."""
    logger.debug(f"Unified search for: '{query}', top_k={top_k}")
    if SEMANTIC_SEARCH_ENABLED:
        logger.info("Using semantic search.")
        results = semantic_search(query, top_k)
        logger.info(f"Semantic search returned {len(results)} results.")
        return results
    else:
        logger.info("Using simple keyword search.")
        results = simple_search_by_keyword(query)
        logger.info(f"Simple keyword search returned {len(results)} results.")
        return results[:top_k]

def create_folder(folder_name: str) -> str:
    """Creates a folder within the vault."""
    logger.debug(f"Creating folder: '{folder_name}'")
    if not folder_name:
        return "Folder name cannot be empty."
    try:
        vault = _get_vault()
        relative_folder_path = Path(folder_name.strip('/\\'))
        folder_path = vault.dirpath / relative_folder_path
        display_name = relative_folder_path.as_posix()

        if folder_path.exists():
            if folder_path.is_dir():
                 logger.warning(f"Folder '{display_name}' already exists.")
                 return f"Folder '{display_name}' already exists."
            else:
                 logger.error(f"Path '{display_name}' exists but is not a folder.")
                 return f"Path '{display_name}' exists but is not a folder."

        logger.debug(f"Creating directory: {folder_path}")
        folder_path.mkdir(parents=True, exist_ok=True)
        return f"Folder '{display_name}' created."

    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        logger.error(f"Failed to create folder '{folder_name}': {e}", exc_info=isinstance(e, (RuntimeError, OSError)))
        return f"Failed to create folder '{folder_name}'. Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error creating folder '{folder_name}': {e}", exc_info=True)
        return f"Failed to create folder '{folder_name}'. Unexpected error: {e}"


def delete_folder(folder_name: str) -> str:
    """Deletes an *empty* folder from the vault."""
    logger.debug(f"Deleting folder: '{folder_name}'")
    display_name = ""
    if not folder_name:
        return "Folder name cannot be empty."
    try:
        vault = _get_vault()
        relative_folder_path = Path(folder_name.strip('/\\'))
        folder_path = vault.dirpath / relative_folder_path
        display_name = relative_folder_path.as_posix()

        if not folder_path.exists():
            logger.warning(f"Folder '{display_name}' not found for deletion.")
            return f"Folder '{display_name}' not found."
        if not folder_path.is_dir():
            logger.warning(f"Path '{display_name}' is not a folder.")
            return f"Path '{display_name}' is not a folder."

        logger.debug(f"Attempting to delete directory: {folder_path}")
        folder_path.rmdir()
        return f"Folder '{display_name}' deleted."

    except OSError as e:
        if "Directory not empty" in str(e) or "[Errno 39]" in str(e) or "[WinError 145]" in str(e) :
            logger.warning(f"Folder '{display_name}' is not empty.")
            return f"Folder '{display_name}' is not empty. Delete content first."
        else:
             logger.error(f"OS Error deleting folder '{display_name}': {e}", exc_info=True)
             return f"Failed to delete folder '{display_name}'. OS Error: {e}"
    except (FileNotFoundError, ValueError, RuntimeError) as e:
         logger.error(f"Failed to delete folder '{display_name}': {e}", exc_info=isinstance(e, RuntimeError))
         return f"Failed to delete folder '{display_name}'. Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error deleting folder '{display_name}': {e}", exc_info=True)
        return f"Failed to delete folder '{display_name}'. Unexpected error: {e}"


def search_folders(keyword: str) -> List[str]:
    """Searches folder names within the vault."""
    logger.debug(f"Searching folders with keyword: '{keyword}'")
    matching_folders_relative: List[str] = []
    try:
        vault = _get_vault()
        keyword_lower = keyword.lower()

        for item in vault.dirpath.rglob("*"):
            if item.is_dir() and keyword_lower in item.name.lower():
                 relative_path = str(item.relative_to(vault.dirpath))
                 matching_folders_relative.append(relative_path)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to search folders: {e}")
    except Exception as e:
        logger.error(f"Unexpected error searching folders: {e}", exc_info=True)

    return sorted(list(set(matching_folders_relative)))


def list_folders() -> List[str]:
    """Lists all folders within the Obsidian vault."""
    logger.debug("Listing all folders")
    folders_relative: List[str] = []
    try:
        vault = _get_vault()
        for item in vault.dirpath.rglob("*"):
            if item.is_dir():
                 relative_path = str(item.relative_to(vault.dirpath))
                 folders_relative.append(relative_path)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to list folders: {e}")
    except Exception as e:
        logger.error(f"Unexpected error listing folders: {e}", exc_info=True)

    return sorted([f for f in list(set(folders_relative)) if f != '.'])

if SEMANTIC_SEARCH_ENABLED:
    initialize_semantic_search()
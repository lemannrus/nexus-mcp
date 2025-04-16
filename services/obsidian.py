from pathlib import Path
from typing import Optional, List
from config import OBSIDIAN_VAULT_PATH, OBSIDIAN_DEFAULT_FOLDER
from services.logger import logger


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


def search_notes_by_content(keyword: str) -> List[str]:
    """
    Search for notes containing a specific keyword in their content or filename.

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

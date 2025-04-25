"""
Main entry point for the Personal Assistant MCP Server.

This script initializes a `FastMCP` server and registers various tools
for interacting with external services including:

- Google Calendar (event management)
- Obsidian Vault (markdown notes and folder utilities)
- Trello (task and project management)
- Web page parsing (HTML content scraping)

Each tool is registered using the `@mcp.tool()` decorator, enabling
MCP-compatible clients to invoke them via a common interface.
"""

import logging
from mcp.server.fastmcp import FastMCP

from config import SEMANTIC_SEARCH_ENABLED, OBSIDIAN_VAULT_PATH, EMBEDDINGS_PATH

# Calendar tools
from services.gcalendar import create_event, list_events, update_event, delete_event

# Obsidian notes tools
from services.obsidian import (
    create_note,
    read_note,
    update_note,
    delete_note,
    search_notes_by_content,
    create_folder,
    delete_folder,
    search_folders,
    list_folders,
    initialize_semantic_search,
)

# Trello tools
from services.trello_service import (
    list_boards,
    list_lists,
    list_cards,
    create_card,
    update_card,
    delete_card,
    search_cards,
)

# Web parser
from services.parser import parse_webpage

# Initialize basic logging
logging.basicConfig(level=logging.INFO)

# Initialize FastMCP server instance
mcp = FastMCP("Personal Assistant MCP server")

# ======================
# Calendar Tool Registration
# ======================

mcp.tool(
    description="Create a new event in Google Calendar. Requires title, time, and optionally description and participants."
)(create_event)
mcp.tool(
    description="List upcoming Google Calendar events. Optional filters by date or keyword."
)(list_events)
mcp.tool(
    description="Update an existing Google Calendar event by ID. Title, time, and description can be modified."
)(update_event)
mcp.tool(description="Delete an existing Google Calendar event by its ID.")(
    delete_event
)

# ======================
# Obsidian Vault Tool Registration
# ======================

mcp.tool(
    description="Create a new markdown note in the Obsidian vault. Title must be unique. Content is optional."
)(create_note)
mcp.tool(
    description="Read the contents of a note by title. Searches the entire Obsidian vault recursively by file name."
)(read_note)
mcp.tool(
    description="Update the content of an existing note by title. The note is found recursively by name."
)(update_note)
mcp.tool(
    description="Delete a note by title. The note is located recursively across the vault. Always ask confirmation for this"
)(delete_note)
mcp.tool(
    description="Search for notes using either semantic similarity (if enabled) or keyword matching. Returns a list of relative paths."
)(search_notes_by_content)
mcp.tool(
    description="Create a new folder inside the Obsidian vault. Folder names can include subpaths like 'projects/ai'."
)(create_folder)
mcp.tool(
    description="Delete an empty folder from the vault by name. Fails if the folder contains files."
)(delete_folder)
mcp.tool(
    description="Search vault folders by keyword in their name. Returns relative paths."
)(search_folders)
mcp.tool(description="List all folders currently present inside the Obsidian vault.")(
    list_folders
)

# ======================
# Trello Tool Registration
# ======================

mcp.tool(description="List all Trello boards available to the user.")(list_boards)
mcp.tool(
    description="List all lists (columns) on a given Trello board. Requires the board ID."
)(list_lists)
mcp.tool(description="List all cards in a specific Trello list. Requires the list ID.")(
    list_cards
)
mcp.tool(
    description="Create a new Trello card in the specified list. Requires list ID and card title."
)(create_card)
mcp.tool(
    description="Update a Trello card by ID. You can modify title, description, and other attributes."
)(update_card)
mcp.tool(description="Delete a Trello card by its ID.")(delete_card)
mcp.tool(description="Search Trello cards by a keyword in their title or description.")(
    search_cards
)

# ======================
# Web Parsing Tool
# ======================

mcp.tool(
    description="Extracts and cleans text content from a given webpage URL. Removes scripts, styles, and navigation clutter."
)(parse_webpage)

# ======================
# Entry Point
# ======================

if __name__ == "__main__":
    if SEMANTIC_SEARCH_ENABLED:
        initialize_semantic_search(OBSIDIAN_VAULT_PATH / EMBEDDINGS_PATH)
    mcp.run()

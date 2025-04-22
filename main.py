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

Modules Registered:
- services/gcalendar.py
- services/obsidian.py
- services/trello_service.py
- services/parser.py
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
    list_folders, initialize_semantic_search,
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

mcp.tool()(create_event)  # Create a new calendar event
mcp.tool()(list_events)  # List upcoming calendar events
mcp.tool()(update_event)  # Update an existing event by ID
mcp.tool()(delete_event)  # Delete a calendar event by ID

# ======================
# Obsidian Vault Tool Registration
# ======================

mcp.tool()(create_note)  # Create a new markdown note
mcp.tool()(read_note)  # Read contents of a markdown note
mcp.tool()(update_note)  # Update note content
mcp.tool()(delete_note)  # Delete a note
mcp.tool()(search_notes_by_content)  # Full-text search in notes
mcp.tool()(create_folder)  # Create a new folder in the vault
mcp.tool()(delete_folder)  # Delete an existing folder
mcp.tool()(search_folders)  # Search for folders by name
mcp.tool()(list_folders)  # List all folders in the vault

# ======================
# Trello Tool Registration
# ======================

mcp.tool()(list_boards)  # List all Trello boards
mcp.tool()(list_lists)  # List lists in a Trello board
mcp.tool()(list_cards)  # List cards in a Trello list
mcp.tool()(create_card)  # Create a new Trello card
mcp.tool()(update_card)  # Update an existing card
mcp.tool()(delete_card)  # Delete a card by ID
mcp.tool()(search_cards)  # Search Trello cards by text query

# ======================
# Web Parsing Tool
# ======================

mcp.tool()(parse_webpage)  # Extract and clean HTML content from a given URL

# ======================
# Entry Point
# ======================

if __name__ == "__main__":
    if SEMANTIC_SEARCH_ENABLED:
        initialize_semantic_search(OBSIDIAN_VAULT_PATH / EMBEDDINGS_PATH)
    # Start the FastMCP server
    mcp.run()

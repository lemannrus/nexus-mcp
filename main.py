"""
Main entry point for the Personal Assistant MCP Server.

This script initializes a `FastMCP` server and dynamically registers various tools
based on configuration flags.

Registered tools:
- Google Calendar (event management)
- Obsidian Vault (markdown notes and folder utilities)
- Trello (task and project management)
- Web page parsing (HTML content scraping)
- Google News Search
"""

import logging
from mcp.server.fastmcp import FastMCP

from config import (
    ENABLE_OBSIDIAN_TOOLS,
    ENABLE_TRELLO_TOOLS,
    ENABLE_CALENDAR_TOOLS,
    ENABLE_NEWS_SEARCH,
    ENABLE_WEB_PARSER,
)
from services.gcalendar import create_event, list_events, update_event, delete_event
from services.logger import logger
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
)
from services.trello_service import (
    list_boards,
    list_lists,
    list_cards,
    create_card,
    update_card,
    delete_card,
    search_cards,
)
from services.parser import parse_webpage

# Google News
from services.google_news import search_news_google

# Initialize basic logging
logging.basicConfig(level=logging.INFO)

# Initialize FastMCP server instance
mcp = FastMCP("Nexus MCP server")

# ======================
# Tool registry
# ======================

tools = []

if ENABLE_CALENDAR_TOOLS:
    tools += [
        (
            create_event,
            "Create a new event in Google Calendar. Requires title, time, and optionally description and participants.",
        ),
        (
            list_events,
            "List upcoming Google Calendar events. Optional filters by date or keyword.",
        ),
        (
            update_event,
            "Update an existing Google Calendar event by ID. Title, time, and description can be modified.",
        ),
        (delete_event, "Delete an existing Google Calendar event by its ID."),
    ]

if ENABLE_OBSIDIAN_TOOLS:
    tools += [
        (
            create_note,
            "Create a new markdown note in the Obsidian vault. Title must be unique. Content is optional.",
        ),
        (
            read_note,
            "Read the contents of a note by title. Searches the entire Obsidian vault recursively by file name.",
        ),
        (
            update_note,
            "Update the content of an existing note by title. The note is found recursively by name.",
        ),
        (
            delete_note,
            "Delete a note by title. The note is located recursively across the vault.",
        ),
        (
            search_notes_by_content,
            "Search for notes using either semantic similarity (if enabled) or keyword matching. Returns a list of relative paths.",
        ),
        (
            create_folder,
            "Create a new folder inside the Obsidian vault. Folder names can include subpaths like 'projects/ai'.",
        ),
        (
            delete_folder,
            "Delete an empty folder from the vault by name. Fails if the folder contains files.",
        ),
        (
            search_folders,
            "Search vault folders by keyword in their name. Returns relative paths.",
        ),
        (list_folders, "List all folders currently present inside the Obsidian vault."),
    ]

if ENABLE_TRELLO_TOOLS:
    tools += [
        (list_boards, "List all Trello boards available to the user."),
        (
            list_lists,
            "List all lists (columns) on a given Trello board. Requires the board ID.",
        ),
        (list_cards, "List all cards in a specific Trello list. Requires the list ID."),
        (
            create_card,
            "Create a new Trello card in the specified list. Requires list ID and card title.",
        ),
        (
            update_card,
            "Update a Trello card by ID. You can modify title, description, and other attributes.",
        ),
        (delete_card, "Delete a Trello card by its ID."),
        (
            search_cards,
            "Search Trello cards by a keyword in their title or description.",
        ),
    ]

if ENABLE_WEB_PARSER:
    tools.append(
        (
            parse_webpage,
            "Extracts and cleans text content from a given webpage URL. Removes scripts, styles, and navigation clutter.",
        )
    )

if ENABLE_NEWS_SEARCH:
    tools.append((search_news_google, "Search for news articles on Google News."))

# ======================
# Register tools
# ======================

for func, description in tools:
    mcp.tool(description=description)(func)

# ======================
# Entry Point
# ======================

if __name__ == "__main__":
    mcp.run()

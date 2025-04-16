"""
This module provides various utility functions to interact with Trello, such as managing boards, lists, and cards.

The module relies on the `TrelloClient` object from the `trello` package, which is initialized using API credentials.
It includes functions for listing boards, lists, and cards, as well as creating, updating, deleting, and searching for cards.
In addition, it logs errors for any exceptions using the `logging` module.
"""

from typing import List, Optional
from trello import TrelloClient
from config import TRELLO_API_KEY, TRELLO_TOKEN
from services.logger import logger

# Initialize the Trello client
client = TrelloClient(api_key=TRELLO_API_KEY, token=TRELLO_TOKEN)


def list_boards() -> List[str]:
    """
    Fetches and lists all Trello boards associated with the API credentials.

    Returns:
        List[str]: A list of strings containing board names and their IDs.
                   Format: "<board name> (ID: <board id>)"
                   Returns an empty list if the operation fails.
    Logs:
        Errors are logged in case of an exception.
    """
    try:
        boards = client.list_boards()
        return [f"{board.name} (ID: {board.id})" for board in boards]
    except Exception as e:
        logger.error(f"Failed to list boards: {e}")
        return []


def list_lists(board_id: str) -> List[str]:
    """
    Fetches and lists all the lists of a specified Trello board.

    Args:
        board_id (str): The ID of the Trello board.

    Returns:
        List[str]: A list of strings containing list names and their IDs.
                   Format: "<list name> (ID: <list id>)"
                   Returns an empty list if the operation fails.
    Logs:
        Errors are logged in case of an exception.
    """
    try:
        board = client.get_board(board_id)
        lists = board.list_lists()
        return [f"{lst.name} (ID: {lst.id})" for lst in lists]
    except Exception as e:
        logger.error(f"Failed to list lists: {e}")
        return []


def list_cards(list_id: str) -> List[str]:
    """
    Fetches and lists all the cards of a specified Trello list.

    Args:
        list_id (str): The ID of the Trello list.

    Returns:
        List[str]: A list of strings containing card names and their IDs.
                   Format: "<card name> (ID: <card id>)"
                   Returns an empty list if the operation fails.
    Logs:
        Errors are logged in case of an exception.
    """
    try:
        lst = client.get_list(list_id)
        cards = lst.list_cards()
        return [f"{card.name} (ID: {card.id})" for card in cards]
    except Exception as e:
        logger.error(f"Failed to list cards: {e}")
        return []


def create_card(list_id: str, name: str, description: Optional[str] = "") -> str:
    """
    Creates a new card in a Trello list.

    Args:
        list_id (str): The ID of the list where the card is to be created.
        name (str): The name of the card.
        description (Optional[str]): A description of the card (defaults to an empty string).

    Returns:
        str: A message indicating whether the card creation was successful or not.
    Logs:
        Errors are logged in case of an exception.
    """
    try:
        lst = client.get_list(list_id)
        card = lst.add_card(name, description)
        return f"Card '{card.name}' created with ID: {card.id}"
    except Exception as e:
        logger.error(f"Failed to create card: {e}")
        return "Failed to create card. "


# Update a card's name, description, and optionally move it to a different list
def update_card(
    card_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    list_id: Optional[str] = None,
) -> str:
    """
    Updates an existing card's name, description, and/or list.

    Args:
        card_id (str): The ID of the card to be updated.
        name (Optional[str]): The new name for the card (optional).
        description (Optional[str]): The new description for the card (optional).
        list_id (Optional[str]): The ID of the new list to move the card to (optional).

    Returns:
        str: A message indicating whether the card update was successful or not.
    Logs:
        Errors are logged in case of an exception.
    """
    try:
        card = client.get_card(card_id)
        if name:
            card.set_name(name)
        if description:
            card.set_description(description)
        if list_id:
            card.change_list(list_id)
        return f"Card '{card.name}' updated."
    except Exception as e:
        logger.error(f"Failed to update card: {e}")
        return "Failed to update card."


def delete_card(card_id: str) -> str:
    """
    Deletes a specified card by its ID.

    Args:
        card_id (str): The ID of the card to be deleted.

    Returns:
        str: A message indicating whether the card deletion was successful or not.
    Logs:
        Errors are logged in case of an exception.
    """
    try:
        card = client.get_card(card_id)
        card.delete()
        return f"Card with ID {card_id} deleted."
    except Exception as e:
        logger.error(f"Failed to delete card: {e}")
        return "Failed to delete card."


def search_cards(query: str) -> List[str]:
    """
    Searches for cards across Trello using a text query.

    Args:
        query (str): The search query to look for matching cards.

    Returns:
        List[str]: A list of strings containing card names and their IDs.
                   Format: "<card name> (ID: <card id>)"
                   Returns an empty list if no matches are found or if the operation fails.
    Logs:
        Errors are logged in case of an exception.
    """
    try:
        cards = client.search(query, models=["cards"])
        return [f"{card['name']} (ID: {card['id']})" for card in cards]
    except Exception as e:
        logger.error(f"Failed to search cards: {e}")
        return []

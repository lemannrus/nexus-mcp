"""
This module provides helper functions to interact with the Google Calendar API.

The functions in this module allow users to perform various operations on Google Calendar, such as:
- Fetching calendar service credentials
- Creating events
- Listing events
- Updating existing events
- Deleting events

The module also utilizes a logger to log errors and notable actions for debugging and monitoring.
"""

import datetime
import logging
from pathlib import Path
from typing import Optional, List
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from google.auth.transport.requests import Request
from config import CREDENTIALS_PATH, TOKEN_PATH, SCOPES

logger = logging.getLogger(__name__)


def get_calendar_service() -> Resource:
    """
    Establish and return an authenticated Google Calendar API service instance.

    This function handles the authentication flow, including:
    - Loading saved user credentials if available
    - Refreshing expired tokens
    - Performing the OAuth flow if no valid token is found

    Returns:
        Resource: An authorized instance of the Google Calendar API service.

    Raises:
        Exception: If the authentication or service construction fails.
    """
    creds = None
    try:
        if Path(TOKEN_PATH).exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(TOKEN_PATH, 'w') as token:
                token.write(creds.to_json())
        return build('calendar', 'v3', credentials=creds)
    except Exception as e:
        logger.error(f"Failed to get calendar service: {e}")
        raise


def create_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: Optional[str] = None,
    location: Optional[str] = None
) -> str:
    """
    Create a new event in the primary Google Calendar.

    Args:
        summary (str): The title of the event.
        start_time (str): Start time in ISO 8601 format (e.g., '2025-04-15T10:00:00').
        end_time (str): End time in ISO 8601 format.
        description (Optional[str]): Description text for the event.
        location (Optional[str]): Optional physical or virtual location.

    Returns:
        str: A confirmation message with a link to the created event, or an error message.
    """
    try:
        service = get_calendar_service()
        event = {
            'summary': summary,
            'location': location,
            'description': description,
            'start': {
                'dateTime': start_time,
                'timeZone': 'Europe/Kyiv',
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'Europe/Kyiv',
            },
        }
        event = service.events().insert(calendarId='primary', body=event).execute()
        return f"Event created: {event.get('htmlLink')}"
    except Exception as e:
        logger.error(f"Failed to create event: {e}")
        return f"Failed to create event. {e}"


def list_events(max_results: int = 10) -> List[str]:
    """
    Retrieve a list of upcoming events from the primary calendar.

    Args:
        max_results (int): The maximum number of events to return.

    Returns:
        List[str]: A list of formatted strings representing event start times and summaries.
    """
    try:
        service = get_calendar_service()
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        event_list = [
            f"{event['start'].get('dateTime', event['start'].get('date'))} - {event.get('summary')}"
            for event in events
        ]
        return event_list
    except Exception as e:
        logger.error(f"Failed to list events: {e}")
        return []


def update_event(
    event_id: str,
    summary: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None
) -> str:
    """
    Update an existing event in the primary calendar.

    Args:
        event_id (str): The ID of the event to update.
        summary (Optional[str]): Updated title of the event.
        start_time (Optional[str]): Updated start time (ISO 8601 format).
        end_time (Optional[str]): Updated end time (ISO 8601 format).
        description (Optional[str]): Updated event description.
        location (Optional[str]): Updated event location.

    Returns:
        str: A message indicating success or failure.
    """
    try:
        service = get_calendar_service()
        event = service.events().get(calendarId='primary', eventId=event_id).execute()

        if summary:
            event['summary'] = summary
        if start_time:
            event['start']['dateTime'] = start_time
        if end_time:
            event['end']['dateTime'] = end_time
        if description:
            event['description'] = description
        if location:
            event['location'] = location

        updated_event = service.events().update(
            calendarId='primary',
            eventId=event_id,
            body=event
        ).execute()

        return f"Event updated: {updated_event.get('htmlLink')}"
    except Exception as e:
        logger.error(f"Failed to update event: {e}")
        return "Failed to update event."


def delete_event(event_id: str) -> str:
    """
    Delete an event from the primary calendar.

    Args:
        event_id (str): The ID of the event to delete.

    Returns:
        str: A message indicating whether the event was successfully deleted.
    """
    try:
        service = get_calendar_service()
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        return "Event deleted."
    except Exception as e:
        logger.error(f"Failed to delete event: {e}")
        return "Failed to delete event."

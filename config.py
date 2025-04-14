import os
from pathlib import Path

CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH")
TOKEN_PATH = os.getenv("TOKEN_PATH")
VAULT_PATH = Path(os.getenv("VAULT_PATH"))
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
SCOPES = ['https://www.googleapis.com/auth/calendar']
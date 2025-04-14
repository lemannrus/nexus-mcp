# Personal Assistant MCP Server

A powerful personal assistant server that integrates with various services including Google Calendar, Obsidian Vault, Trello, and web page parsing capabilities. Built using FastMCP, this server provides a unified interface for managing your digital life.

## Features

- **Google Calendar Integration**
  - Create, read, update, and delete calendar events
  - List upcoming events

- **Obsidian Vault Management**
  - Create, read, update, and delete notes
  - Full-text search in notes
  - Folder management (create, delete, search, list)

- **Trello Integration**
  - Board, list, and card management
  - Create, update, and delete cards
  - Search cards by text query

- **Web Page Parsing**
  - Extract and clean HTML content from any URL

## Prerequisites

- Python 3.10 or higher
- Poetry (Python package manager)
- Google Calendar API credentials
- Trello API credentials (if using Trello features)
- Obsidian Vault (if using Obsidian features)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/personal-assistant-mcp.git
   cd personal-assistant-mcp
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up Google Calendar API:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Google Calendar API
   - Create OAuth 2.0 credentials
   - Download the credentials and save them as `credentials.json` in the project root

4. Set up Trello API (optional):
   - Go to [Trello Developer Portal](https://trello.com/app-key)
   - Get your API key and token
   - Add them to your environment variables or config file

## Configuration

1. Run the initial setup to authenticate with Google Calendar:
   ```bash
   poetry run python main.py
   ```
   - This will open a browser window for Google authentication
   - Follow the prompts to authorize the application

## Usage

1. Start the server:
   ```bash
   poetry run python main.py
   ```

2. The server will start and be ready to accept MCP-compatible client connections.

3. Use any MCP-compatible client to interact with the server. The server provides the following tools:
   - Calendar management
   - Obsidian vault operations
   - Trello board management
   - Web page parsing

## Anthropic Claude Desktop Configuration

To use this server with Anthropic Claude Desktop, add the following configuration to your Claude Desktop settings:

```json
{
   "mcpServers": {
      "personal-assistant": {
         "command": "/path/to/your/venv/bin/python",
         "args": ["/path/to/your/project/main.py"],
         "env": {
                 "CREDENTIALS_PATH": "/path/to/your/credentials.json",
                 "TOKEN_PATH": "/path/to/your/token.json",
                 "VAULT_PATH": "/path/to/your/obsidian/vault",
                 "TRELLO_TOKEN": "your_trello_token",
                 "TRELLO_API_KEY": "your_trello_api_key"
         }
      }
   }
}
```

Replace the paths and credentials with your actual values:
- `command`: Path to your Python virtual environment's Python executable
- `args`: Path to your project's `main.py` file
- `CREDENTIALS_PATH`: Path to your Google Calendar credentials file
- `TOKEN_PATH`: Path to your Google Calendar token file
- `VAULT_PATH`: Path to your Obsidian vault
- `TRELLO_TOKEN`: Your Trello API token
- `TRELLO_API_KEY`: Your Trello API key

## Development

- The project uses Poetry for dependency management
- All tools are registered in `main.py`
- Service-specific implementations are in the `services/` directory
- Follow PEP 8 style guidelines for Python code

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

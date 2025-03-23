# Notion to Qdrant Indexer

This tool incrementally indexes Notion pages into a Qdrant vector database. It uses sentence-transformers for generating multilingual embeddings (including Chinese support), enabling semantic search capabilities over your Notion notes.

## Features

- Multilingual support (including Chinese) using sentence-transformers
- Local embedding generation (no API key required)
- Incremental indexing (only updates changed pages)
- Content hashing for change detection
- Progress bar for indexing status
- Comprehensive error handling and logging
- Support for multiple Notion block types
- Retry mechanism for API calls
- Environment variable configuration

## Prerequisites

- Python 3.8+
- A running Qdrant instance
- Notion API integration token

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` with your credentials:
   - `NOTION_TOKEN`: Your Notion integration token
   - `NOTION_DATABASES`: List of databases to index (format: name:id,name2:id2)
   - `QDRANT_HOST`: Qdrant host (default: localhost)
   - `QDRANT_PORT`: Qdrant port (default: 6333)

## Usage

Run the indexer:
```bash
python notion_indexer.py
```

The script will:
1. Connect to your Notion workspace
2. Fetch all pages from the specified databases
3. Generate embeddings using sentence-transformers locally
4. Store the embeddings and metadata in Qdrant
5. Skip unchanged pages

## Supported Notion Block Types

- Paragraphs
- Headings (H1, H2, H3)
- Bulleted lists
- Numbered lists
- To-do items
- Toggle blocks
- Code blocks

## Error Handling

The script includes comprehensive error handling:
- Retries for API calls with exponential backoff
- Detailed logging of errors and progress
- Graceful handling of missing environment variables
- Individual page processing errors don't stop the entire indexing process

## Logging

Logs are written to stdout with timestamps and log levels. Set the logging level in the script by modifying:
```python
logging.basicConfig(level=logging.INFO)
```

## Contributing

Feel free to submit issues and enhancement requests! 
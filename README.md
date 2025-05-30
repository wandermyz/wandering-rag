# Wandering RAG

A CLI tool for personal RAG that retrieves data from Notion, Obsidian, Apple Notes, etc., stored in Qdrant and exposed as an MCP server. 

(So that Claude Desktop successfully answers my question "When did I adopt my cat and when did I change her cat litter most recently? ")

## Installation

Using uv (recommended):

```bash
uv pip install -e .
```

Run a Qdrant server if you don't have one:

```bash
cd qdrant-docker
docker-compose up -d
```

Copy `.env.example` as `.env` and specify the Markdown folders (or Obsidian vaults)

## Usage

The CLI provides several subcommands:

### Markdown commands

```bash
./wandering-rag md index
```


### Notion commands (WIP)

```bash
./wandering-rag notion index
```

### MCP commands

```bash
./wandering-rag mcp run-server
```

## Configure for Claude Desktop

```
{
    "mcpServers": {
        "wandering-rag": {
            "command": "<your git checkout path>/wandering-rag/wandering-rag",
            "args": ["mcp", "run-server"],
            "env": {
                "PATH": "<your home folder>/.local/bin:/usr/bin:$PATH"
            }
        }
    }
}
```

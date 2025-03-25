# Wandering RAG

A CLI tool for personal RAG that retrieves data from Notion, Obsidian, Apple Notes, etc., stored in Qdrand and exposed as an MCP.

(So that Claude Desktop successfully answers my question "When did I adopt my cat and when did I change her cat litter most recently? ")

## Installation

Using uv (recommended):

```bash
uv pip install -e .
```

## Usage

The CLI provides several subcommands:

### Notion commands

```bash
wandering-rag notion index
```

### Markdown commands

```bash
wandering-rag md index
```

### MCP commands

```bash
wandering-rag mcp run-server
```

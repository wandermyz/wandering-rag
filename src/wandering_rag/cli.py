"""Command-line interface for wandering-rag."""
import click
import logging

from wandering_rag.notion import notion_cli
from wandering_rag.md import md_cli
from wandering_rag.mcp import mcp_cli
from wandering_rag.search import search_cli

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@click.group()
@click.version_option()
def cli():
    """Wandering RAG CLI tool."""
    pass


# Register subcommand groups
cli.add_command(notion_cli)
cli.add_command(md_cli)
cli.add_command(mcp_cli)
cli.add_command(search_cli)


if __name__ == "__main__":
    cli()

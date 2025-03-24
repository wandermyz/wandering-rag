"""Command-line interface for wandering-rag."""
import click

from wandering_rag.notion import notion_cli
from wandering_rag.md import md_cli
from wandering_rag.mcp import mcp_cli


@click.group()
@click.version_option()
def cli():
    """Wandering RAG CLI tool."""
    pass


# Register subcommand groups
cli.add_command(notion_cli)
cli.add_command(md_cli)
cli.add_command(mcp_cli)


if __name__ == "__main__":
    cli()

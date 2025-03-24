"""Command implementations for Markdown module."""
import click
from wandering_rag.vector_store import QdrantStore


@click.group(name="md")
def md_cli():
    """Markdown commands."""
    pass


@md_cli.command("index")
def index():
    """Index markdown documents."""
    click.echo("Indexing markdown documents...")
    # Implement markdown indexing functionality here

    # BAAI/bge-large-zh-v1.5
    vector_size = 1024
    qd = QdrantStore(vector_size=vector_size)
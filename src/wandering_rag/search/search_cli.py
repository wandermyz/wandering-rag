import asyncio
import click
from rich.console import Console
from rich.table import Table
from rich import box
from typing import List, Dict
from wandering_rag.embeddings.factory import create_embedding_provider
from wandering_rag.vector_store import QdrantStore
from wandering_rag.embeddings.base import EmbeddingProvider
from wandering_rag.vector_store.vector_doc import VectorDoc

async def get_embedding(embedding_provider, text: str) -> List[float]:
    """
    Get embeddings for text 
    
    Args:
        text: Text to embed
        
    Returns:
        List of embedding values
    """
    res = await embedding_provider.embed_query(text)
    return res

def display_results(results: VectorDoc):
    """Display search results in a formatted table."""
    console = Console()
    
    if not results:
        console.print("\n[yellow]No matching results found.[/yellow]")
        return
    
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        title="Search Results"
    )
    
    table.add_column("Score", justify="right", style="cyan", width=8)
    table.add_column("Title", style="blue")
    table.add_column("Preview", width=50)
    table.add_column("URL", style="bright_blue")  # No width limit to show full URLs
    
    for result in results:
        url_encoded = result.payload.doc_url.replace(" ", "%20")
        table.add_row(
            f"{result.score:.3f}",
            result.payload.title,
            result.payload.content[:100],
            f'[link={url_encoded}]Source[/link]'
        )
    
    console.print("\n")
    console.print(table)
    console.print("\n")

async def single_search(store:QdrantStore, embedding_provider:EmbeddingProvider, query: str, limit: int = 5, threshold: float = 0.6):
    embeddings = await get_embedding(embedding_provider, query)
    results = store.search(embeddings, limit, threshold)
    display_results(results)

async def interactive_search(store:QdrantStore, embedding_provider:EmbeddingProvider, limit: int = 5, threshold: float = 0.6):
    """
    Run interactive search loop.
    
    Args:
        searcher: ObsidianSearcher instance
        limit: Number of results to return (from command line args)
        threshold: Minimum similarity score (from command line args)
    """
    console = Console()
    print("\nEnter your search queries (press Ctrl+C or type 'q' to quit)")
    
    while True:
        try:
            # Get search query using standard input
            query = input("\n[Search]: ").strip()
            if not query or query.lower() in ('q', 'quit', 'exit'):
                break
            
            # Perform search with command line argument values
            embeddings = await get_embedding(embedding_provider, query)
            results = store.search(embeddings, limit, threshold)
            display_results(results)
            
        except KeyboardInterrupt:
            console.print("\nExiting...")
            break
        except ValueError as e:
            console.print_exception()
        except Exception as e:
            console.print_exception()
            break

@click.command(name="search")
@click.argument("query", required=False)
@click.option("--limit", type=int, default=5)
@click.option("--threshold", type=float, default=0.6)
def search_cli(query, limit, threshold):
    """Search command
    
    QUERY: Search query (if not provided, runs in interactive mode)
    LIMIT: Maximum number of results
    THRESHOLD: Minimum similarity score
    """

    embedding_provider = create_embedding_provider()
    qd = QdrantStore(vector_size=embedding_provider.get_dimension())

    if query:
        # Single search mode
        asyncio.run(single_search(qd, embedding_provider, query, limit, threshold))
    else:
        # Interactive mode with command line args as defaults
        asyncio.run(interactive_search(qd, embedding_provider, limit, threshold))
"""
Search Notion Notes in Qdrant

This script allows searching through indexed Notion pages using semantic search.
It uses the same sentence-transformers model as the indexer for consistency.
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os
import argparse
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich import box
import sys

def load_environment():
    """Load environment variables."""
    load_dotenv()
    
    required_vars = ["QDRANT_HOST", "QDRANT_PORT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {
        "qdrant_host": os.getenv("QDRANT_HOST"),
        "qdrant_port": int(os.getenv("QDRANT_PORT"))
    }

class NotionSearcher:
    def __init__(self):
        """Initialize the searcher with model and client."""
        env = load_environment()
        print("Loading embedding model (this may take a few seconds)...")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.qdrant = QdrantClient(host=env["qdrant_host"], port=env["qdrant_port"])
        print("Ready for search!")

    def search_notes(self, query: str, limit: int = 5, score_threshold: float = 0.6) -> List[Dict]:
        """
        Search for Notion pages similar to the query.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1) for results
            
        Returns:
            List of matching pages with their similarity scores
        """
        # Generate embedding for the query
        query_embedding = self.model.encode(query).tolist()
        
        # Search in Qdrant
        search_results = self.qdrant.search(
            collection_name="notion_notes",
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                "score": result.score,
                "title": result.payload["title"],
                "database": result.payload["database_name"],
                "url": result.payload["url"],
                "content": result.payload["content"][:200] + "..."  # Preview of content
            }
            for result in search_results
        ]

def display_results(results: List[Dict]):
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
    table.add_column("Database", style="green", width=15)
    table.add_column("Title", style="blue")
    table.add_column("Preview", width=50)
    table.add_column("URL", style="bright_blue")  # No width limit to show full URLs
    
    for result in results:
        table.add_row(
            f"{result['score']:.3f}",
            result['database'],
            result['title'],
            result['content'],
            result['url']
        )
    
    console.print("\n")
    console.print(table)
    console.print("\n")

def interactive_search(searcher: NotionSearcher, limit: int = 5, threshold: float = 0.6):
    """
    Run interactive search loop.
    
    Args:
        searcher: NotionSearcher instance
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
            results = searcher.search_notes(query, limit, threshold)
            display_results(results)
            
        except KeyboardInterrupt:
            console.print("\nExiting...")
            break
        except ValueError as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
        except Exception as e:
            console.print(f"\n[red]Unexpected error:[/red] {str(e)}")
            break

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Search through indexed Notion pages")
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (if not provided, runs in interactive mode)"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.6,
        help="Minimum similarity score 0-1 (default: 0.6)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize searcher (loads model once)
        searcher = NotionSearcher()
        
        if args.query:
            # Single search mode
            results = searcher.search_notes(args.query, args.limit, args.threshold)
            display_results(results)
        else:
            # Interactive mode with command line args as defaults
            interactive_search(searcher, args.limit, args.threshold)
            
    except Exception as e:
        print(f"Error during search: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
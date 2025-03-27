"""Command implementations for Notion module."""
import os
import logging
from dotenv import load_dotenv
from notion_client import Client
from wandering_rag.embeddings.factory import create_embedding_provider
from wandering_rag.vector_store.qdrant_store import QdrantStore
import click

logger = logging.getLogger(__name__)

class NotionQdrantIndexer:
    def __init__(self):
        """
        Initialize the NotionQdrantIndexer.
        """
        load_dotenv()
        
        # Required environment variables
        required_vars = ["NOTION_TOKEN"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Initialize Notion client
        logger.info("Initializing Notion client...")
        self.notion = Client(auth=os.environ["NOTION_TOKEN"])
        logger.info("Notion client initialized")

        # Load embeddings
        logger.info("Loading embeddings...")
        self.embeddings = create_embedding_provider()
        vector_size = self.embeddings.get_dimension()
        logger.info(f"Embeddings loaded. dim={vector_size}")

        # Connect to Qdrant
        self.qd = QdrantStore(vector_size=vector_size)

    def _process_page(self, page_id: str):
        """
        Process a single page and all its blocks.
        
        Args:
            page_id: The ID of the page to process
        """
        try:
            # Get page details to get the title
            page = self.notion.pages.retrieve(page_id=page_id)
            page_properties = page.get("properties", {})
            page_title_block = page_properties.get("Name", None) or page_properties.get("title", [{}])
            page_title = page_title_block.get("title", [{}])[0].get("plain_text", "<Untitled>")
            logger.info(f"Processing page: {page_title} (ID: {page_id})")
            
            # Get all blocks from the page
            blocks = []
            results = self.notion.blocks.children.list(block_id=page_id)
            blocks.extend(results["results"])
            
            # Handle pagination
            while results.get("has_more"):
                results = self.notion.blocks.children.list(
                    block_id=page_id,
                    start_cursor=results["next_cursor"]
                )
                blocks.extend(results["results"])
            
            logger.info(f"Found {len(blocks)} blocks in page '{page_title}'")
            
            # TODO: Process blocks into chunks and index them
            # This will be similar to markdown processing
            
        except Exception as e:
            logger.exception(f"Error processing page content {page_id}: {str(e)}")

    def index_notes(self):
        """
        Traverse all notes in Notion and index them into Qdrant.
        """
        logger.info("Searching for all Notion pages...")
        
        try:
            # Search for all pages
            query = self.notion.search(filter={"property": "object", "value": "page"})
            
            def _process_search_results(results):
                """Process a batch of search results."""
                for item in results:
                    try:
                        self._process_page(item["id"])
                    except Exception as e:
                        logger.error(f"Error processing page {item['id']}: {str(e)}")
                        continue

            # Process initial results
            _process_search_results(query["results"])

            # Handle pagination and process items as we get them
            while query.get("has_more"):
                query = self.notion.search(
                    filter={"property": "object", "value": "page"},
                    start_cursor=query["next_cursor"]
                )
                self._process_search_results(query["results"])

        except Exception as e:
            logger.error(f"Error during Notion search: {str(e)}")
            raise




@click.group(name="notion")
def notion_cli():
    """Notion commands."""
    pass


@notion_cli.command("index")
def index():
    """Index Notion documents."""

    indexer = NotionQdrantIndexer()
    indexer.index_notes()

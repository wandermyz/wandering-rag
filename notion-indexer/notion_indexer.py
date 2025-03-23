"""
Notion to Qdrant Indexer

This script incrementally indexes Notion pages into Qdrant vector database.
It uses sentence-transformers for multilingual embeddings.
"""

from notion_client import Client
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from typing import Dict, List, NamedTuple
import hashlib
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NotionDatabase(NamedTuple):
    """Represents a Notion database configuration"""
    name: str
    id: str

class NotionQdrantIndexer:
    def __init__(self, load_env: bool = True):
        """
        Initialize the NotionQdrantIndexer.
        
        Args:
            load_env: Whether to load environment variables from .env file
        """
        if load_env:
            load_dotenv("../.env")
        
        # Required environment variables
        required_vars = ["NOTION_TOKEN", "QDRANT_HOST", "QDRANT_PORT", "NOTION_DATABASES"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.notion = Client(auth=os.environ["NOTION_TOKEN"])
        self.qdrant = QdrantClient(
            host=os.environ["QDRANT_HOST"],
            port=int(os.environ["QDRANT_PORT"])
        )
        
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
        self.collection_name = "notion_notes"
        self.embedding_dimension = 768  # paraphrase-multilingual-mpnet-base-v2 dimension
        self._ensure_collection_exists()
        
        # Parse database configurations
        self.databases = self._parse_database_configs()
        
    def _parse_database_configs(self) -> List[NotionDatabase]:
        """Parse database configurations from environment variables."""
        databases_str = os.environ["NOTION_DATABASES"]
        databases = []
        
        for db_str in databases_str.split(','):
            db_str = db_str.strip()
            if not db_str:
                continue
            
            try:
                name, db_id = db_str.split(':')
                databases.append(NotionDatabase(name=name.strip(), id=db_id.strip()))
            except ValueError:
                logger.warning(f"Invalid database configuration: {db_str}")
                continue
        
        if not databases:
            raise ValueError("No valid database configurations found in NOTION_DATABASES")
        
        return databases

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            self.qdrant.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception:
            logger.info(f"Creating collection '{self.collection_name}'")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension,
                    distance=models.Distance.COSINE
                )
            )

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for text using sentence-transformers model.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        return self.model.encode(text).tolist()

    def _get_content_hash(self, content: str) -> str:
        """
        Generate a hash of the content for change detection.
        
        Args:
            content: Text content to hash
            
        Returns:
            MD5 hash of the content
        """
        return hashlib.md5(content.encode()).hexdigest()

    def _process_page(self, page_id: str, database: NotionDatabase) -> Dict:
        """
        Process a single Notion page.
        
        Args:
            page_id: Notion page ID
            database: Source database configuration
            
        Returns:
            Dictionary containing processed page information
        """
        page = self.notion.pages.retrieve(page_id)
        blocks = self.notion.blocks.children.list(page_id)
        
        content = self._extract_text_from_blocks(blocks)
        
        return {
            "page_id": page_id,
            "database_name": database.name,
            "database_id": database.id,
            "title": self._extract_title(page["properties"]),
            "tags": self._extract_tags(page["properties"]),
            "content": content,
            "last_edited": page["last_edited_time"],
            "content_hash": self._get_content_hash(content),
            "url": page["url"]
        }

    def _extract_title(self, page_properties):
        name = page_properties.get('Name', {})
        title = name.get('title', [])
        plain_text = title[0].get('plain_text', "") if len(title) > 0 else ""
        return plain_text

    def _extract_tags(self, page_properties):
        tags = page_properties.get('Tags', {})
        multi_select = tags.get('multi_select', [])
        return [ item.get('name', '') for item in multi_select ]

    def _extract_text_from_blocks(self, blocks) -> str:
        """Extract text content from Notion blocks."""
        text = []
        for block in blocks["results"]:
            block_type = block["type"]
            
            if block_type == "paragraph":
                text.extend([t.get("plain_text", "") for t in block["paragraph"]["rich_text"]])
            elif block_type == "heading_1":
                text.extend([t.get("plain_text", "") for t in block["heading_1"]["rich_text"]])
            elif block_type == "heading_2":
                text.extend([t.get("plain_text", "") for t in block["heading_2"]["rich_text"]])
            elif block_type == "heading_3":
                text.extend([t.get("plain_text", "") for t in block["heading_3"]["rich_text"]])
            elif block_type == "bulleted_list_item":
                text.extend([t.get("plain_text", "") for t in block["bulleted_list_item"]["rich_text"]])
            elif block_type == "numbered_list_item":
                text.extend([t.get("plain_text", "") for t in block["numbered_list_item"]["rich_text"]])
            elif block_type == "to_do":
                text.extend([t.get("plain_text", "") for t in block["to_do"]["rich_text"]])
            elif block_type == "toggle":
                text.extend([t.get("plain_text", "") for t in block["toggle"]["rich_text"]])
            elif block_type == "code":
                text.extend([t.get("plain_text", "") for t in block["code"]["rich_text"]])
            
        return " ".join(text)

    def index_database(self, database: NotionDatabase):
        """
        Incrementally index a single Notion database to Qdrant.
        
        Args:
            database: Database configuration to index
        """
        logger.info(f"Indexing database: {database.name} ({database.id})")
        
        # Get all pages from the database
        pages = self.notion.databases.query(database.id)
        
        # Process pages with progress bar
        for page in tqdm(pages["results"], desc=f"Processing {database.name}"):
            try:
                page_id = page["id"]
                
                # Check if page exists in Qdrant
                search_results = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="page_id",
                                match=models.MatchValue(value=page_id)
                            )
                        ]
                    )
                )
                
                # Process page content
                processed_page = self._process_page(page_id, database)
                
                if not search_results[0]:  # Page doesn't exist
                    logger.info(f"Adding new page: {processed_page['title']}")
                    embedding = self._get_embedding(processed_page["content"])
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=[
                            models.PointStruct(
                                id=page_id,
                                vector=embedding,
                                payload=processed_page
                            )
                        ]
                    )
                else:
                    # Check if content has changed
                    existing_point = search_results[0][0]
                    if existing_point.payload["content_hash"] != processed_page["content_hash"]:
                        logger.info(f"Updating changed page: {processed_page['title']}")
                        embedding = self._get_embedding(processed_page["content"])
                        self.qdrant.upsert(
                            collection_name=self.collection_name,
                            points=[
                                models.PointStruct(
                                    id=page_id,
                                    vector=embedding,
                                    payload=processed_page
                                )
                            ]
                        )
                    else:
                        logger.debug(f"Skipping unchanged page: {processed_page['title']}")
                        
            except Exception as e:
                logger.error(f"Error processing page {page['id']}: {str(e)}")

    def index_all_databases(self):
        """Index all configured databases."""
        for database in self.databases:
            try:
                self.index_database(database)
            except Exception as e:
                logger.error(f"Error indexing database {database.name}: {str(e)}")
                continue

def main():
    """Main entry point"""
    try:
        indexer = NotionQdrantIndexer()
        indexer.index_all_databases()
        logger.info("Indexing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
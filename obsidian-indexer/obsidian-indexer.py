"""
Obsidian to Qdrant Indexer
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PayloadSchemaType
import os
from typing import Dict, List, NamedTuple
import hashlib
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from pathlib import Path
import uuid
import frontmatter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObsidianQdrantIndexer:
    def __init__(self, load_env: bool = True):
        """
        Initialize the NotionQdrantIndexer.
        
        Args:
            load_env: Whether to load environment variables from .env file
        """
        if load_env:
            load_dotenv("../.env")
        
        # Required environment variables
        required_vars = ["QDRANT_HOST", "QDRANT_PORT", "OBSIDIAN_VAULTS"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.qdrant = QdrantClient(
            host=os.environ["QDRANT_HOST"],
            port=int(os.environ["QDRANT_PORT"])
        )
        
        self.collection_name = "obsidian-notes"
        self.embedding_dimension = 768  # paraphrase-multilingual-mpnet-base-v2 dimension
        self._ensure_collection_exists()
        
        # Parse database configurations
        self.vaults = self._parse_vaults_configs()

        # Initialize the sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
    def _parse_vaults_configs(self) -> List[str]:
        """Parse vaults paths from environment variables."""
        vaults_str = os.environ["OBSIDIAN_VAULTS"]
        vaults = []
        
        for vault_str in vaults_str.split(','):
            vault_str = os.path.expandvars(os.path.expanduser(vault_str.strip()))
            if not vault_str:
                continue
            
            if os.path.exists(vault_str):
                vaults.append(vault_str)
            else: 
                raise Exception("Obsidian vault folder not found: " + vault_str)
        
        return vaults

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

        self.qdrant.create_payload_index(
            collection_name=self.collection_name,
            field_name="note_id",  
            field_schema=PayloadSchemaType.KEYWORD  
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

    def _standardize_metadata(self, fm: frontmatter.Post) -> dict:
        res = {}

        for key, value in fm.metadata.items():
            if key == "Created at":
                res["created_at"] = value
            elif key == "Last updated at":
                res["last_edited"] = value
            elif key == "tags":
                res["tags"] = value
            elif key == "Source URL":
                res['url'] = value
            else:
                res['key'] = value
        return res

    def _process_note(self, md_file: Path, vault: Path) -> dict:
        relative_path = md_file.relative_to(vault)
        title = md_file.stem
        folder = relative_path.parent

        fm = frontmatter.load(md_file)
        properties = self._standardize_metadata(fm)

        note_id = f"{vault.stem}:{relative_path}"
        
        return properties | {
            "note_id": note_id,
            "vault_name": vault.stem,
            "folder": folder,
            "title": title,
            "content": fm.content,
            "content_hash": self._get_content_hash(fm.content),
        } 


    def index_vault(self, vault: str):
        """
        Incrementally index a single Notion database to Qdrant.
        
        Args:
            database: Database configuration to index
        """
        logger.info(f"Indexing vault: {vault}")
        
        # Get all Markdown files from the vault
        vault = Path(vault)
        files = vault.rglob("*.md")
        
        # Process pages with progress bar
        for md_file in tqdm(files, desc=f"Processing {vault}"):
            try:
                logger.info(f"Parsing note: {md_file}")
                processed_note = self._process_note(md_file, vault)
                note_id = processed_note['note_id']

                # Check if file exists in Qdrant
                search_results = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="note_id",
                                match=models.MatchValue(value=note_id)
                            )
                        ]
                    )
                )

                if not search_results[0]:  # File doesn't exist
                    logger.info(f"Adding new note: {note_id}")
                    embedding = self._get_embedding(processed_note['content'])
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=[
                            models.PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embedding,
                                payload=processed_note
                            )
                        ]
                    )
                else:
                    # Check if content has changed
                    existing_point = search_results[0][0]
                    if existing_point.payload["content_hash"] != processed_note["content_hash"]:
                        logger.info(f"Updating changed note: {note_id}")
                        embedding = self._get_embedding(processed_note["content"])
                        self.qdrant.upsert(
                            collection_name=self.collection_name,
                            points=[
                                models.PointStruct(
                                    id=existing_point.id,
                                    vector=embedding,
                                    payload=processed_note
                                )
                            ]
                        )
                    else:
                        logger.debug(f"Skipping unchanged note: {note_id}")
                        
            except Exception as e:
                logger.exception(f"Error processing file {md_file}: {str(e)}", )

    def index_all_vaults(self):
        """Index all configured vaults."""
        for vault in self.vaults:
            try:
                self.index_vault(vault)
            except Exception as e:
                logger.error(f"Error indexing vault {vault}: {str(e)}")
                continue


def main():
    """Main entry point"""
    try:
        indexer = ObsidianQdrantIndexer()
        indexer.index_all_vaults()
        logger.info("Indexing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
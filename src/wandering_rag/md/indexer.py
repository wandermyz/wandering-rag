"""Command implementations for Markdown module."""
import asyncio
import click
import os
import hashlib
import frontmatter
import uuid
import logging
from tqdm import tqdm
from typing import List
from pathlib import Path
from wandering_rag.vector_store import QdrantStore, VectorDoc, VectorDocPayload
from wandering_rag.embeddings.factory import create_embedding_provider
from dotenv import load_dotenv
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter

from wandering_rag.vector_store.vector_doc import VectorDocSourceType

logger = logging.getLogger(__name__)

class MarkdownQdrantIndexer:
    def __init__(self):
        """
        Initialize the MarkdownQdrantIndexer.
        """
        load_dotenv()
        
        # Required environment variables
        required_vars = ["MARKDOWN_FOLDERS"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Load embeddings
        logger.info("Loading embeddings...")
        self.embeddings = create_embedding_provider()
        vector_size = self.embeddings.get_dimension()
        logger.info(f"Embeddings loaded. dim={vector_size}")

        # Connect to Qdrant
        self.qd = QdrantStore(vector_size=vector_size)

        # Parse folder configurations
        self.folders = self._parse_folder_configs()

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Full-width comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Full-width full stop
                "\u3002",  # Ideographic full stop
                "",
            ]
        )

    def _parse_folder_configs(self) -> List[str]:
        """Parse folder paths from environment variables."""
        folders_str = os.environ["MARKDOWN_FOLDERS"]
        folders = []
        
        for folder_str in folders_str.split(','):
            folder_str = os.path.expandvars(os.path.expanduser(folder_str.strip()))
            if not folder_str:
                continue
            
            if os.path.exists(folder_str):
                folders.append(folder_str)
            else: 
                raise Exception("Folder not found: " + folder_str)
        
        return folders

    async def _get_embedding(self, documents: str) -> List[float]:
        """
        Get embeddings for text using sentence-transformers model.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        res = await self.embeddings.embed_documents(documents)
        return res

    def _get_content_hash(self, content: str) -> str:
        """
        Generate a hash of the content for change detection.
        
        Args:
            content: Text content to hash
            
        Returns:
            MD5 hash of the content
        """
        return hashlib.md5(content.encode()).hexdigest()

    def _standardize_metadata(self, fm: frontmatter.Post, doc: VectorDocPayload) -> None:
        for key, value in fm.metadata.items():
            if key == "Created at":
                try:
                    # Try to parse the date string to a datetime object
                    doc.payload.created_at = datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    # If parsing fails, store the original value
                    doc.payload.created_at = value
            elif key == "Last updated at":
                try:
                    # Try to parse the date string to a datetime object
                    doc.payload.last_modified_at = datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    # If parsing fails, store the original value
                    doc.payload.last_modified_at = value
            elif key == "tags":
                doc.payload.tags = value
            elif key == "Source URL":
                doc.payload.source_url = value
            else:
                doc.payload.extra_data[key] = value

    def _process_note(self, md_file: Path, root_folder: Path) -> List[VectorDoc]:
        relative_path = md_file.relative_to(root_folder)
        root = root_folder.stem
        title = md_file.stem
        subfolder = relative_path.parent

        fm = frontmatter.load(md_file)

        # Split content into chunks
        chunks = self.text_splitter.split_text(fm.content)
        docs = []

        for i, chunk in enumerate(chunks):
            doc = VectorDoc(source=VectorDocSourceType.Markdown)
            doc.payload.doc_id = f"{root}/{relative_path}"
            doc.payload.title = f"{title}"
            doc.payload.content = chunk
            doc.payload.content_hash = self._get_content_hash(chunk)
            doc.payload.doc_url = f"obsidian://open?vault={root}&file={relative_path}"
            doc.payload.chunk_index = i
            doc.payload.total_chunks = len(chunks)
            doc.payload.extra_data = {
                "root": root,
                "subfolder": subfolder,
            }

            self._standardize_metadata(fm, doc)

            if doc.payload.created_at is None:
                doc.payload.created_at = datetime.fromtimestamp(md_file.stat().st_ctime)
            if doc.payload.last_modified_at is None:
                doc.payload.last_modified_at = datetime.fromtimestamp(md_file.stat().st_mtime)

            docs.append(doc)

        return docs

    async def index_folder(self, folder: str):
        """
        Incrementally index a single markdown to Qdrant.
        
        Args:
            folder: folder to index
        """
        logger.info(f"Indexing folder: {folder}")
        
        # Get all Markdown files from the vault
        folder = Path(folder)
        files = folder.rglob("*.md")
        
        # Process pages with progress bar
        for md_file in tqdm(files, desc=f"Processing {folder}"):
            try:
                logger.info(f"Parsing note: {md_file}")
                docs = self._process_note(md_file, folder)
                
                # Prepare list for batch processing
                docs_to_upsert = []
                
                # First pass - check which docs need processing
                for doc in docs:
                    doc_id = doc.payload.doc_id
                    existing_point = self.qd.find_point_by_chunk(doc_id, doc.payload.chunk_index)

                    if not existing_point:  # Chunk doesn't exist
                        logger.info(f"Adding new chunk: {doc_id}.{doc.payload.chunk_index}")
                        doc.id = uuid.uuid4()
                        docs_to_upsert.append(doc)
                    else:
                        # Check if content has changed
                        if existing_point.payload["content_hash"] != doc.payload.content_hash:
                            logger.info(f"Updating changed chunk: {doc_id}.{doc.payload.chunk_index}")
                            doc.id = existing_point.id
                            docs_to_upsert.append(doc)
                        else:
                            logger.debug(f"Skipping unchanged chunk: {doc_id}.{doc.payload.chunk_index}")

                # Get embeddings and add vectors in batch
                if docs_to_upsert:
                    contents = [doc.payload.content for doc in docs_to_upsert]
                    vectors = await self._get_embedding(contents)
                    for doc, vector in zip(docs_to_upsert, vectors):
                        doc.vector = vector
                    self.qd.add_vectors(docs_to_upsert)
                        
            except Exception as e:
                logger.exception(f"Error processing file {md_file}: {str(e)}")

    async def index_all_folders(self):
        """Index all configured folders."""
        for folder in self.folders:
            try:
                await self.index_folder(folder)
            except Exception as e:
                logger.exception(f"Error indexing folder {folder}: {str(e)}")
                continue



@click.group(name="md")
def md_cli():
    """Markdown commands."""
    pass

@md_cli.command("index")
def index():
    """Index markdown documents."""

    logger.info("Indexing markdown documents...")

    """Main entry point"""
    try:
        indexer = MarkdownQdrantIndexer()
        asyncio.run(indexer.index_all_folders())
        logger.info("Indexing completed successfully")
        
    except Exception as e:
        logger.exception(f"Error during indexing: {str(e)}")
        raise
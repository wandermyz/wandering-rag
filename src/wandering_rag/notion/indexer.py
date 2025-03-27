"""Command implementations for Notion module."""
import asyncio
import os
import logging
import hashlib
import uuid
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from notion_client import Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from wandering_rag.embeddings.factory import create_embedding_provider
from wandering_rag.vector_store.qdrant_store import QdrantStore
from wandering_rag.vector_store.vector_doc import VectorDoc, VectorDocPayload, VectorDocSourceType
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

    def _extract_text_from_block(self, block: Dict[str, Any]) -> str:
        """
        Extract text content from a Notion block.
        
        Args:
            block: A Notion block object
            
        Returns:
            Extracted text content
        """
        block_type = block.get("type")
        block_content = block.get(block_type, {})
        
        if block_type == "paragraph":
            return "".join(span.get("plain_text", "") for span in block_content.get("rich_text", []))
        elif block_type == "heading_1":
            return "# " + "".join(span.get("plain_text", "") for span in block_content.get("rich_text", []))
        elif block_type == "heading_2":
            return "## " + "".join(span.get("plain_text", "") for span in block_content.get("rich_text", []))
        elif block_type == "heading_3":
            return "### " + "".join(span.get("plain_text", "") for span in block_content.get("rich_text", []))
        elif block_type == "bulleted_list_item":
            return "- " + "".join(span.get("plain_text", "") for span in block_content.get("rich_text", []))
        elif block_type == "numbered_list_item":
            return "1. " + "".join(span.get("plain_text", "") for span in block_content.get("rich_text", []))
        elif block_type == "code":
            return "```" + block_content.get("language", "") + "\n" + "".join(span.get("plain_text", "") for span in block_content.get("rich_text", [])) + "\n```"
        elif block_type == "quote":
            return "> " + "".join(span.get("plain_text", "") for span in block_content.get("rich_text", []))
        elif block_type == "callout":
            color = block_content.get("color", "default")
            return f"::: {color}\n" + "".join(span.get("plain_text", "") for span in block_content.get("rich_text", [])) + "\n:::\n"
        else:
            return ""

    def _get_content_hash(self, content: str) -> str:
        """
        Generate a hash of the content for change detection.
        
        Args:
            content: Text content to hash
            
        Returns:
            MD5 hash of the content
        """
        return hashlib.md5(content.encode()).hexdigest()

    def _enrich_with_metadata(self, doc: VectorDoc) -> None:
        """
        Enrich the document content with metadata.
        For better RAG results
        """
        metadata = []
        if doc.payload.title:
            metadata.append(f'title:{doc.payload.title}')
        
        if doc.payload.extra_data.get("notion_page_id"):
            metadata.append(f'page_id:{doc.payload.extra_data["notion_page_id"]}')

        doc.payload.content = "\n".join(metadata) + "\n\n" + doc.payload.content

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

    async def _process_page(self, page_id: str):
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
            
            # Extract text from all blocks
            text_blocks = []
            for block in blocks:
                text = self._extract_text_from_block(block)
                if text:
                    text_blocks.append(text)
            
            # Join all text blocks with newlines
            full_text = "\n\n".join(text_blocks)
            
            # Split text into chunks using LangChain's text splitter
            chunks = self.text_splitter.split_text(full_text)
            
            # Create VectorDoc objects for each chunk
            docs: List[VectorDoc] = []
            for i, chunk in enumerate(chunks):
                doc = VectorDoc(VectorDocSourceType.Notion)
                doc.payload.doc_id = f"{page_id}"
                doc.payload.title = page_title
                doc.payload.content = chunk
                doc.payload.content_hash = self._get_content_hash(chunk)
                doc.payload.doc_url = f"https://notion.so/{page_id.replace('-', '')}"
                doc.payload.chunk_index = i
                doc.payload.total_chunks = len(chunks)
                doc.payload.created_at = datetime.fromisoformat(page.get("created_time", "").replace("Z", "+00:00"))
                doc.payload.last_modified_at = datetime.fromisoformat(page.get("last_edited_time", "").replace("Z", "+00:00"))
                doc.payload.extra_data = {}
                
                # Enrich content with metadata
                if i == 0:
                    self._enrich_with_metadata(doc)
                
                docs.append(doc)
            
            # Prepare list for batch processing
            docs_to_upsert : List[VectorDoc] = []
            
            # TODO: Redundant as in md indexer. Refactor.
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
            
            logger.info(f"Indexed {len(docs)} chunks from page '{page_title}'")
            
        except Exception as e:
            logger.exception(f"Error processing page content {page_id}: {str(e)}")

    async def index_notes(self):
        """
        Traverse all notes in Notion and index them into Qdrant.
        """
        logger.info("Searching for all Notion pages...")
        
        try:
            # Search for all pages
            query = self.notion.search(filter={"property": "object", "value": "page"})
            
            async def _process_search_results(results):
                """Process a batch of search results."""
                for item in results:
                    try:
                        await self._process_page(item["id"])
                    except Exception as e:
                        logger.error(f"Error processing page {item['id']}: {str(e)}")
                        continue

            # Process initial results
            await _process_search_results(query["results"])

            # Handle pagination and process items as we get them
            while query.get("has_more"):
                query = self.notion.search(
                    filter={"property": "object", "value": "page"},
                    start_cursor=query["next_cursor"]
                )
                await self._process_search_results(query["results"])

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

    try:
        indexer = NotionQdrantIndexer()
        asyncio.run(indexer.index_notes())
        logger.info("Indexing completed successfully")
    except Exception as e:
        logger.exception(f"Error during indexing: {str(e)}")
        raise

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from wandering_rag.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    """
    SentenceTransformer implementation of the embedding provider.
    :param model_name: The name of the SentenceTransformer model to use.
    """

    MAX_WORKERS = 1

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        self.dimension = None
        self.thread_local = threading.local()

    def _encode(self, text: str | List[str]) -> List[List[float]]:
        if not hasattr(self.thread_local, "embedding_model"):
            logging.info(f"Creating embedding model for thread: {threading.get_ident()}")
            self.thread_local.embedding_model = SentenceTransformer(self.model_name)

        return self.thread_local.embedding_model.encode(text)

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since SentenceTransformer is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self._encode,
            documents
        )
        return embeddings.tolist()

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since SentenceTransformer is synchronous
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self.executor,
            self._encode,
            query
        )
        return embedding.tolist()

    def get_dimension(self):
        """
        Return the number of dimensions in the embeddings.
        """
        if self.dimension is None:
            # Get a sample embedding to determine its dimensions
            loop = asyncio.get_event_loop()
            sample_text = "This is a sample text to get embedding dimensions."
            sample_embedding = self._encode(sample_text)
            self.dimension = len(sample_embedding)
        return self.dimension

    def get_model_name(self):
        """
        Return the model name
        """
        return self.model_name

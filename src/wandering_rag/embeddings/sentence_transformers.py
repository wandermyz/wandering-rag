import asyncio
from typing import List

from sentence_transformers import SentenceTransformer

from wandering_rag.embeddings.base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """
    SentenceTransformer implementation of the embedding provider.
    :param model_name: The name of the SentenceTransformer model to use.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since SentenceTransformer is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.embedding_model.encode(documents)
        )
        return embeddings.tolist()

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since SentenceTransformer is synchronous
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.embedding_model.encode(query)
        )
        return embedding.tolist()

    def get_dimension(self):
        """
        Return the number of dimensions in the embeddings.
        """
        # Get a sample embedding to determine its dimensions
        sample_text = "This is a sample text to get embedding dimensions."
        sample_embedding = self.embedding_model.encode(sample_text)
        return len(sample_embedding)
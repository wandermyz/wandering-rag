import asyncio
from typing import List

from fastembed import TextEmbedding

from wandering_rag.embeddings.base import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    """
    FastEmbed implementation of the embedding provider.
    :param model_name: The name of the FastEmbed model to use.
    """

    def __init__(self, model_name: str):
        # [print(x['model']) for x in TextEmbedding.list_supported_models()]
        self.model_name = model_name
        self.embedding_model = TextEmbedding(model_name)

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.passage_embed(documents))
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        return embeddings[0].tolist()

    def get_dimension(self):
        """
        Return the number of dimensions in the embeddings.
        """
        # Get a sample embedding to determine its dimensions
        sample_text = "This is a sample text to get embedding dimensions."
        sample_embedding = list(self.embedding_model.passage_embed([sample_text]))[0]
        return len(sample_embedding)
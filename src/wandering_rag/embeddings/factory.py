from wandering_rag.embeddings.base import EmbeddingProvider
from wandering_rag.embeddings.types import EmbeddingProviderType


def create_embedding_provider(
        provider_type : EmbeddingProviderType = EmbeddingProviderType.SENTENCE_TRANSFORMERS, 
        model_name : str = "BAAI/bge-large-zh-v1.5"
        # provider_type : EmbeddingProviderType = EmbeddingProviderType.FASTEMBED, 
        # model_name : str = "BAAI/bge-small-zh-v1.5"
    ) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if provider_type == EmbeddingProviderType.FASTEMBED:
        from wandering_rag.embeddings.fastembed import FastEmbedProvider
        return FastEmbedProvider(model_name)
    elif provider_type == EmbeddingProviderType.SENTENCE_TRANSFORMERS:
        from wandering_rag.embeddings.sentence_transformers import SentenceTransformerProvider
        return SentenceTransformerProvider(model_name)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")

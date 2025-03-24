from enum import Enum


class EmbeddingProviderType(Enum):
    FASTEMBED = "fastembed"
    SENTENCE_TRANSFORMERS = "sentence-transformers"

import os
import logging
from typing import List, Optional 
from qdrant_client import QdrantClient
from qdrant_client.http import models
import qdrant_client.conversions.common_types as types
from dotenv import load_dotenv
from wandering_rag.vector_store.vector_doc import VectorDoc, VectorDocSourceType

class QdrantStore:
    """A vector store implementation using Qdrant."""
    
    def __init__(self, vector_size: int, collection_name: str = "wandering-rag-docs", ):
        """
        Initialize the QdrantStore.
        
        Args:
            collection_name: Name of the collection to use
            vector_size: Dimensionality of vectors to store
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = self._connect_to_qdrant()
        self.logger = logging.getLogger(__name__)
        self.ensure_collection_exists()
        self.ensure_payload_indexes_exist()
    
    def _connect_to_qdrant(self) -> QdrantClient:
        """Connect to Qdrant using environment variables."""
        # Environment variables loaded from .env
        host = os.environ.get("QDRANT_HOST")
        port = os.environ.get("QDRANT_PORT")
        
        if not host or not port:
            raise ValueError("QDRANT_HOST and QDRANT_PORT must be defined in the environment variables.")
        
        return QdrantClient(host=host, port=int(port))
    
    def ensure_collection_exists(self) -> None:
        """Ensure that the collection exists, create it if it doesn't."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            self.logger.info(f"Created collection: {self.collection_name}")
    
    def ensure_payload_indexes_exist(self) -> None:
        """Ensure that the required payload indexes exist."""
        required_indexes = [
            {"field_name": "doc_id", "field_schema": models.PayloadSchemaType.KEYWORD},
            {"field_name": "source", "field_schema": models.PayloadSchemaType.KEYWORD},
            {"field_name": "tags", "field_schema": models.PayloadSchemaType.KEYWORD},
            {"field_name": "chunk_index", "field_schema": models.PayloadSchemaType.INTEGER},
            {"field_name": "created_at", "field_schema": models.PayloadSchemaType.DATETIME},
            {"field_name": "last_modified_at", "field_schema": models.PayloadSchemaType.DATETIME},
        ]
        
        for index in required_indexes:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=index["field_name"],
                field_schema=index["field_schema"]
            )

    # Add methods for vector operations
    def add_vectors(self, docs : List[VectorDoc]) -> None:
        """
        Add vectors to the collection.
        
        Args:
            vectors: List of vectors to add
            payloads: List of payloads corresponding to each vector
            ids: Optional list of IDs for each vector
        """
        points = []
        for doc in docs:
            point_id = str(doc.id)
            payload = doc.payload.to_dict()
            points.append(models.PointStruct(id=point_id, vector=doc.vector, payload=payload))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def find_point_by_chunk(self, doc_id: str, chunk_index: int) -> models.Record | None:
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id)
                    ),
                    models.FieldCondition(
                        key="chunk_index",
                        match=models.MatchValue(value=chunk_index)
                    )
                ]
            )
        )

        return records[0] if len(records) > 0 else None

    def search(self, query_vector: List[float], limit: int = 10, threshold: float = 0.6, filter: Optional[types.Filter] = None) -> List[VectorDoc]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query vector
            limit: Maximum number of results to return
            threshold: Minimum similarity score threshold
            filter_json: Optional filter condition as a JSON string
        
        Returns:
            List of VectorDoc objects
        """

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter,
            limit=limit,
            score_threshold=threshold,
        )
        
        # Convert search results to VectorDoc objects
        return [
            VectorDoc.from_vector_point(id=result.id, vector=result.vector, score=result.score, payload_dict=result.payload)
            for result in search_results
        ]
    
    def scroll(self, limit: int = 10, filter: Optional[types.Filter] = None) -> List[VectorDoc]:
        """
        Scroll through the collection.
        
        Args:
            limit: Maximum number of results to return
            threshold: Minimum similarity score threshold
            filter: Optional filter condition
        """

        search_results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter,
            limit=limit,
        )

        return [
            VectorDoc.from_vector_point(id=result.id, payload_dict=result.payload)
            for result in search_results
        ]


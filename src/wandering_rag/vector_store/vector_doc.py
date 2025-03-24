from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid
import datetime

class VectorDocSourceType(Enum):
    Markdown = "markdown"
    Notion = "notion"

@dataclass
class VectorDocPayload:
    """
    Payload for a document in the vector store.
    
    Attributes:
        doc_id: Unique identifier for the document
        title: Title of the document
        source: Source type of the document (Markdown, Notion, etc.)
        content: Text content of the document
        content_hash: Hash of the content for change detection
        chunk_index: Index of the chunk if document is split into multiple chunks
        doc_url: URL to access the original document
        source_url: Original URL if document was imported from a web source
        tags: List of tags associated with the document
        created_at: Timestamp when the document was created
        last_modified_at: Timestamp when the document was last modified
        extra_data: Additional metadata as key-value pairs
    """

    doc_id: str
    title: str
    source: VectorDocSourceType
    content: str
    content_hash: str
    chunk_index: int = 0
    total_chunks: int = 1
    doc_url: Optional[str] = None
    source_url: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: Optional[datetime.datetime] = None
    last_modified_at: Optional[datetime.datetime] = None
    extra_data: Optional[Dict[str, Any]] = None

    def __init__(self, source: VectorDocSourceType):
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {k:v for k, v in asdict(self).items() if v is not None}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'VectorDocPayload':
        """
        Create a payload from a dictionary.
        
        Args:
            data: Dictionary containing payload data
            
        Returns:
            A new VectorDocPayload instance populated with the data
        """
        source_type = VectorDocSourceType(data["source"]) if "source" in data and isinstance(data["source"], str) else None
        payload = VectorDocPayload(source_type)
        
        for key, value in data.items():
            if key == "source" and isinstance(value, str):
                payload.source = VectorDocSourceType(value)
            elif key in ["created_at", "last_modified_at"] and value is not None:
                if isinstance(value, str):
                    try:
                        # Convert string to datetime
                        setattr(payload, key, datetime.datetime.fromisoformat(value))
                    except ValueError:
                        # If parsing fails, store the original value
                        setattr(payload, key, value)
                else:
                    # Already a datetime or other type
                    setattr(payload, key, value)
            elif key == "extra_data" and value is None:
                payload.extra_data = {}
            elif key == "tags" and value is None:
                payload.tags = []
            else:
                setattr(payload, key, value)
        
        # Initialize extra_data if not present
        if not hasattr(payload, "extra_data") or payload.extra_data is None:
            payload.extra_data = {}
            
        return payload
@dataclass
class VectorDoc:
    """
    Schema for a doc in the vector store.
    """
    id: uuid.UUID 
    vector: List[float]
    payload: VectorDocPayload
    score: Optional[float] = None

    def __init__(self, source: VectorDocSourceType):
        self.payload = VectorDocPayload(source)

    @staticmethod
    def from_vector_point(id: uuid.UUID, vector: List[float], score: Optional[float] = None, payload_dict: Optional[Dict[str, Any]] = None):
        doc = VectorDoc(payload_dict["source"])
        doc.id = id
        doc.vector = vector
        doc.score = score
        doc.payload = VectorDocPayload.from_dict(payload_dict)
        return doc
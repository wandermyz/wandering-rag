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
    Schema for a doc's payload in the vector store.
    """
    doc_id: str
    title: str
    source: VectorDocSourceType
    content: str
    content_hash: str
    chunk_index: int = 0
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

@dataclass
class VectorDoc:
    """
    Schema for a doc in the vector store.
    """
    id: uuid.UUID 
    vector: List[float]
    payload: VectorDocPayload

    def __init__(self, source: VectorDocSourceType):
        self.payload = VectorDocPayload(source)
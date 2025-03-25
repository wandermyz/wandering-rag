import logging
import uuid
import datetime
import hashlib
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List, Optional
import qdrant_client.conversions.common_types
import qdrant_client.models

from mcp.server import Server
from mcp.server.fastmcp import Context, FastMCP

from wandering_rag.vector_store import QdrantStore, VectorDoc, VectorDocSourceType
from wandering_rag.embeddings.factory import create_embedding_provider, EmbeddingProvider
from wandering_rag.mcp.settings import *

logger = logging.getLogger(__name__)


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:  # noqa
    """
    Context manager to handle the lifespan of the server.
    This is used to configure the embedding provider and Qdrant connector.
    All the configuration is now loaded from the environment variables.
    Settings handle that for us.
    """
    try:
        # Embedding provider is created with a factory function so we can add
        # some more providers in the future. Currently, only FastEmbed is supported.
        embedding_provider = create_embedding_provider()
        qdrant_store = QdrantStore(vector_size=embedding_provider.get_dimension())
        logger.info(
            f"Using embedding provider {type(embedding_provider)} with "
            f"model {embedding_provider.get_model_name()}"
        )

        yield {
            "embedding_provider": embedding_provider,
            "qdrant_store": qdrant_store,
        }
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        pass


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
mcp = FastMCP("wandering-rag", lifespan=server_lifespan, debug=True)


@mcp.tool(name="wandering-rag-store", description=DEFAULT_TOOL_STORE_DESCRIPTION)
async def store(
    ctx: Context,
    information: str,
    # The `metadata` parameter is defined as non-optional, but it can be None.
    # If we set it to be optional, some of the MCP clients, like Cursor, cannot
    # handle the optional parameter correctly.
    metadata: dict[str, Any] = None,
) -> str:
    """
    Store some information in wandering-rag's vector store.
    :param ctx: The context for the request.
    :param information: The information to store.
    :param metadata: JSON metadata to store with the information, optional.
    :return: A message indicating that the information was stored.
    """
    await ctx.debug(f"Storing information {information} in Qdrant")
    qd = ctx.request_context.lifespan_context[
        "qdrant_store"
    ]
    embedding_provider = ctx.request_context.lifespan_context[
        "embedding_provider"
    ]

    doc = VectorDoc(source=VectorDocSourceType.MCP)
    embedding = await embedding_provider.embed_documents([information])
    doc.vector = embedding[0]
    doc.payload.content = information
    doc.payload.content_hash = hashlib.md5(information.encode()).hexdigest()
    doc.payload.title = "MCP Memory"
    doc.payload.doc_id = str(uuid.uuid4())
    doc.payload.created_at = datetime.now()
    doc.payload.last_modified_at = datetime.now()
    doc.payload.tags = ["mcp", "memory"]
    doc.payload.extra_data = metadata
    qd.add_vectors([doc])

    return f"Remembered: {information}"


@mcp.tool(name="wandering-rag-find", description=DEFAULT_TOOL_FIND_DESCRIPTION)
async def find(
    ctx: Context, 
    query: Optional[str] = None, 
    doc_id: Optional[str] = None,
    tag: Optional[str] = None,
    first_chunk_index: Optional[int] = None,
    created_before: Optional[str] = None,
    created_after: Optional[str] = None,
    last_modified_before: Optional[datetime.datetime] = None,
    last_modified_after: Optional[datetime.datetime] = None,
) -> List[str]:
    """
    Find memories in wandering-rag's vector store. 
    :param ctx: The context for the request.
    :param query: The query to use for full-text search. If not provided, the tool will scroll through the collection.
    :param doc_id: Retrieve chunks by its doc_id
    :param any_tags: Retrieve chunks with any of the tags
    :param first_chunk_index: Retrieve chunks starting from this position
    :param created_before: Retrieve chunks created before this timestamp
    :param created_after: Retrieve chunks created after this timestamp
    :param last_modified_before: Retrieve chunks last modified before this timestamp
    :param last_modified_after: Retrieve chunks last modified after this timestamp
    :param filter: A JSON dictionary of freeform filters. 
    :return: A list of entries found.
    """
    await ctx.info(f"Finding results for query {query}")
    logger.info(f"Finding results for query: {query}, filter: {filter} of type {type(filter)}")

    try: 
        qdrant_filter = build_qdrant_filter(
            doc_id=doc_id,
            first_chunk_index=first_chunk_index,
            tag=tag,
            created_before=created_before,
            created_after=created_after,
            last_modified_before=last_modified_before,
            last_modified_after=last_modified_after,
        )

        qd: QdrantStore = ctx.request_context.lifespan_context[
            "qdrant_store"
        ]

        entries = []
        if query:
            embedding_provider: EmbeddingProvider = ctx.request_context.lifespan_context[
                "embedding_provider"
            ]

            embedding = await embedding_provider.embed_query(query)
            entries = qd.search(embedding, limit=DEFAULT_QUERY_LIMIT, threshold=DEFAULT_QUERY_THRESHOLD, filter=qdrant_filter)
        else:
            entries = qd.scroll(limit=DEFAULT_QUERY_LIMIT, filter=qdrant_filter)

        if not entries:
            return [f"No information found for the query '{query}'"]
        content = [
            f"Results for the query '{query}'",
        ]
        for entry in entries:
            # Format the metadata as a JSON string and produce XML-like output
            content.append(
                f"<entry><content>{entry.payload.content}</content><metadata>{entry.get_metadata_json()}</metadata></entry>"
            )
        return content
    except Exception as e:
        logger.exception(e)
        raise e

def build_qdrant_filter(
    doc_id: Optional[str] = None,
    first_chunk_index: Optional[int] = None,
    tag: Optional[str] = None,
    created_before: Optional[datetime.datetime] = None,
    created_after: Optional[datetime.datetime] = None,
    last_modified_before: Optional[datetime.datetime] = None,
    last_modified_after: Optional[datetime.datetime] = None,
) -> qdrant_client.conversions.common_types.Filter:
    """
    Build a Qdrant filter from the given parameters.
    """

    must_conditions = []

    if doc_id:
        must_conditions.append(
            qdrant_client.models.FieldCondition(
                key="doc_id",
                match=qdrant_client.models.MatchValue(value=doc_id)
            )
        )

    if first_chunk_index is not None:
        must_conditions.append(
            qdrant_client.models.FieldCondition(
                key="chunk_index",
                range=qdrant_client.models.Range(gte=first_chunk_index)
            )
        )

    if tag:
        must_conditions.append(
            qdrant_client.models.FieldCondition(
                key="tags",
                match=qdrant_client.models.MatchValue(value=tag)
            )
        )

    if created_before or created_after:
        must_conditions.append(
            qdrant_client.models.FieldCondition(
                key="created_at",
                range=qdrant_client.models.DatetimeRange(
                    gte=created_after if created_after else None,
                    lte=created_before if created_before else None
                )
            )
        )

    if last_modified_before or last_modified_after:
        must_conditions.append(
            qdrant_client.models.FieldCondition(
                key="last_modified_at",
                range=qdrant_client.models.DatetimeRange(
                    gte=last_modified_after if last_modified_after else None,
                    lte=last_modified_before if last_modified_before else None
                )
            )
        )

    if not must_conditions:
        return None

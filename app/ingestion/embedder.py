import uuid
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "masis_documents"

# Qdrant client (Docker internal host)
client = QdrantClient(host="qdrant", port=6333)

# OpenAI embeddings
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)


# =====================================================
# Ensure Collection Exists
# =====================================================


def ensure_collection_exists():
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if "masis_documents" in existing:
        return  # already exists, do nothing

    client.create_collection(
        collection_name="masis_documents",
        vectors_config=...
    )


# =====================================================
# Multimodal Embed & Upsert
# =====================================================

def embed_chunks_single_upsert(chunks, workspace_id, document_id, file_name):

    if not chunks:
        return

    ensure_collection_exists()

    # ---------------------------------------------
    # Extract text for embedding
    # ---------------------------------------------
    texts = []

    for chunk in chunks:
        text = chunk.get("text", "")

        if text and text.strip():
            texts.append(text.strip())
        else:
            texts.append(" ")  # Avoid empty string crash

    # ---------------------------------------------
    # Batch embedding
    # ---------------------------------------------
    batch_size = 64
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)

    # ---------------------------------------------
    # Create Qdrant points
    # ---------------------------------------------
    points = []

    for idx, (vector, chunk) in enumerate(zip(all_vectors, chunks)):

        payload = {
            "workspace_id": workspace_id,
            "document_id": str(document_id),
            "file_name": file_name,
            "chunk_index": idx,
            "chunk_type": chunk.get("chunk_type", "text"),
            "text": chunk.get("text"),
            "structured_data": chunk.get("structured_data"),
        }

        # Optional metadata
        if chunk.get("page_number") is not None:
            payload["page_number"] = chunk.get("page_number")

        if chunk.get("table_index") is not None:
            payload["table_index"] = chunk.get("table_index")

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
        )

    # ---------------------------------------------
    # Atomic upsert
    # ---------------------------------------------
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    logger.info(
        f"Embedded {len(points)} chunks for document {file_name}"
    )
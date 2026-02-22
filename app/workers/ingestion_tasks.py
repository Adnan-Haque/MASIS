from app.workers.celery_app import celery_app
import logging
import zipfile
import io
import os
import uuid

logger = logging.getLogger(__name__)

ENABLE_VISION = os.getenv("ENABLE_VISION", "true") == "true"
EMBED_BATCH_SIZE = 32  # safer than 64


@celery_app.task
def ingest_document(workspace_id, file_name, file_path):

    from app.db.database import SessionLocal
    from app.db.models import Document

    db = SessionLocal()

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        doc = db.query(Document).filter(
            Document.workspace_id == workspace_id,
            Document.file_name == file_name
        ).first()

        if not doc:
            return

        # ================= ZIP HANDLING =================

        if file_name.lower().endswith(".zip"):

            # 🔥 KEEP ZIP DOC (do not delete)
            doc.status = "PROCESSING"
            doc.total_chunks = 0
            doc.processed_chunks = 0
            db.commit()

            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:

                for inner_name in z.namelist():

                    if inner_name.endswith("/"):
                        continue

                    clean_name = inner_name.split("/")[-1]
                    inner_bytes = z.read(inner_name)

                    existing = db.query(Document).filter(
                        Document.workspace_id == workspace_id,
                        Document.file_name == clean_name
                    ).first()

                    if existing:
                        continue

                    child_doc = Document(
                        workspace_id=workspace_id,
                        file_name=clean_name,
                        status="PROCESSING",
                        total_chunks=0,
                        processed_chunks=0
                    )

                    db.add(child_doc)
                    db.commit()

                    _process_single_file(
                        db=db,
                        doc=child_doc,
                        workspace_id=workspace_id,
                        file_name=clean_name,
                        file_bytes=inner_bytes,
                        parent_doc=doc  # 🔥 aggregate into ZIP
                    )

            doc.status = "READY"
            db.commit()
            return

        # ================= NORMAL FILE =================

        doc.status = "PROCESSING"
        doc.total_chunks = 0
        doc.processed_chunks = 0
        db.commit()

        _process_single_file(
            db=db,
            doc=doc,
            workspace_id=workspace_id,
            file_name=file_name,
            file_bytes=file_bytes,
            parent_doc=None
        )

    except Exception as e:
        logger.error(f"Ingestion failed for {file_name}: {str(e)}")
        doc.status = "FAILED"
        db.commit()
        raise e

    finally:
        db.close()

# =====================================================
# SINGLE FILE PROCESSOR
# =====================================================

def _process_single_file(
    db,
    doc,
    workspace_id,
    file_name,
    file_bytes,
    parent_doc=None
):

    from app.ingestion.loader import extract_text_stream
    from app.ingestion.embedder import (
        ensure_collection_exists,
        embeddings,
        client,
        COLLECTION_NAME
    )
    from qdrant_client.models import PointStruct

    try:

        # ----------------- EXTRACT -----------------

        chunks = []

        for item in extract_text_stream(file_name, file_bytes):
            if isinstance(item, dict):
                chunks.append(item)

        if not chunks:
            doc.status = "FAILED"
            db.commit()
            return

        doc.total_chunks = len(chunks)
        doc.processed_chunks = 0

        # 🔥 Aggregate total into ZIP parent
        if parent_doc:
            parent_doc.total_chunks += len(chunks)

        db.commit()

        ensure_collection_exists()

        # ----------------- BATCH EMBEDDING -----------------

        for i in range(0, len(chunks), EMBED_BATCH_SIZE):

            batch_chunks = chunks[i:i + EMBED_BATCH_SIZE]
            batch_texts = [
                chunk.get("text", " ")
                for chunk in batch_chunks
            ]

            vectors = embeddings.embed_documents(batch_texts)

            points = []

            for idx, (vector, chunk) in enumerate(
                zip(vectors, batch_chunks)
            ):

                payload = {
                    "workspace_id": workspace_id,
                    "document_id": str(doc.id),
                    "file_name": file_name,
                    "chunk_index": i + idx,
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "text": chunk.get("text"),
                    "structured_data": chunk.get("structured_data"),
                }

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

            # 🔥 Progressive upsert
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

            # 🔥 Update child progress
            doc.processed_chunks += len(batch_chunks)

            # 🔥 Update ZIP aggregated progress
            if parent_doc:
                parent_doc.processed_chunks += len(batch_chunks)

            db.commit()

            logger.info(
                f"{file_name} progress: "
                f"{doc.processed_chunks}/{doc.total_chunks}"
            )

        doc.status = "READY"
        db.commit()

    except Exception as e:
        logger.error(
            f"Processing error for {file_name}: {str(e)}"
        )
        doc.status = "FAILED"
        db.commit()
        raise e
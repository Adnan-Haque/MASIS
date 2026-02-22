from fastapi import APIRouter, UploadFile, HTTPException
from app.services.document_service import handle_upload
from app.db.database import SessionLocal
from app.db.models import Document
from qdrant_client import QdrantClient

document_router = APIRouter()

@document_router.post("/workspaces/{workspace_id}/upload")
async def upload(workspace_id: str, file: UploadFile):
    return await handle_upload(workspace_id, file)

@document_router.get("/workspaces/{workspace_id}/documents")
def list_documents(workspace_id: str):
    db = SessionLocal()
    docs = db.query(Document).filter(
        Document.workspace_id == workspace_id
    ).all()

    return [
        {
            "id": str(d.id),   # <-- ADD THIS
            "file_name": d.file_name,
            "status": d.status
        }
        for d in docs
    ]

from qdrant_client.models import Filter, FieldCondition, MatchValue

@document_router.delete("/workspaces/{workspace_id}/documents/{document_id}")
def delete_document(workspace_id: str, document_id: str):

    db = SessionLocal()

    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.workspace_id == workspace_id
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    client = QdrantClient(host="qdrant", port=6333)

    client.delete(
        collection_name="masis_documents",
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
    )

    db.delete(doc)
    db.commit()

    return {"status": "deleted"}

@document_router.get("/workspaces/{workspace_id}/documents/{doc_id}/progress")
def get_document_progress(workspace_id: str, doc_id: str):

    db = SessionLocal()

    try:
        doc = db.query(Document).filter(
            Document.workspace_id == workspace_id,
            Document.id == doc_id
        ).first()

        if not doc:
            return {"error": "Document not found"}

        total = doc.total_chunks or 0
        processed = doc.processed_chunks or 0

        percentage = 0
        if total > 0:
            percentage = int((processed / total) * 100)

        return {
            "status": doc.status,
            "total": total,
            "processed": processed,
            "percentage": percentage
        }

    finally:
        db.close()
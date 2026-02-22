from app.db.database import SessionLocal
from app.db.models import Workspace
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from app.db.models import Document
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient

workspace_router = APIRouter()

# List workspaces
@workspace_router.get("/workspaces")
def list_workspaces():
    db = SessionLocal()
    workspaces = db.query(Workspace).all()
    return [w.id for w in workspaces]


@workspace_router.post("/workspaces/{workspace_id}")
def create_workspace(workspace_id: str):
    db = SessionLocal()

    existing = db.query(Workspace).filter(
        Workspace.id == workspace_id
    ).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Workspace already exists"
        )

    workspace = Workspace(id=workspace_id)
    db.add(workspace)
    db.commit()

    return {"status": "created"}

@workspace_router.post("/workspaces/{workspace_id}/cleanup")
def cleanup_stuck_documents(workspace_id: str):
    db = SessionLocal()

    threshold = datetime.utcnow() - timedelta(minutes=10)

    stuck_docs = db.query(Document).filter(
        Document.workspace_id == workspace_id,
        Document.status == "PROCESSING",
        Document.created_at < threshold
    ).all()

    for doc in stuck_docs:
        doc.status = "FAILED"

    db.commit()

    return {"cleaned": len(stuck_docs)}

@workspace_router.delete("/workspaces/{workspace_id}")
def delete_workspace(workspace_id: str):

    db = SessionLocal()

    workspace = db.query(Workspace).filter(
        Workspace.id == workspace_id
    ).first()

    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    client = QdrantClient(host="qdrant", port=6333)

    # Proper filter object
    client.delete(
        collection_name="masis_documents",
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=workspace_id)
                )
            ]
        )
    )

    # Delete document metadata
    db.query(Document).filter(
        Document.workspace_id == workspace_id
    ).delete()

    db.delete(workspace)
    db.commit()

    return {"status": "workspace deleted"}
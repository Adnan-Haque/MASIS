from app.db.database import SessionLocal
from app.db.models import Document
from app.services.dedup_service import compute_hash
from app.workers.ingestion_tasks import ingest_document
from fastapi import HTTPException
import os
import hashlib

UPLOAD_DIR = "/code/uploads"


async def handle_upload(workspace_id: str, file):

    db = SessionLocal()

    try:
        file_bytes = await file.read()

        # 🔥 Create file hash
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # 🔥 Check duplicate by hash
        existing = db.query(Document).filter(
            Document.workspace_id == workspace_id,
            Document.file_hash == file_hash
        ).first()

        if existing:
            raise HTTPException(
                status_code=409,
                detail="Duplicate document"
            )

        # Save file
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(
            UPLOAD_DIR,
            f"{file.filename}"
        )

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Create document row
        doc = Document(
            workspace_id=workspace_id,
            file_name=file.filename,
            file_hash=file_hash,
            status="PROCESSING"
        )

        db.add(doc)
        db.commit()

        # Start async ingestion
        ingest_document.delay(
            workspace_id,
            file.filename,
            file_path
        )

        return {"message": "Upload started"}

    finally:
        db.close()
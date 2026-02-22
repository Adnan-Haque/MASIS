import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .database import Base


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(String, primary_key=True)

    # relationship (optional but good practice)
    documents = relationship(
        "Document",
        back_populates="workspace",
        cascade="all, delete"
    )


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    workspace_id = Column(
        String,
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True
    )

    file_name = Column(String)
    file_hash = Column(String, index=True)

    status = Column(String)  # PROCESSING, READY, FAILED

    created_at = Column(DateTime, default=datetime.utcnow)

    workspace = relationship("Workspace", back_populates="documents")

    total_chunks = Column(Integer, default=0)   
    processed_chunks = Column(Integer, default=0)
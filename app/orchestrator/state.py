from typing import List, TypedDict, Optional
from pydantic import BaseModel
from pydantic import BaseModel


class EvidenceChunk(BaseModel):
    chunk_id: str
    file_name: str
    text: str
    score: float


class MASISState(BaseModel):
    user_query: str
    workspace_id: str
    evidence: List = []
    draft_answer: Optional[str] = None
    final_answer: Optional[str] = None
    confidence: float = 0.0
    retry_count: int = 0
    max_retries: int = 2
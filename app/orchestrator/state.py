from typing import List, Optional, TypedDict
from pydantic import BaseModel


class EvidenceChunk(BaseModel):
    chunk_id: str
    file_name: str
    text: str
    score: float


# ✅ LangGraph state must be a TypedDict (supports dict-style access in nodes)
class MASISState(TypedDict, total=False):
    user_query: str
    workspace_id: str
    evidence: List
    draft_answer: Optional[str]
    final_answer: Optional[str]
    confidence: float
    retry_count: int
    max_retries: int
    critique: Optional[dict]
    requires_human_review: bool
    clarification_question: Optional[str]
    trace: List
    metrics: dict


# ✅ Pydantic model for API input validation (used in routes, not in graph)
class MASISInput(BaseModel):
    user_query: str
    workspace_id: str
    max_retries: int = 2

    def to_state(self) -> MASISState:
        return MASISState(
            user_query=self.user_query,
            workspace_id=self.workspace_id,
            max_retries=self.max_retries,
            evidence=[],
            draft_answer=None,
            final_answer=None,
            confidence=0.0,
            retry_count=0,
            critique=None,
            requires_human_review=False,
            clarification_question=None,
            trace=[],
            metrics={}
        )

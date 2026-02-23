from fastapi import APIRouter
from pydantic import BaseModel
from app.orchestrator.graph import graph

masis_router = APIRouter(prefix="/masis", tags=["MASIS"])


class MASISRequest(BaseModel):
    query: str


@masis_router.post("/workspaces/{workspace_id}")
def masis_query(workspace_id: str, request: MASISRequest):

    initial_state = {
        "user_query": request.query,
        "workspace_id": workspace_id,
        "retry_count": 0,
        "max_retries": 2,

        # 🔥 NEW
        "requires_human_review": False,
        "clarification_question": None,

        "trace": [],
        "metrics": {
            "avg_retrieval_score": 0,
            "retrieval_scores": [],
            "answer_length": 0,
            "citation_count": 0,
            "confidence_history": [],
            "confidence_delta": None,
            "retry_reasons": [],
            "node_latency_ms": {},
            "iterations": [],
            "citation_violations": []
        }
    }

    result = graph.invoke(initial_state)

    return {
        "answer": result.get("final_answer"),
        "confidence": result.get("confidence"),
        "critique": result.get("critique"),
        "evaluation": result.get("metrics", {}).get("evaluation"),
        "requires_human_review": result.get("requires_human_review", False),
        "clarification_question": result.get("clarification_question"),
        "trace": result.get("trace"),
        "metrics": result.get("metrics")
    }
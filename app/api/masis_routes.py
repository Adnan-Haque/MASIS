# import logging
# from fastapi import APIRouter, HTTPException
# from fastapi.concurrency import run_in_threadpool
# from pydantic import BaseModel
# from app.orchestrator.graph import graph
# from app.orchestrator.state import MASISInput

# logger = logging.getLogger(__name__)

# masis_router = APIRouter(prefix="/masis", tags=["MASIS"])


# class MASISRequest(BaseModel):
#     query: str
#     max_retries: int = 2  # ✅ Exposed so callers can tune retry depth per request


# @masis_router.post("/workspaces/{workspace_id}")
# async def masis_query(workspace_id: str, request: MASISRequest):
#     """
#     Run the MASIS multi-agent pipeline for a given workspace and query.

#     - Researcher fetches evidence from Qdrant (scoped to workspace_id)
#     - Synthesizer generates a cited draft answer
#     - Critic audits for hallucinations and citation validity
#     - Evaluator scores quality across 4 dimensions
#     - Supervisor decides to retry, escalate to HITL, or finalize

#     Returns the final answer, confidence score, full audit trace, and evaluation metrics.
#     If the system cannot reach acceptable quality, returns requires_human_review=True
#     with a clarification_question explaining what the user should do.
#     """

#     # ✅ Build initial state via MASISInput — single source of truth for state shape
#     initial_state = MASISInput(
#         user_query=request.query,
#         workspace_id=workspace_id,
#         max_retries=request.max_retries,
#     ).to_state()

#     # ✅ Run graph in a thread pool — graph.invoke() is sync and blocks on LLM calls.
#     #    Without run_in_threadpool, this would block FastAPI's async event loop,
#     #    preventing other requests from being handled during LLM wait time.
#     try:
#         result = await run_in_threadpool(graph.invoke, initial_state)
#     except Exception as e:
#         logger.exception("[masis_query] Graph execution failed for workspace=%s", workspace_id)
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "error": "Pipeline execution failed.",
#                 "reason": str(e),
#                 "workspace_id": workspace_id,
#             }
#         )

#     # ✅ HITL response — system couldn't reach quality threshold, human input needed
#     #    Still return final_answer (the best draft we managed) so the frontend
#     #    can show it alongside the warning — better than a blank screen.
#     if result.get("requires_human_review"):
#         return {
#             "status": "needs_clarification",
#             "answer": result.get("final_answer"),   # best draft, may be low quality
#             "confidence": result.get("confidence"),
#             "requires_human_review": True,
#             "clarification_question": result.get("clarification_question"),
#             "trace": result.get("trace"),
#             "metrics": result.get("metrics"),
#         }

#     # ✅ Success response
#     return {
#         "status": "success",
#         "answer": result.get("final_answer"),
#         "confidence": result.get("confidence"),
#         "requires_human_review": False,
#         "clarification_question": None,
#         "critique": result.get("critique"),
#         "evaluation": result.get("metrics", {}).get("evaluation"),
#         "trace": result.get("trace"),
#         "metrics": result.get("metrics"),
#     }


import logging
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from app.orchestrator.graph import graph
from app.orchestrator.state import MASISInput

logger = logging.getLogger(__name__)

masis_router = APIRouter(prefix="/masis", tags=["MASIS"])


class MASISRequest(BaseModel):
    query: str
    max_retries: int = 2  # ✅ Exposed so callers can tune retry depth per request


@masis_router.post("/workspaces/{workspace_id}")
async def masis_query(workspace_id: str, request: MASISRequest):
    """
    Run the MASIS multi-agent pipeline for a given workspace and query.

    - Researcher fetches evidence from Qdrant (scoped to workspace_id)
    - Synthesizer generates a cited draft answer
    - Critic audits for hallucinations and citation validity
    - Evaluator scores quality across 4 dimensions
    - Supervisor decides to retry, escalate to HITL, or finalize

    Returns the final answer, confidence score, full audit trace, and evaluation metrics.
    If the system cannot reach acceptable quality, returns requires_human_review=True
    with a clarification_question explaining what the user should do.
    """

    # ✅ Build initial state via MASISInput — single source of truth for state shape
    initial_state = MASISInput(
        user_query=request.query,
        workspace_id=workspace_id,
        max_retries=request.max_retries,
    ).to_state()

    # ✅ Run graph in a thread pool — graph.invoke() is sync and blocks on LLM calls.
    #    Without run_in_threadpool, this would block FastAPI's async event loop,
    #    preventing other requests from being handled during LLM wait time.
    try:
        result = await run_in_threadpool(graph.invoke, initial_state)
    except Exception as e:
        logger.exception("[masis_query] Graph execution failed for workspace=%s", workspace_id)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Pipeline execution failed.",
                "reason": str(e),
                "workspace_id": workspace_id,
            }
        )

    # ✅ HITL response — system couldn't reach quality threshold, human input needed
    #    Still return final_answer (the best draft we managed) so the frontend
    #    can show it alongside the warning — better than a blank screen.
    if result.get("requires_human_review"):
        return {
            "status": "needs_clarification",
            "answer": result.get("final_answer"),   # best draft, may be low quality
            "confidence": result.get("confidence"),
            "requires_human_review": True,
            "clarification_question": result.get("clarification_question"),
            "critique": result.get("critique"),
            "evaluation": result.get("metrics", {}).get("evaluation"),  # ✅ always include — especially useful when confidence is low
            "trace": result.get("trace"),
            "metrics": result.get("metrics"),
        }

    # ✅ Success response
    return {
        "status": "success",
        "answer": result.get("final_answer"),
        "confidence": result.get("confidence"),
        "requires_human_review": False,
        "clarification_question": None,
        "critique": result.get("critique"),
        "evaluation": result.get("metrics", {}).get("evaluation"),
        "trace": result.get("trace"),
        "metrics": result.get("metrics"),
    }
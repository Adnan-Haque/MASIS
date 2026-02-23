from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from pydantic import BaseModel
from .state import EvidenceChunk, MASISState
import time
import re
import threading

# 🔹 IMPORTANT: Use Docker service name
qdrant_client = QdrantClient(host="qdrant", port=6333)

embeddings = OpenAIEmbeddings()

# ✅ Model selection by task complexity:
#    - gpt-4o-mini: generation (synthesizer, compression) — cheap, fast
#    - gpt-4o: auditing & scoring (critic, evaluator) — stronger reasoning
llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o")
evaluator_llm = ChatOpenAI(model="gpt-4o")

# =========================================
# Rate Limiting — token bucket (10 calls/min)
# =========================================
_rate_lock = threading.Lock()
_call_timestamps = []
MAX_CALLS_PER_MINUTE = 10

def _rate_limit():
    """Block if we've exceeded MAX_CALLS_PER_MINUTE in the last 60 seconds."""
    with _rate_lock:
        now = time.time()
        global _call_timestamps
        _call_timestamps = [t for t in _call_timestamps if now - t < 60]
        if len(_call_timestamps) >= MAX_CALLS_PER_MINUTE:
            sleep_for = 60 - (now - _call_timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        _call_timestamps.append(time.time())


def _init_metrics(state: MASISState):
    """Ensure metrics dict and all sub-keys exist."""
    if "metrics" not in state or state["metrics"] is None:
        state["metrics"] = {}
    m = state["metrics"]
    m.setdefault("node_latency_ms", {})
    m.setdefault("confidence_history", [])
    m.setdefault("retry_reasons", [])
    m.setdefault("iterations", [])
    m.setdefault("citation_violations", [])
    m.setdefault("evaluation", {})

    if "trace" not in state or state["trace"] is None:
        state["trace"] = []


# =========================================
# 1️⃣ RESEARCHER NODE
# =========================================
def researcher_node(state: MASISState) -> MASISState:
    start = time.time()
    _init_metrics(state)

    query = state["user_query"]
    workspace_id = state["workspace_id"]
    retry_count = state.get("retry_count", 0)
    critique = state.get("critique") or {}

    # 🔥 Critique-aware query augmentation on retry
    augmented_query = query
    if retry_count > 0 and critique:
        focus_terms = (
            critique.get("unsupported_claims", []) +
            critique.get("logical_gaps", [])
        )
        if focus_terms:
            augmented_query += " " + " ".join(str(t) for t in focus_terms)

    # 🔥 Dynamic retrieval expansion
    limit = 5 if retry_count == 0 else 10
    query_vector = embeddings.embed_query(augmented_query)

    results = qdrant_client.search(
        collection_name="masis_documents",
        query_vector=query_vector,
        limit=limit,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=workspace_id)
                )
            ]
        )
    )

    seen_ids = set()
    evidence = []
    scores = []

    for r in results:
        if str(r.id) not in seen_ids:
            seen_ids.add(str(r.id))
            scores.append(r.score)
            evidence.append(
                EvidenceChunk(
                    chunk_id=str(r.id),
                    file_name=r.payload.get("file_name"),
                    text=r.payload.get("text"),
                    score=r.score
                )
            )

    duration = int((time.time() - start) * 1000)

    # ✅ Handle empty retrieval gracefully — escalate to HITL immediately
    if not evidence:
        state["requires_human_review"] = True
        state["clarification_question"] = (
            "No relevant documents were found for your query in this workspace. "
            "Please upload relevant documents or refine your question."
        )
        state["trace"].append({
            "node": "researcher",
            "retry_count": retry_count,
            "warning": "zero_results_retrieved",
            "duration_ms": duration
        })
        state["evidence"] = []
        return state

    avg_score = sum(scores) / len(scores)

    state["metrics"]["retrieval_scores"] = scores
    state["metrics"]["avg_retrieval_score"] = avg_score
    state["metrics"]["node_latency_ms"]["researcher"] = duration

    state["trace"].append({
        "node": "researcher",
        "retry_count": retry_count,
        "chunks": len(scores),
        "avg_score": round(avg_score, 3),
        "augmented_query_used": retry_count > 0,
        "duration_ms": duration
    })

    state["evidence"] = evidence
    return state


# =========================================
# 2️⃣ SYNTHESIZER NODE (WITH CONTEXT CONTROL)
# =========================================
def synthesizer_node(state: MASISState) -> MASISState:
    start = time.time()
    _init_metrics(state)

    query = state["user_query"]
    evidence = state.get("evidence", [])
    retry_count = state.get("retry_count", 0)
    critique = state.get("critique")

    compression_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    original_context_chars = sum(len(e.text) for e in evidence)
    state["metrics"]["original_context_chars"] = original_context_chars

    MAX_CONTEXT_CHARS = 6000
    context_compressed = False
    compression_duration = 0

    if original_context_chars > MAX_CONTEXT_CHARS:
        compression_start = time.time()

        sorted_evidence = sorted(evidence, key=lambda x: x.score, reverse=True)
        top_chunks = sorted_evidence[:3]
        low_chunks = sorted_evidence[3:]

        formatted_chunks = "\n\n".join(
            [f"[{e.chunk_id}] {e.text}" for e in low_chunks]
        )

        compress_prompt = f"""
Summarize each chunk in under 200 characters.
Preserve numbers and metrics.

Format:
[chunk_id]: summary

Chunks:
{formatted_chunks}
"""
        _rate_limit()
        compressed_output = compression_llm.invoke(compress_prompt).content

        compressed_map = {}
        for line in compressed_output.split("\n"):
            if ":" in line:
                cid, summary = line.split(":", 1)
                compressed_map[cid.strip().replace("[","").replace("]","")] = summary.strip()

        compressed_evidence = list(top_chunks)
        for e in low_chunks:
            summary = compressed_map.get(e.chunk_id, e.text[:200])
            compressed_evidence.append(
                EvidenceChunk(
                    chunk_id=e.chunk_id,
                    file_name=e.file_name,
                    text=summary,
                    score=e.score
                )
            )

        evidence = compressed_evidence
        context_compressed = True
        compression_duration = int((time.time() - compression_start) * 1000)

        compressed_chars = sum(len(e.text) for e in evidence)
        state["metrics"]["compressed_context_chars"] = compressed_chars
        state["metrics"]["compression_ratio"] = round(
            compressed_chars / original_context_chars, 3
        )

        # ✅ Over-compression: signal supervisor to retry with broader retrieval
        if compressed_chars / original_context_chars < 0.35:
            state["metrics"]["over_compression_flag"] = True
            current_critique = state.get("critique") or {}
            current_critique["needs_retry"] = True
            current_critique["logical_gaps"] = current_critique.get("logical_gaps", []) + [
                "Evidence was over-compressed; critical context may have been lost."
            ]
            state["critique"] = current_critique

    context = "\n\n".join(
        [f"[{e.chunk_id}] {e.text}" for e in evidence]
    )

    critique_feedback = ""
    if retry_count > 0 and critique:
        critique_feedback = f"""
Previous critique:
Hallucination: {critique.get("hallucination_detected")}
Unsupported claims: {critique.get("unsupported_claims")}
Logical gaps: {critique.get("logical_gaps")}
Conflicts: {critique.get("conflicting_evidence")}
Correct these issues.
"""

    prompt = f"""
Use ONLY the evidence below.
Every claim must cite [chunk_id].
If there is insufficient evidence, explicitly state so.

{critique_feedback}

Question:
{query}

Evidence:
{context}
"""

    tagged_llm = llm.with_config(
        tags=["synthesizer"],
        metadata={
            "retry_count": retry_count,
            "workspace_id": state["workspace_id"]
        }
    )

    _rate_limit()
    response = tagged_llm.invoke(prompt)
    answer = response.content

    citation_count = answer.count("[")
    answer_length = len(answer)
    duration = int((time.time() - start) * 1000)

    state["metrics"]["citation_count"] = citation_count
    state["metrics"]["answer_length"] = answer_length
    state["metrics"]["compression_latency_ms"] = compression_duration
    state["metrics"]["node_latency_ms"]["synthesizer"] = duration

    state["trace"].append({
        "node": "synthesizer",
        "retry_count": retry_count,
        "context_chars": original_context_chars,
        "context_compressed": context_compressed,
        "compression_latency_ms": compression_duration,
        "answer_length": answer_length,
        "citations": citation_count,
        "duration_ms": duration
    })

    state["draft_answer"] = answer
    return state


# =========================================
# 3️⃣ CRITIC NODE + HARD CITATION ENGINE
# =========================================
class Critique(BaseModel):
    confidence: float
    hallucination_detected: bool
    unsupported_claims: list[str]
    logical_gaps: list[str]
    conflicting_evidence: list[str]   # ✅ list[str] — matches prompt instruction
    needs_retry: bool


def critic_node(state: MASISState) -> MASISState:
    start = time.time()
    _init_metrics(state)

    answer = state.get("draft_answer", "")
    evidence = state.get("evidence", [])

    context = "\n\n".join(
        [f"[{e.chunk_id}] {e.text}" for e in evidence]
    )

    prompt = f"""
You are an AI Auditor.

Evaluate the answer strictly using the provided evidence.

Return structured fields:
- confidence (float between 0 and 1)
- hallucination_detected (bool)
- unsupported_claims (list[str])
- logical_gaps (list[str])
- conflicting_evidence (list[str])
- needs_retry (bool)

Answer:
{answer}

Evidence:
{context}
"""

    tagged_llm = (
        critic_llm
        .with_structured_output(Critique)
        .with_config(tags=["critic"])
    )

    _rate_limit()
    critique = tagged_llm.invoke(prompt)

    if hasattr(critique, "model_dump"):
        critique = critique.model_dump()

    # Normalize confidence to 0–1
    confidence = critique.get("confidence", 0.0)
    if confidence > 1:
        confidence = confidence / 100.0
    confidence = max(0.0, min(confidence, 1.0))
    critique["confidence"] = confidence

    # ===============================
    # 🔥 HARD CITATION ENGINE
    # ===============================
    citations = re.findall(r"\[(.*?)\]", answer)
    valid_ids = {e.chunk_id for e in evidence}

    invalid_citations = [c for c in citations if c not in valid_ids]

    sentences = re.split(r"[.!?]", answer)
    uncited_claims = [
        s.strip()
        for s in sentences
        if s.strip()
        and "[" not in s
        and "insufficient evidence" not in s.lower()
        and "not provided" not in s.lower()
        and "cannot provide" not in s.lower()
    ]

    # Penalty logic
    penalty_factor = 1.0
    if invalid_citations:
        critique["hallucination_detected"] = True
        critique["needs_retry"] = True
        penalty_factor *= 0.5
    if uncited_claims and citations:
        penalty_factor *= 0.9

    confidence = max(0.0, min(confidence * penalty_factor, 1.0))
    critique["confidence"] = confidence

    # Telemetry
    state["metrics"]["citation_violations"].append({
        "invalid_ids": invalid_citations,
        "uncited_claims": len(uncited_claims),
        "iteration": state.get("retry_count", 0)
    })
    state["metrics"]["confidence_history"].append(confidence)

    state["critique"] = critique
    state["confidence"] = confidence
    state["final_answer"] = answer

    duration = int((time.time() - start) * 1000)
    state["metrics"]["node_latency_ms"]["critic"] = duration

    state["trace"].append({
        "node": "critic",
        "confidence": confidence,
        "hallucination": critique.get("hallucination_detected", False),
        "needs_retry": critique.get("needs_retry", False),
        "conflicts": len(critique.get("conflicting_evidence", [])),
        "invalid_citations": len(invalid_citations),
        "uncited_claims": len(uncited_claims),
        "duration_ms": duration
    })

    return state


# =========================================
# 4️⃣ SUPERVISOR NODE
# =========================================
def supervisor_node(state: MASISState) -> MASISState:
    _init_metrics(state)

    # First call — no answer yet, pass through to start the pipeline
    if state.get("draft_answer") is None:
        return state

    critique = state.get("critique") or {}
    confidence = critique.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries") or 2

    citation_violations = state["metrics"].get("citation_violations", [])

    citation_issue = False
    if citation_violations:
        latest = citation_violations[-1]
        if latest.get("invalid_ids"):
            citation_issue = True

    hallucination_flag = critique.get("hallucination_detected", False)
    critic_retry_flag = critique.get("needs_retry", False)
    has_conflicts = bool(critique.get("conflicting_evidence"))

    LOW_CONF_THRESHOLD = 0.75

    quality_issue = (
        confidence < LOW_CONF_THRESHOLD
        or citation_issue
        or hallucination_flag
        or critic_retry_flag
    )

    # =====================================================
    # 1️⃣ Retry — attempt resolution before escalating
    # =====================================================
    if (quality_issue or has_conflicts) and retry_count < max_retries:
        state["retry_count"] = retry_count + 1

        reason = "quality_issue_detected"
        if has_conflicts and not quality_issue:
            reason = "conflicting_evidence_attempting_resolution"

        state["trace"].append({
            "node": "supervisor",
            "decision": "retry",
            "confidence": confidence,
            "retry_count": state["retry_count"],
            "reason": reason
        })

        return state

    # =====================================================
    # 2️⃣ HITL — conflict could not be resolved after retries
    # =====================================================
    if has_conflicts and retry_count >= max_retries:
        state["requires_human_review"] = True
        state["clarification_question"] = (
            "Conflicting information was detected across documents and could not be "
            "automatically resolved after multiple attempts. "
            "Please review the competing claims and select a preferred source."
        )

        state["trace"].append({
            "node": "supervisor",
            "decision": "HITL_conflict",
            "confidence": confidence,
            "retry_count": retry_count
        })

        return state

    # =====================================================
    # 3️⃣ HITL — quality issue, retries exhausted
    # =====================================================
    if quality_issue and retry_count >= max_retries:
        state["requires_human_review"] = True
        state["clarification_question"] = (
            f"After {max_retries} refinement attempts, confidence remains "
            f"{round(confidence * 100, 1)}%. "
            "You may refine your query or upload additional evidence."
        )

        state["trace"].append({
            "node": "supervisor",
            "decision": "HITL_triggered",
            "confidence": confidence,
            "retry_count": retry_count
        })

        return state

    # =====================================================
    # 4️⃣ Finalize — healthy state
    # =====================================================
    state["trace"].append({
        "node": "supervisor",
        "decision": "finalize",
        "confidence": confidence,
        "retry_count": retry_count
    })

    return state


# =========================================
# 5️⃣ EVALUATOR NODE (LLM-as-Judge)
# =========================================
class Evaluation(BaseModel):
    faithfulness: float
    relevance: float
    completeness: float
    reasoning_quality: float
    overall_score: float
    improvement_suggestions: list[str]


def evaluator_node(state: MASISState) -> MASISState:
    start = time.time()
    _init_metrics(state)

    query = state["user_query"]
    answer = state.get("final_answer", "")
    evidence = state.get("evidence", [])

    context = "\n\n".join(
        [f"[{e.chunk_id}] {e.text}" for e in evidence]
    )

    # ✅ user_query included so Relevance and Completeness can be scored meaningfully
    prompt = f"""
You are an evaluation agent.

Score strictly from 0 to 1:

Faithfulness:
1 = every claim directly supported by cited evidence
0.5 = partially supported
0 = unsupported

Relevance:
1 = fully answers user question
0.5 = partially relevant
0 = irrelevant

Completeness:
1 = covers all aspects of the question
0.5 = partially complete
0 = incomplete

Reasoning Quality:
1 = strong structured reasoning
0.5 = shallow reasoning
0 = no reasoning

Be strict. Do NOT default to 1.

Question:
{query}

Answer:
{answer}

Evidence:
{context}
"""

    tagged_llm = (
        evaluator_llm
        .with_structured_output(Evaluation)
        .with_config(tags=["evaluation"])
    )

    _rate_limit()
    evaluation = tagged_llm.invoke(prompt)

    if hasattr(evaluation, "model_dump"):
        evaluation = evaluation.model_dump()

    # Normalize scores
    for k in ["faithfulness", "relevance", "completeness", "reasoning_quality", "overall_score"]:
        if evaluation.get(k, 0) > 1:
            evaluation[k] = evaluation[k] / 100.0

    state["metrics"]["evaluation"] = evaluation

    duration = int((time.time() - start) * 1000)
    state["metrics"]["node_latency_ms"]["evaluator"] = duration

    state["trace"].append({
        "node": "evaluator",
        "overall_score": evaluation.get("overall_score"),
        "faithfulness": evaluation.get("faithfulness"),
        "relevance": evaluation.get("relevance"),
        "completeness": evaluation.get("completeness"),
        "duration_ms": duration
    })

    return state

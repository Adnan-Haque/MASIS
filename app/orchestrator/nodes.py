from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from pydantic import BaseModel
from .state import EvidenceChunk
import time
import re

# 🔹 IMPORTANT: Use Docker service name
qdrant_client = QdrantClient(host="qdrant", port=6333)

embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o-mini")

# =========================================
# 1️⃣ RESEARCHER NODE
# =========================================
def researcher_node(state):
    start = time.time()

    state.setdefault("metrics", {})
    state["metrics"].setdefault("node_latency_ms", {})
    state["metrics"].setdefault("confidence_history", [])
    state["metrics"].setdefault("retry_reasons", [])
    state["metrics"].setdefault("iterations", [])
    state["metrics"].setdefault("citation_violations", [])
    state.setdefault("trace", [])

    query = state["user_query"]
    workspace_id = state["workspace_id"]
    retry_count = state.get("retry_count", 0)
    critique = state.get("critique", {})

    # 🔥 Critique-aware augmentation
    augmented_query = query
    if retry_count > 0 and critique:
        focus_terms = (
            critique.get("unsupported_claims", []) +
            critique.get("logical_gaps", [])
        )
        if focus_terms:
            augmented_query += " " + " ".join(focus_terms)

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

    avg_score = sum(scores) / len(scores) if scores else 0
    duration = int((time.time() - start) * 1000)

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
def synthesizer_node(state):
    start = time.time()

    query = state["user_query"]
    evidence = state["evidence"]
    retry_count = state.get("retry_count", 0)
    critique = state.get("critique")

    compression_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 🔥 Context size calculation
    original_context_chars = sum(len(e.text) for e in evidence)
    state["metrics"]["original_context_chars"] = original_context_chars

    MAX_CONTEXT_CHARS = 6000
    context_compressed = False
    compression_duration = 0

    if original_context_chars > MAX_CONTEXT_CHARS:

        compression_start = time.time()

        # 🔥 Sort by relevance
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

        compressed_output = compression_llm.invoke(compress_prompt).content

        # 🔥 Parse summaries
        compressed_map = {}
        for line in compressed_output.split("\n"):
            if ":" in line:
                cid, summary = line.split(":", 1)
                compressed_map[cid.strip().replace("[","").replace("]","")] = summary.strip()

        compressed_evidence = []

        for e in top_chunks:
            compressed_evidence.append(e)

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

        if compressed_chars / original_context_chars < 0.35:
            state["metrics"]["over_compression_flag"] = True

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
Use ONLY the evidence.
Every claim must cite [chunk_id].
If insufficient evidence, say so.

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
    conflicting_evidence: list[dict]
    needs_retry: bool


def critic_node(state):
    start = time.time()

    state.setdefault("metrics", {})
    state["metrics"].setdefault("node_latency_ms", {})
    state["metrics"].setdefault("citation_violations", [])
    state["metrics"].setdefault("confidence_history", [])

    answer = state["draft_answer"]
    evidence = state["evidence"]

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

    critique = tagged_llm.invoke(prompt)

    if hasattr(critique, "model_dump"):
        critique = critique.model_dump()

    # ===============================
    # 🔥 Normalize Confidence
    # ===============================
    confidence = critique.get("confidence", 0.0)

    # Handle 0–100 scale
    if confidence > 1:
        confidence = confidence / 100.0

    confidence = max(0.0, min(confidence, 1.0))
    critique["confidence"] = confidence

    # ===============================
    # 🔥 HARD CITATION ENGINE
    # ===============================
    import re

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

    # ===============================
    # 🔥 Penalty Logic (Stable)
    # ===============================
    penalty_factor = 1.0

    if invalid_citations:
        critique["hallucination_detected"] = True
        critique["needs_retry"] = True
        penalty_factor *= 0.5

    if uncited_claims and citations:
        penalty_factor *= 0.9

    confidence = confidence * penalty_factor
    confidence = max(0.0, min(confidence, 1.0))

    critique["confidence"] = confidence

    # ===============================
    # Save Telemetry
    # ===============================
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
def supervisor_node(state):

    if state.get("draft_answer") is None:
        return state

    critique = state.get("critique", {})
    confidence = critique.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    metrics = state.get("metrics", {})
    citation_violations = metrics.get("citation_violations", [])
    confidence_history = metrics.get("confidence_history", [])

    state.setdefault("requires_human_review", False)
    state.setdefault("clarification_question", None)
    state.setdefault("trace", [])

    LOW_CONF_THRESHOLD = 0.75

    # =====================================================
    # 1️⃣ Hard Conflict Escalation
    # =====================================================
    if critique.get("conflicting_evidence"):
        state["requires_human_review"] = True
        state["clarification_question"] = (
            "Conflicting information detected across documents. "
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
    # 2️⃣ Quality Signals
    # =====================================================
    citation_issue = False
    uncited_claims = False

    if citation_violations:
        latest = citation_violations[-1]
        if latest.get("invalid_ids"):
            citation_issue = True
        if latest.get("uncited_claims", 0) > 0:
            uncited_claims = True

    hallucination_flag = critique.get("hallucination_detected", False)
    critic_retry_flag = critique.get("needs_retry", False)

    quality_issue = (
        confidence < LOW_CONF_THRESHOLD
        or citation_issue
        or hallucination_flag
        or critic_retry_flag
    )

    # =====================================================
    # 3️⃣ Retry Logic
    # =====================================================
    if quality_issue and retry_count < max_retries:
        state["retry_count"] += 1

        state["trace"].append({
            "node": "supervisor",
            "decision": "retry",
            "confidence": confidence,
            "retry_count": state["retry_count"],
            "reason": "quality_issue_detected"
        })

        return state

    # =====================================================
    # 4️⃣ HITL Trigger (Only If Retries Exhausted)
    # =====================================================
    if quality_issue and retry_count >= max_retries:
        state["requires_human_review"] = True

        state["clarification_question"] = (
            f"After {max_retries} refinement attempts, confidence remains "
            f"{round(confidence*100,1)}%. "
            "You may refine the query or upload additional evidence."
        )

        state["trace"].append({
            "node": "supervisor",
            "decision": "HITL_triggered",
            "confidence": confidence,
            "retry_count": retry_count
        })

        return state

    # =====================================================
    # 5️⃣ Finalize (Healthy State)
    # =====================================================
    state["trace"].append({
        "node": "supervisor",
        "decision": "finalize",
        "confidence": confidence,
        "retry_count": retry_count
    })

    return state
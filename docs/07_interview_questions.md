# 🎯 Interview Questions — MASIS Deep Dive Preparation

Questions drawn from the case study's curveball and deep-dive sections, updated to reflect the current implementation including the score threshold filter, adaptive retrieval, proportional penalties, citation audit injection, and the lowered confidence threshold.

---

## Section 1: Architecture & Design

---

**Q1. Walk me through your orchestration strategy. Why a Supervisor-controlled DAG over a sequential chain or fully autonomous loop?**

A sequential chain (A→B→C→END) cannot express feedback loops. A fully autonomous loop (agents decide their own next steps) is unpredictable and can't guarantee termination.

MASIS uses a **Supervisor-controlled DAG**: the inner pipeline (Researcher→Synthesizer→Critic→Evaluator) is a fixed linear chain. All conditional logic lives in the Supervisor. This means:
- The inner pipeline is always predictable and testable in isolation.
- All routing intelligence is in one place — easy to audit, modify, explain.
- Termination is guaranteed by `max_retries` — impossible to enforce in a fully autonomous loop.

---

**Q2. What happens if the Researcher enters an infinite search loop?**

It cannot, by design. The Researcher is a node that executes once per cycle and returns. The cycle is bounded by `max_retries` in the Supervisor. After 2 retries the Supervisor always escalates to HITL → END.

Additionally, if Qdrant returns candidates that all fail the score threshold filter (`MIN_SCORE_THRESHOLD = 0.60`), the Researcher sets `requires_human_review = True` immediately — exiting to END without any retry cycle starting. The graph cannot loop on a workspace with no qualifying evidence.

---

**Q3. How do you prevent agentic drift — sub-tasks diverging from original user intent?**

1. **`user_query` is immutable** — set at init, never overwritten. Every node receives the original query.
2. **Augmentation is additive** — on retry, focus terms are appended to the original: `augmented_query = query + " " + focus_terms`. Intent is preserved; only search scope is widened.
3. **Synthesizer always receives the original question** — even on retry 2.
4. **Evaluator scores Relevance** — explicitly checks whether the final answer addresses the original question. Low scores surface in monitoring.
5. **Supervisor routing is query-agnostic** — never reformulates the goal.

---

**Q4. With 10,000 documents, how do you avoid losing critical context?**

Three layers:

**Layer 1 — Score-filtered retrieval.** Qdrant returns up to 20 candidates (on retry), but only those scoring ≥ 0.60 cosine similarity pass to the Synthesizer. This narrows the haystack to genuinely relevant chunks, not just the "closest" ones.

**Layer 2 — Relevance-ranked compression.** If surviving chunks exceed 6,000 characters, the Synthesizer sorts by score, keeps the top 3 full, and compresses the rest to 200 characters each. Highest-value evidence is never degraded.

**Layer 3 — Position-aware assembly.** After compression, highest-scoring chunks appear first. LLMs perform best on content at the beginning of prompts.

Production improvement: add MMR (Maximal Marginal Relevance) to diversify retrieved chunks — rather than 5 near-identical chunks about the same paragraph, get 5 covering different aspects.

---

**Q5. How does the system handle conflicting evidence between documents?**

1. **First** the Supervisor triggers a retry with augmented query and expanded retrieval (20 candidates vs. 10). The Researcher may surface chunks that clarify which source is authoritative.

2. **If retries are exhausted and conflict persists**, the Supervisor sets `requires_human_review = True`:
   *"Conflicting information was detected across documents and could not be automatically resolved. Please review the competing claims and select a preferred source."*

3. The frontend shows the best draft alongside the conflict warning and the full quality assessment panel — including the Evaluator's faithfulness score, which will be low due to citation audit findings.

We deliberately chose not to auto-resolve conflicts by picking the higher-confidence source — in a strategic intelligence system, silently choosing one source over another without human knowledge is more dangerous than surfacing the conflict.

---

**Q6. Where would you swap GPT-4-class models for smaller SLMs?**

Current allocation:
- `gpt-4o-mini` — Synthesizer (generation), compression (summarization)
- `gpt-4o` — Critic (hallucination detection), Evaluator (quality scoring)

**Keep large models for:** Critic must detect subtle semantic hallucinations and calibrate structured confidence scores. Evaluator must apply nuanced scoring rubrics and honour hard constraints on faithfulness. Both require deep reasoning.

**Use small model for:** Synthesizer follows explicit instructions ("cite every claim, hedge when evidence is partial") on well-scoped content — a pattern-following task, not deep reasoning. `gpt-4o-mini` costs ~15-20x less and is fully capable. Compression (summarize to 200 chars, preserve numbers) is even simpler.

**Future swap:** Test a fine-tuned `Mistral-7B` for the Synthesizer at scale. Keep `gpt-4o` for Critic and Evaluator.

---

## Section 2: RAG & Retrieval

---

**Q7. You added a score threshold filter. How did you calibrate 0.60 and what are the tradeoffs?**

0.60 cosine similarity with OpenAI's `text-embedding-ada-002` is a pragmatic baseline — in practice, chunks below this threshold are semantically too distant from the query to produce useful citations. Above it, chunks are reliably topically related.

Tradeoffs:
- **Too high (e.g., 0.75):** Even well-phrased queries may return no qualifying evidence. Frequent HITL escalations for answerable questions.
- **Too low (e.g., 0.45):** Weak evidence passes through, vague answers, low confidence — the original problem.

The threshold is adaptive: drops to 0.55 on retry because the augmented query is more specific. What scored 0.57 on the original query may be highly relevant to the augmented terms.

In production I'd tune this per-workspace or per-document-type using offline evaluation: run a set of known-answerable queries and find the threshold where recall stays high while precision doesn't collapse.

---

**Q8. Your Researcher does semantic search only. How would you extend to hybrid retrieval?**

```python
# Semantic leg (existing)
dense_vector = embeddings.embed_query(augmented_query)
semantic_results = qdrant_client.search(
    query_vector=("dense", dense_vector), limit=limit, ...
)

# Keyword leg (new — BM25 sparse vectors)
sparse_results = qdrant_client.search(
    query_vector=("sparse", sparse_vector),  # SPLADE or BM25 encoder
    limit=limit, ...
)

# Fusion
final_results = reciprocal_rank_fusion(semantic_results, sparse_results)
```

Keyword search catches exact-match terms (product names, codes, dates) that semantic search misses. Semantic search catches conceptually related content where exact words differ. Hybrid is almost always better than either alone. Qdrant natively supports both dense and sparse vectors in a single collection with RRF fusion.

---

**Q9. How do you ensure repeated queries produce stable, consistent conclusions?**

1. **Score threshold filter** — same query always retrieves the same qualifying chunks (Qdrant's ANN is deterministic for the same index state).
2. **Temperature = 0 on compression LLM** — compression is deterministic.
3. **Structured output via function calling** — Critic and Evaluator use `.with_structured_output()`, significantly reducing variance vs. free-form generation.
4. **Deterministic routing** — Supervisor uses pure Python. Same state → same decision.
5. **Proportional penalty** — Critic's confidence penalty is deterministic code, not LLM output.

Main variance source: Synthesizer and Critic LLM calls at non-zero temperature. For production consistency I'd set `temperature=0` on both and evaluate quality impact.

---

**Q10. How does the "lost-in-the-middle" problem apply here?**

With the score filter retaining only high-quality chunks, and the compression step keeping top 3 full + compressing the rest, the highest-relevance content is always at the front of the context — where LLM attention is strongest. The lost-in-the-middle problem primarily affects systems that dump 50+ chunks into a prompt indiscriminately. MASIS's two-stage filtering (score threshold + compression ranking) prevents that.

For very large document corpora I'd add:
1. **Re-ranking** — cross-encoder (e.g., `ms-marco-MiniLM-L-6-v2`) after initial retrieval for more accurate relevance scores.
2. **Recursive summarisation** — for very long documents, chunk → summarise → embed summaries.
3. **Map-reduce synthesis** — split chunks into groups, synthesise each, then synthesise the partial answers.

---

## Section 3: State Management & Engineering

---

**Q11. Why is graph state a TypedDict and not a Pydantic BaseModel?**

Real bug encountered. LangGraph's `StateGraph` requires dict-style access internally — nodes must use `state["key"]` and `state.get("key")`. Pydantic `BaseModel` uses attribute access (`state.field`) and raises `AttributeError` when you try `state.get()`.

`TypedDict` gives:
- Dict-style access (compatible with LangGraph)
- Type hints (IDE support)
- `total=False` (all fields optional — correct for a state object that evolves through the pipeline)

API boundary uses `MASISInput` (Pydantic `BaseModel`) for validation, converted to `MASISState` via `.to_state()`. Validation at the edge, TypedDict inside the graph.

---

**Q12. How is state shared between agents? How do you prevent uncontrolled context growth?**

State is a single shared `MASISState` dict. Each node mutates and returns it — LangGraph merges the result back into the running state.

Context growth is prevented at the **prompt level**, not the state level:
- State stores full `EvidenceChunk` objects (necessary for the citation engine to validate IDs).
- Prompts use compressed text when chunks exceed 6,000 characters.
- The Synthesizer decides what enters the LLM's context window; the state stores raw evidence.

New in current implementation: `last_citation_audit` in `metrics` — a compact structured dict (not the full evidence) passed from Critic to Evaluator. This keeps inter-node communication lightweight.

If `trace` growth became a concern over many retries, I'd add a trimming step in the Supervisor keeping only the last N entries.

---

**Q13. What's your rate limiting strategy, and why does it matter in multi-agent systems?**

```python
MAX_CALLS_PER_MINUTE = 10

def _rate_limit():
    with _rate_lock:
        now = time.time()
        _call_timestamps = [t for t in _call_timestamps if now - t < 60]
        if len(_call_timestamps) >= MAX_CALLS_PER_MINUTE:
            sleep_for = 60 - (now - _call_timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        _call_timestamps.append(time.time())
```

Token bucket — sliding window over the last 60 seconds. If 10 calls have been made, the next call blocks until the oldest drops off.

Why it matters: a single MASIS request with 2 retries triggers up to 9 LLM calls (3 agents × 3 iterations). At scale, 10 concurrent users generate 90 LLM calls/minute — easily hitting OpenAI's RPM limits. Without rate limiting the system degrades catastrophically under load.

Production improvement: Redis-backed distributed rate limiter (multiple API server instances share the same OpenAI quota) + exponential backoff on 429 responses.

---

## Section 4: Reliability & Production

---

**Q14. What's your fallback strategy if the Critic LLM call fails?**

Currently, any LLM exception propagates up and the graph invocation crashes → 500 error.

Production hardening:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)
```

Exponential backoff retry (3 attempts, 1–10 second waits). If all retries fail, the Critic raises, the graph catches it, and a fallback HITL response is returned. For the Critic specifically: failure defaults to treating the draft as needing retry (conservative) rather than auto-approving it (dangerous).

---

**Q15. How would you monitor this system in production?**

Every response contains `metrics` and `trace` — the observability primitives.

1. **Log the full state** on every completion — ship to CloudWatch/Datadog.
2. **Alert on**: confidence dropping below threshold across a time window; `requires_human_review` rate spiking; evaluator `faithfulness` declining; `filtered_out` rate rising (indicates query patterns are drifting from document content); `node_latency_ms` exceeding SLAs.
3. **Dashboard**: confidence distribution histogram; retry rate over time; HITL rate by workspace; `filtered_out` rate by workspace (workspace-level retrieval health signal); per-node latency percentiles.
4. **`retry_reasons` aggregation**: which signals (citation_issue, hallucination, low_confidence) most frequently trigger retries — guides where to invest improvement effort.

---

**Q16. How would you test individual agents in isolation?**

Each node is a plain Python function `(MASISState) -> MASISState`. No graph invocation needed:

```python
def test_researcher_score_filter():
    # Simulate Qdrant returning weak results
    # ... mock qdrant_client.search to return results with score 0.50
    state = {
        "user_query": "NovaTech findings",
        "workspace_id": "test-ws",
        "retry_count": 0,
        "metrics": {},
        "trace": [],
        "critique": {}
    }
    result = researcher_node(state)
    # All results below 0.60 should be filtered
    assert result["requires_human_review"] == True
    assert result["trace"][-1]["warning"] == "no_qualifying_evidence"
    assert result["trace"][-1]["results_before_filter"] == 3

def test_critic_proportional_penalty():
    # 4 uncited claims → 4 * 0.03 = 12% penalty
    state = {
        "draft_answer": "Claim one. Claim two. Claim three. Claim four.",
        "evidence": [EvidenceChunk(chunk_id="c1", ...)],
        "retry_count": 0, "metrics": {}, "trace": []
    }
    result = critic_node(state)
    # LLM confidence say 0.80 → penalty 0.88 → final ~0.70
    assert result["confidence"] < 0.80
    assert result["metrics"]["citation_violations"][-1]["uncited_claims"] == 4

def test_evaluator_hard_clamp():
    # Hallucination detected → faithfulness must be <= 0.4 regardless of LLM score
    state = {
        "user_query": "...", "final_answer": "...", "evidence": [...],
        "metrics": {
            "last_citation_audit": {
                "invalid_citations": ["fake_id"],
                "uncited_claim_count": 2,
                "hallucination_detected": True,
                "unsupported_claims": []
            }
        },
        "trace": []
    }
    result = evaluator_node(state)
    assert result["metrics"]["evaluation"]["faithfulness"] <= 0.4
```

Deterministic parts (score filter, citation engine, clamp logic) require no LLM mock. LLM-dependent parts use `unittest.mock.patch` on the LLM invoke.

---

## Section 5: Curveball Follow-Ups

---

**Q17. A simple query like "what are the findings of NovaTech" gets low confidence. Why and how did you fix it?**

The word "findings" doesn't appear in the documents. The embedding of "findings of NovaTech" points to a semantic space that returns chunks with cosine similarity around 0.51–0.55 — below the `MIN_SCORE_THRESHOLD` of 0.60. All candidates are filtered out. The Researcher sets HITL immediately with the message: *"Your query did not match any documents with sufficient relevance. Try rephrasing with more specific terms."*

The fix has two parts:
1. **User guidance**: the HITL message now explicitly tells the user to rephrase with specific terms from their documents.
2. **Correct query**: *"What are the key financial results and business performance highlights of NovaTech in FY2023?"* — these words appear directly in the documents and return high-scoring chunks.

This is working as designed. A query that doesn't match the document vocabulary shouldn't produce a confident answer. The system correctly refuses rather than generating vague, low-cited content.

---

**Q18. What if two agents disagree — Critic says confidence is high but citation engine finds invalid IDs?**

Deterministic code beats LLM self-assessment. Always.

```python
if invalid_citations:
    critique["hallucination_detected"] = True  # overrides LLM's assessment
    critique["needs_retry"] = True
    penalty_factor *= 0.5
```

If the code finds a fake citation, that's a fact — regardless of what the LLM thinks. The Evaluator then receives this via `last_citation_audit` and its faithfulness is hard-clamped to ≤ 0.40 by code, even if the LLM tried to assign 0.85. Two layers of deterministic override: the Critic's penalty, and the Evaluator's clamp.

---

**Q19. The Evaluator returned faithfulness = 0.40 even though the LLM wanted to give 0.85. Why?**

The Critic's citation engine found invalid chunk IDs in the answer — references to chunk IDs that don't exist in the retrieved evidence. This is a confirmed hallucination. The Critic stored this in `last_citation_audit`. The Evaluator received it as a hard constraint: `"If invalid citations exist, Faithfulness MUST be <= 0.4"`. After the LLM responded with 0.85, the code clamped it to 0.40 and recalculated the overall score using the weighted mean. The code's deterministic finding took precedence over the LLM's subjective assessment.

---

**Q20. How does your system perform if the vector database goes down?**

Currently: `qdrant_client.search()` throws an exception → graph crashes → 500 error.

Production fix:

```python
try:
    results = qdrant_client.search(...)
except Exception as e:
    state["requires_human_review"] = True
    state["clarification_question"] = "Document retrieval is temporarily unavailable. Please try again shortly."
    state["trace"].append({"node": "researcher", "warning": "qdrant_unavailable", "error": str(e)})
    state["evidence"] = []
    return state
```

Longer term: a circuit breaker that opens after N consecutive Qdrant failures, failing fast on all subsequent requests rather than waiting for each to time out individually. This is the production improvement I'd mention in interview even though it's not in the current implementation.

---

**Q21. How does the confidence history tell you whether retries are actually helping?**

`state["metrics"]["confidence_history"]` appends the Critic's penalized confidence after each iteration. If it looks like `[0.54, 0.71, 0.88]`, retries are working — each cycle is finding better evidence and producing better-cited answers. If it looks like `[0.54, 0.55, 0.53]`, retries aren't helping — the documents probably don't contain the answer, and the system should escalate to HITL faster rather than wasting API calls. In production, monitoring this pattern per workspace would tell you which workspaces have inadequate document coverage for their query patterns.

# 🎯 Interview Questions — MASIS Deep Dive Preparation

Questions are drawn from the case study's "Curveball Questions" and "Deep-Dive Design Questions" sections, plus additional questions likely to be asked based on the architecture. Each includes a model answer grounded in the actual implementation.

---

## Section 1: Architecture & Design

---

**Q1. Walk me through your orchestration strategy. Why did you choose a Supervisor-controlled DAG over a sequential chain or a fully autonomous multi-agent loop?**

A sequential chain (A→B→C→END) cannot express feedback loops — if C's output is poor, the chain can't go back to A with improved context. A fully autonomous loop (where agents decide their own next steps) is unpredictable and hard to debug — you can't guarantee termination or control which agent acts next.

MASIS uses a **Supervisor-controlled DAG**: the inner pipeline (Researcher→Synthesizer→Critic→Evaluator) is a fixed linear chain, and the only conditional logic lives in the Supervisor. This means:
- The inner pipeline is always predictable and testable in isolation.
- All routing intelligence is in one place — the Supervisor — making the system's decision logic easy to audit, modify, and explain.
- Termination is guaranteed by the `max_retries` ceiling, which is impossible to enforce in a fully autonomous loop.

---

**Q2. What happens if the Researcher agent enters an infinite search loop?**

It can't, by design. The Researcher is a node in a controlled graph — it executes once per cycle and returns. It has no internal loop. The cycle itself is bounded by `max_retries` in the Supervisor. Even if Qdrant were infinitely slow, the Researcher would just block until it returned or timed out — it doesn't loop.

If the concern is about the graph cycling infinitely (Supervisor always deciding "retry"), that's prevented by the `retry_count < max_retries` guard. After 2 retries (configurable), the Supervisor always escalates to HITL, which routes to END.

For production hardening, I'd add a per-node timeout wrapper and a circuit breaker on the Qdrant client — if the vector DB is unresponsive after N milliseconds, fail fast and surface a HITL message rather than hanging.

---

**Q3. How do you prevent agentic drift, where sub-tasks diverge from the original user intent?**

Several mechanisms:

1. **`user_query` is immutable in state** — it's set at init and never overwritten. Every node that generates content receives the original query, not a reformulated version.
2. **Critique-aware augmentation is additive** — on retry, the Researcher appends gap terms to the original query (`augmented_query = query + " " + focus_terms`). The original intent is preserved; only the search scope is widened.
3. **The Synthesizer always receives the original question** — even on retry 2, the synthesis prompt starts with the user's exact question.
4. **The Evaluator scores Relevance** — it explicitly checks whether the final answer addresses the user's question. Low relevance scores would surface in monitoring.
5. **The Supervisor's routing is query-agnostic** — it never reformulates the goal, only decides whether quality is sufficient.

---

**Q4. With 10,000 documents, how do you avoid losing critical context in the middle of the prompt?**

Three layers:

**Layer 1 — Selective retrieval.** Qdrant returns only the top-K most relevant chunks (5 on first run, 10 on retry) — not all 10,000 documents. The vector search narrows the haystack to the most relevant handful.

**Layer 2 — Relevance-ranked compression.** If those chunks still exceed 6,000 characters, the Synthesizer sorts them by score and keeps the top 3 full. Lower-ranked chunks are compressed to 200 characters each. This means the highest-relevance content is never degraded.

**Layer 3 — Position-aware assembly.** After compression, chunks are assembled with the highest-scoring ones first. LLMs perform best on content at the beginning and end of prompts — so the most critical evidence is positioned where attention is strongest.

In a further production improvement, I'd add MMR (Maximal Marginal Relevance) to diversify retrieved chunks — rather than 5 near-identical chunks about the same paragraph, get 5 chunks covering different aspects of the topic.

---

**Q5. How does the system handle conflicting evidence between Document A and Document B?**

The Critic's LLM audit is specifically prompted to identify `conflicting_evidence` — a list of statements where two retrieved chunks make opposing claims. When this is detected:

1. **First**, the Supervisor triggers a retry with the augmented query. The Researcher may surface additional chunks that clarify which source is authoritative (e.g., one document is more recent, or a third document resolves the ambiguity).

2. **If retries are exhausted and conflict persists**, the Supervisor sets `requires_human_review = True` with the message: *"Conflicting information was detected across documents and could not be automatically resolved. Please review the competing claims and select a preferred source."*

3. The human response can then be fed back as a preference signal in a follow-up request (e.g., "Use Document B's figure").

We deliberately chose not to auto-resolve conflicts by picking the higher-confidence source — in a strategic intelligence system, silently choosing one source over another without human knowledge could be more dangerous than surfacing the conflict.

---

**Q6. Where would you swap GPT-4-class models for smaller SLMs?**

Current model allocation:
- `gpt-4o-mini` — Synthesizer (generation), compression (summarization)
- `gpt-4o` — Critic (hallucination detection), Evaluator (quality scoring)

The reasoning:

**Keep large model for:** Tasks requiring deep reasoning — the Critic must detect subtle semantic hallucinations, logical gaps, and conflicting claims. A smaller model is more likely to miss these. The Evaluator similarly requires calibrated, nuanced scoring.

**Use smaller model for:** The Synthesizer is following explicit instructions ("cite every claim") on well-scoped content. This is a pattern-following task, not a deep-reasoning task. `gpt-4o-mini` is capable of it and costs ~15-20x less. Compression is even simpler — summarize each chunk in 200 chars — clearly within mini's capability.

**Future swap:** If I were deploying at higher volume, I'd test using a fine-tuned `gpt-3.5` or `Mistral-7B` for the Synthesizer, since it's doing a constrained, repetitive task. The Critic and Evaluator I would not compromise.

---

## Section 2: RAG & Retrieval

---

**Q7. Your Researcher only does semantic search. How would you extend it to hybrid retrieval?**

Currently: `embeddings.embed_query(query)` + Qdrant vector search.

To add hybrid retrieval:

```python
# Semantic leg (existing)
semantic_results = qdrant_client.search(
    collection_name="masis_documents",
    query_vector=("dense", dense_vector),
    limit=limit,
    ...
)

# Keyword leg (BM25 / sparse vectors)
sparse_results = qdrant_client.search(
    collection_name="masis_documents", 
    query_vector=("sparse", sparse_vector),  # using SPLADE or BM25 sparse encoder
    limit=limit,
    ...
)

# Fusion with Reciprocal Rank Fusion
final_results = reciprocal_rank_fusion(semantic_results, sparse_results)
```

Keyword search catches exact-match terms (product names, codes, dates) that semantic search may miss. Semantic search catches conceptually related content where exact words differ. Hybrid is almost always better than either alone.

Qdrant natively supports both dense and sparse vectors in a single collection and can run hybrid queries with RRF fusion. The case study explicitly lists "hybrid approaches" as a retrieval strategy — this is the implementation answer.

---

**Q8. How do you ensure repeated queries produce stable, consistent conclusions?**

Several strategies:

1. **Temperature = 0 on compression LLM** — compression is deterministic.
2. **Structured output via function calling** — the Critic and Evaluator use `.with_structured_output()` which forces outputs through OpenAI's function-calling mechanism, significantly reducing variance compared to free-form generation.
3. **Deterministic routing** — the Supervisor uses pure Python logic (no LLM), so the same state always produces the same routing decision.
4. **Seeded embeddings** — OpenAI embeddings are deterministic for the same input, so the same query always retrieves the same chunks from the same Qdrant state.

The main source of variance is the Synthesizer and Critic LLM calls, which don't have `temperature=0`. For production consistency, I'd set `temperature=0` on both and evaluate whether output quality degrades. Often it doesn't for constrained tasks like citation-grounded synthesis.

---

**Q9. How do you handle the "lost-in-the-middle" problem for 50 retrieved chunks?**

The Synthesizer's compression step addresses this directly. With 50 chunks:
- Top 3 by score: full text (highest attention, highest relevance)
- Chunks 4–50: compressed to ~200 chars each (preserves key numbers/metrics, removes padding)

The resulting context is roughly `3 × 800 + 47 × 200 = 11,800 chars` — still significant but manageable. For even larger contexts, I'd add:

1. **Re-ranking**: After initial retrieval, run a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to get more accurate relevance scores than the initial embedding similarity. The top-K after re-ranking are more reliably the most relevant.
2. **Recursive summarisation**: For very long documents, chunk → summarise → embed the summaries instead of raw text.
3. **Map-reduce synthesis**: Split the 50 chunks into groups of 5, synthesise each group separately, then synthesise the partial answers. This keeps each individual LLM call's context manageable.

---

## Section 3: State Management & Engineering

---

**Q10. Why is your graph state a TypedDict and not a Pydantic BaseModel?**

This was a real bug we hit. LangGraph's `StateGraph` requires dict-style access internally — nodes must use `state["key"]` and `state.get("key")`, not `state.field`. Pydantic `BaseModel` uses attribute access (`state.field`) and raises `AttributeError` when you try `state.get()`.

`TypedDict` gives:
- Dict-style access (compatible with LangGraph internals)
- Type hints (for IDE support and documentation)
- `total=False` (all fields optional, suitable for a state object that evolves through the pipeline)

The API boundary uses `MASISInput` (a Pydantic `BaseModel`) for input validation, converted to `MASISState` via `.to_state()`. Validation at the boundary, TypedDict inside the graph. Clean separation.

---

**Q11. How is state shared between agents? How do you prevent uncontrolled context growth?**

State is a single shared `MASISState` dict passed through every node by LangGraph. Each node mutates it and returns it — LangGraph merges the result back into the running state.

Context growth is prevented at the prompt level, not the state level:
- The state stores full `EvidenceChunk` objects (necessary for the citation engine to validate IDs).
- The **prompts** use compressed text, not the full chunk text.
- The Synthesizer decides what goes into the LLM's context window; the state just stores the raw evidence.

If state growth itself became a concern (e.g., `trace` growing very large over many retries), I'd add a state trimming step in the Supervisor that keeps only the last N trace entries.

---

**Q12. What's your rate limiting strategy, and why is it important in multi-agent systems?**

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

A token bucket — tracks LLM call timestamps in a sliding window. If 10 calls have been made in the last 60 seconds, the next call blocks until the oldest drops off.

Why it matters in multi-agent systems: a single user request in MASIS can trigger 6–9 LLM calls (2 retry cycles × 3 agents). At scale, 10 concurrent users could generate 90 LLM calls/minute — easily hitting OpenAI's RPM limits and incurring 429 errors. Without rate limiting, the system degrades catastrophically under load. With it, requests are queued gracefully.

In production, I'd move this to a Redis-backed distributed rate limiter (since multiple API server instances share the same OpenAI quota) and use exponential backoff on 429 responses.

---

## Section 4: Reliability & Production

---

**Q13. What's your fallback/retry strategy for sub-task failures? What if the Critic LLM call fails?**

Currently, if any LLM call throws an exception (network error, 500, rate limit), it propagates up and crashes the graph invocation. The caller gets a 500 error.

A production hardening approach:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)
```

Each LLM call gets wrapped with exponential backoff retry (3 attempts, 1–10 second waits). If all retries fail, the node raises an exception, the graph catches it, and a fallback response is returned to the user with a clear error message.

For the Critic specifically: if it fails, the Supervisor could fall back to treating the draft as needing a retry (conservative default) rather than auto-approving it (which would be dangerous).

---

**Q14. How would you monitor this system in production?**

Every state contains `state["metrics"]` and `state["trace"]` — these are the observability primitives. In production:

1. **Log the full state on every completion** — ship `metrics` and `trace` to a logging backend (CloudWatch, Datadog, etc.).
2. **Alert on**: avg `confidence` dropping below threshold across a time window; `requires_human_review` rate spiking; evaluator `faithfulness` score declining; `node_latency_ms` exceeding SLAs.
3. **Dashboard**: confidence distribution histogram, retry rate over time, HITL escalation rate by workspace, per-node latency percentiles.
4. **Traces**: each `trace` entry is a structured log event — can be stored in an OLAP system and queried to understand why specific queries failed.

---

**Q15. How would you test individual agents in isolation?**

Each node is a plain Python function `(MASISState) -> MASISState`. To test:

```python
def test_critic_node_flags_invalid_citation():
    state = {
        "draft_answer": "Revenue grew [chunk_999].",  # chunk_999 doesn't exist
        "evidence": [EvidenceChunk(chunk_id="chunk_1", ...)],
        "retry_count": 0,
        "metrics": {},
        "trace": []
    }
    result = critic_node(state)
    
    assert result["critique"]["hallucination_detected"] == True
    assert result["critique"]["needs_retry"] == True
    assert result["confidence"] < 0.5
```

No graph invocation needed. Each node can be unit-tested with a crafted state dict. The deterministic citation engine parts can be tested without any LLM mock. The LLM-dependent parts use `unittest.mock.patch` to mock `critic_llm.invoke()`.

---

**Q16. How would you handle a 10x scale increase in document uploads?**

The bottleneck is Qdrant retrieval latency and embedding generation. Strategies:

1. **Qdrant horizontal scaling** — Qdrant supports distributed clusters. Shard the `masis_documents` collection across nodes by `workspace_id`.
2. **Async embedding** — move document ingestion (chunking + embedding + upsert) to a background job queue (Celery, SQS). Don't block the upload endpoint on embedding.
3. **Embedding cache** — cache query embeddings for frequently-asked questions (Redis + LRU). The same query embedding is reused across users asking the same question.
4. **Pre-filtering at index time** — add more metadata to chunks (document date, section type, author) to enable tighter pre-filters, reducing the search space per query.

---

## Section 5: Curveball Follow-Ups

---

**Q17. The Synthesizer said "I cannot provide information on this topic" — why might that happen and how would you fix it?**

The prompt includes: *"If there is insufficient evidence, explicitly state so."* If Qdrant returns chunks that are tangentially related but don't actually contain the answer, the Synthesizer (correctly) refuses to fabricate. This is working as designed.

Fix depends on root cause:
- If the documents genuinely don't contain the answer → HITL is the right outcome.
- If the retrieval failed to surface the right chunks → the retry with augmented query should find them.
- If the query is too specific for the chunking granularity → improve the chunking strategy (smaller chunks, overlapping windows).

---

**Q18. What if two agents disagree — the Critic says confidence is high but the citation engine finds invalid citations?**

This is explicitly handled. The Critic's LLM-reported confidence is treated as an input, then **overridden** by the citation engine's deterministic findings:

```python
if invalid_citations:
    critique["hallucination_detected"] = True  # overrides LLM's assessment
    critique["needs_retry"] = True
    penalty_factor *= 0.5                       # hard penalty on confidence
```

The deterministic check beats the LLM's self-assessment. If the code finds a fake citation, that's a fact — the LLM is wrong. This is an intentional architectural decision: hard evidence (code-verified) always overrides soft evidence (LLM reasoning).

---

**Q19. How does your system perform if the vector database goes down?**

Currently: the Researcher's `qdrant_client.search()` call would throw an exception, crashing the graph. The user gets a 500.

Production fix: wrap the Qdrant call in a try/except, catch `QdrantException` or `ConnectionError`, set `requires_human_review = True` with message "Document retrieval is temporarily unavailable. Please try again shortly." The graph then routes to END gracefully.

Longer term: a circuit breaker that opens after N consecutive Qdrant failures, failing fast on all subsequent requests until the DB recovers, rather than waiting for each request to time out individually.

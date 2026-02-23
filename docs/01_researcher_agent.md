# 🔍 Agent 1: Researcher Node — Deep Dive

## Role & Persona

The Researcher is the **"Librarian"** of MASIS. Its sole responsibility is to retrieve the most relevant, workspace-scoped evidence from the Qdrant vector database. It does not generate answers. It does not judge quality. It fetches raw material and hands it off to the Synthesizer.

This separation of concerns is deliberate — by keeping retrieval and generation as distinct agents, each can be independently improved, monitored, and debugged.

---

## What Problem Does It Solve?

### Problem 1: Retrieval Without Context of Past Failures

On a naive first pass, a system embeds the user's query and retrieves the top-K closest chunks. But what if those chunks were insufficient? The system would retry with the **exact same query**, pulling the **exact same chunks**, and fail again.

**How MASIS solves it — Critique-Aware Query Augmentation:**

```python
if retry_count > 0 and critique:
    focus_terms = (
        critique.get("unsupported_claims", []) +
        critique.get("logical_gaps", [])
    )
    if focus_terms:
        augmented_query += " " + " ".join(str(t) for t in focus_terms)
```

On every retry the Critic's unsupported claims and logical gaps are appended to the original query before re-embedding. The new vector shifts toward missing evidence.

**Example:**
- Original: `"What is the company's AI strategy?"`
- Critic flagged: `["no mention of compute budget", "missing timeline"]`
- Augmented: `"What is the company's AI strategy? no mention of compute budget missing timeline"`
- Qdrant now surfaces chunks about compute spend and roadmap timelines that the first pass missed.

---

### Problem 2: Weak Evidence Poisoning the Synthesizer

Without a quality filter, Qdrant returns the top-K results regardless of their actual relevance score. A vague query like *"findings of NovaTech"* returns chunks scoring 0.51, 0.48, 0.53 — all weakly matched. The Synthesizer builds a vague answer, the Critic penalises it, confidence tanks, and HITL triggers — for a query the documents could have answered with tighter retrieval.

**How MASIS solves it — Minimum Score Threshold:**

```python
MIN_SCORE_THRESHOLD = 0.60

threshold = MIN_SCORE_THRESHOLD if retry_count == 0 else MIN_SCORE_THRESHOLD - 0.05

if r.score < threshold:
    filtered_out += 1
    continue
```

Each chunk is checked against a minimum cosine similarity of **0.60**. Chunks below this are discarded before reaching the Synthesizer. On retry the threshold drops to **0.55** because the augmented query is more specific — slightly weaker-scoring chunks may still be genuinely relevant to the augmented terms.

`filtered_out` tracks how many chunks were dropped and appears in the trace so you can see exactly how aggressive the filter was on each pass.

---

### Problem 3: Fixed Retrieval Limits Regardless of Complexity

With the score filter now potentially removing candidates, the pipeline needs more raw candidates to ensure enough qualify.

**How MASIS solves it — Dynamic Limit Expansion:**

```python
limit = 10 if retry_count == 0 else 20
```

First pass: 10 candidates fetched from Qdrant. On retry: 20. The score filter narrows these down to only high-quality matches. This balances speed on the happy path with broader coverage when the first pass was insufficient.

> The original implementation fetched 5/10. Doubled to ensure the score filter has sufficient candidates to work with.

---

### Problem 4: Cross-Workspace Data Leakage

In a multi-tenant environment, multiple organisations share the same Qdrant collection. Without filtering, Company A's query could return Company B's documents.

**How MASIS solves it — Workspace-Scoped Filtering:**

```python
query_filter=Filter(
    must=[
        FieldCondition(
            key="workspace_id",
            match=MatchValue(value=workspace_id)
        )
    ]
)
```

Applied at the Qdrant layer — not post-processing. Only chunks tagged with the correct `workspace_id` are ever returned, regardless of semantic similarity to other tenants' data.

---

### Problem 5: Duplicate Chunks in Results

Qdrant can occasionally return the same chunk ID twice due to approximate nearest-neighbour overlap.

**How MASIS solves it — Deduplication by ID:**

```python
seen_ids = set()
for r in results:
    if str(r.id) in seen_ids:
        continue
    seen_ids.add(str(r.id))
    # score threshold check, then append
```

Duplicates are caught before the score threshold check, preventing the same text appearing twice in the context window.

---

### Problem 6: Zero Results Causing Wasted LLM Calls

If no chunks survive the score filter, the pipeline would proceed to the Synthesizer with empty evidence — producing an uncited answer the Critic would flag, triggering pointless retries.

**How MASIS solves it — Immediate HITL with Contextual Message:**

```python
if not evidence:
    clarification = (
        "Your query did not match any documents with sufficient relevance. "
        "Try rephrasing with more specific terms from your documents, "
        "or upload documents that cover this topic."
    ) if filtered_out > 0 else (
        "No relevant documents were found for your query in this workspace. "
        "Please upload relevant documents or refine your question."
    )
    state["requires_human_review"] = True
    state["clarification_question"] = clarification
```

Two distinct messages:
- **All filtered out** (`filtered_out > 0`): candidates existed but none were relevant enough → rephrase.
- **Zero from Qdrant**: workspace empty or query completely off-topic → upload documents.

The graph's router reads `requires_human_review` and exits immediately to END. No further LLM calls are made.

---

## State Inputs & Outputs

| Field | Direction | Description |
|---|---|---|
| `user_query` | Input | Original question |
| `workspace_id` | Input | Tenant scoping key |
| `retry_count` | Input | Determines augmentation, limit, and threshold |
| `critique` | Input | Previous Critic output (unsupported claims / gaps) |
| `evidence` | **Output** | Score-filtered `EvidenceChunk` list |
| `requires_human_review` | **Output** | `True` if no qualifying evidence |
| `clarification_question` | **Output** | Contextual HITL message |
| `metrics` | **Output** | `retrieval_scores`, `avg_retrieval_score`, `node_latency_ms` |
| `trace` | **Output** | Retrieval stats including `filtered_out`, `threshold_used` |

---

## Telemetry Emitted

**Happy path:**
```json
{
  "node": "researcher",
  "retry_count": 0,
  "chunks": 7,
  "filtered_out": 3,
  "avg_score": 0.724,
  "threshold_used": 0.60,
  "augmented_query_used": false,
  "duration_ms": 312
}
```

**Retry pass:**
```json
{
  "node": "researcher",
  "retry_count": 1,
  "chunks": 12,
  "filtered_out": 8,
  "avg_score": 0.681,
  "threshold_used": 0.55,
  "augmented_query_used": true,
  "duration_ms": 418
}
```

**All filtered / zero results:**
```json
{
  "node": "researcher",
  "warning": "no_qualifying_evidence",
  "results_before_filter": 10,
  "threshold_used": 0.60,
  "duration_ms": 289
}
```

---

## What the Researcher Does NOT Do

- Does **not** generate, summarize, or interpret content.
- Does **not** decide whether evidence is sufficient — that is the Critic's job.
- Does **not** perform keyword or hybrid search — currently semantic only. Known limitation: keyword search catches exact-match terms (product names, dates, codes) that semantic search misses. BM25 sparse vectors with Reciprocal Rank Fusion would be the production improvement.

# 🔍 Agent 1: Researcher Node — Deep Dive

## Role & Persona

The Researcher is the **"Librarian"** of MASIS. Its sole responsibility is to retrieve the most relevant, workspace-scoped evidence from the Qdrant vector database. It does not generate answers. It does not judge quality. It fetches raw material and hands it off to the Synthesizer.

This separation of concerns is deliberate — by keeping retrieval and generation as distinct agents, each can be independently improved, monitored, and debugged.

---

## What Problem Does It Solve?

### Problem 1: Retrieval Without Context of Past Failures

On a naive first pass, a system would embed the user's query and retrieve the top-K closest chunks — no more thought given to it. But what if those chunks were insufficient? What if the answer built on them had hallucinations or logical gaps? The system would retry with the **exact same query**, pulling the **exact same chunks**, and fail again.

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

On every retry, the Critic's output from the previous iteration is read. The specific claims that were **unsupported** and the **logical gaps** identified are extracted and appended to the original query before embedding. This means the new embedding vector is semantically shifted toward the missing evidence, increasing the probability of surfacing chunks that fill those exact gaps.

**Example:**
- Original query: `"What is the company's AI strategy?"`
- Critic flagged: unsupported_claims = `["no mention of compute budget", "missing timeline"]`
- Augmented query: `"What is the company's AI strategy? no mention of compute budget missing timeline"`
- Result: Qdrant now returns chunks about compute spend and roadmap timelines that the first pass missed.

---

### Problem 2: Fixed Retrieval Limits Regardless of Complexity

Fetching 5 chunks for a simple query is fine. But if that answer was poor and the system is retrying, 5 chunks is likely not enough — there's clearly insufficient coverage.

**How MASIS solves it — Dynamic Limit Expansion:**

```python
limit = 5 if retry_count == 0 else 10
```

Simple but effective. On first pass: 5 chunks (fast, low cost). On any retry: 10 chunks (broader surface area). This balances speed on the happy path with thoroughness when needed.

---

### Problem 3: Cross-Workspace Data Leakage

In a multi-tenant SaaS environment, multiple organizations share the same Qdrant collection (`masis_documents`). Without filtering, a query from Company A could retrieve chunks uploaded by Company B — a serious data privacy violation.

**How MASIS solves it — Workspace-Scoped Filtering:**

```python
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
```

Every search is hard-filtered to `workspace_id`. This is a **metadata filter** applied at the Qdrant layer — not post-processing. Only chunks tagged with the correct workspace are ever returned, regardless of semantic similarity to other tenants' data.

---

### Problem 4: Duplicate Chunks in Results

Qdrant can occasionally return the same chunk ID twice (e.g. if a document was indexed multiple times, or due to approximate nearest-neighbour overlap in certain configurations).

**How MASIS solves it — Deduplication by ID:**

```python
seen_ids = set()
for r in results:
    if str(r.id) not in seen_ids:
        seen_ids.add(str(r.id))
        evidence.append(...)
```

A `seen_ids` set tracks chunk IDs already added. Duplicates are silently skipped. This prevents the same text appearing twice in the context window, which could artificially inflate confidence scores or produce repeated citations.

---

### Problem 5: Zero Results Causing Infinite Loops

If no chunks are returned (empty collection, wrong workspace_id, or query too specific), the pipeline would previously proceed to the Synthesizer with empty evidence. The Synthesizer would produce an answer with no citations. The Critic would flag everything as a hallucination. The Supervisor would trigger a retry. The Researcher would again return zero results. This would loop until `max_retries` was exhausted — wasting API calls and time.

**How MASIS solves it — Immediate HITL Escalation on Zero Results:**

```python
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
```

Zero results is detected immediately. Instead of proceeding, the system sets `requires_human_review = True` and writes a clear `clarification_question` explaining what the user should do. The graph's routing logic then reads `requires_human_review` and exits to `END`, surfacing the message to the user. No further LLM calls are made.

---

## State Inputs & Outputs

| Field | Direction | Description |
|---|---|---|
| `user_query` | Input | The original question from the user |
| `workspace_id` | Input | Tenant scoping key for Qdrant filter |
| `retry_count` | Input | Determines query augmentation and limit |
| `critique` | Input | Unsupported claims / gaps from previous Critic run |
| `evidence` | **Output** | List of `EvidenceChunk` objects |
| `requires_human_review` | **Output** | Set to `True` if zero results found |
| `clarification_question` | **Output** | Human-readable explanation if HITL triggered |
| `metrics` | **Output** | `retrieval_scores`, `avg_retrieval_score`, `node_latency_ms` |
| `trace` | **Output** | Appended entry with retrieval stats |

---

## Telemetry Emitted

```json
{
  "node": "researcher",
  "retry_count": 1,
  "chunks": 10,
  "avg_score": 0.847,
  "augmented_query_used": true,
  "duration_ms": 312
}
```

This trace entry is appended to `state["trace"]` and is part of the full auditable trail available in the final API response.

---

## What the Researcher Does NOT Do

- It does **not** generate, summarize, or interpret any content.
- It does **not** decide whether retrieved evidence is sufficient (that's the Critic's job).
- It does **not** perform keyword or hybrid search — currently semantic only. This is a known limitation worth raising in design discussions (see interview questions doc).

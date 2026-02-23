# 🧠 MASIS — Multi-Agent Strategic Intelligence System

A production-grade multi-agent orchestration layer built on **LangGraph** that functions as a **"Chief of Staff"** — conducting deep-dive research, critiquing its own reasoning, and producing recommendations with a full auditable evidence trail.

---

## What It Does

MASIS takes a user query and a workspace of documents, then runs it through a self-correcting pipeline of specialized AI agents to produce a high-confidence, fully cited answer. If the system cannot reach acceptable quality after multiple refinement attempts, it escalates to the user with a specific explanation of what went wrong and what to do next.

```
User Query → Supervisor → Researcher → Synthesizer → Critic → Evaluator → Supervisor → Answer
                  ↑                                                              |
                  └────────────────── retry (up to N times) ───────────────────┘
```

---

## Architecture

### Agent Roles

| Agent | Persona | Responsibility |
|---|---|---|
| **Supervisor** | The Brain | Reads quality signals, decides to retry, escalate (HITL), or finalize |
| **Researcher** | The Librarian | Retrieves workspace-scoped evidence from Qdrant via semantic search |
| **Synthesizer** | The Writer | Builds a cited, evidence-grounded answer draft |
| **Critic** | The Auditor | Detects hallucinations, validates citations, flags logical gaps |
| **Evaluator** | The Scorecard | Scores the final answer on Faithfulness, Relevance, Completeness, Reasoning |

### Key Design Principles

- **Supervisor-controlled DAG** — all routing logic lives in one place; the inner pipeline is always deterministic
- **Retry-first resolution** — quality issues and conflicting evidence trigger retries before escalating to a human
- **Two-layer auditing** — the Critic combines LLM semantic auditing with a deterministic regex-based citation engine
- **Critique-aware retrieval** — on retry, the Researcher augments its query with the Critic's specific complaints, targeting the exact gaps
- **Typed state** — `MASISState` is a `TypedDict`; `MASISInput` is a Pydantic `BaseModel` for API validation

---

## Project Structure

```
app/
└── orchestrator/
    ├── state.py        # MASISState (TypedDict) + MASISInput (Pydantic) + EvidenceChunk
    ├── graph.py        # LangGraph StateGraph definition, node wiring, conditional routing
    └── nodes.py        # All five agent node functions + rate limiter + shared helpers

docs/
    ├── 01_researcher_agent.md      # Researcher deep dive
    ├── 02_synthesizer_agent.md     # Synthesizer deep dive
    ├── 03_critic_agent.md          # Critic deep dive
    ├── 04_supervisor_agent.md      # Supervisor deep dive
    ├── 05_evaluator_agent.md       # Evaluator deep dive
    ├── 06_graph_flow.md            # Graph architecture & all flow paths
    └── 07_interview_questions.md   # 19 interview Q&As with model answers
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM (Generation) | `gpt-4o-mini` — Synthesizer, Compression |
| LLM (Auditing) | `gpt-4o` — Critic, Evaluator |
| Vector Database | [Qdrant](https://qdrant.tech/) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Infrastructure | Docker Compose |

---

## State Schema

```python
class MASISState(TypedDict, total=False):
    user_query: str               # Original question — never mutated
    workspace_id: str             # Tenant scoping key for Qdrant filter
    evidence: List[EvidenceChunk] # Retrieved chunks (Researcher output)
    draft_answer: Optional[str]   # Current answer draft (Synthesizer output)
    final_answer: Optional[str]   # Audited best answer (Critic output)
    confidence: float             # Penalized confidence score (0–1)
    retry_count: int              # Current iteration (starts at 0)
    max_retries: int              # Retry ceiling (default: 2)
    critique: Optional[dict]      # Structured Critic output
    requires_human_review: bool   # HITL flag — routes to END when True
    clarification_question: str   # Human-readable HITL message
    trace: List[dict]             # Full audit trail — one entry per node per cycle
    metrics: dict                 # Telemetry — latencies, scores, compression stats
```

---

## Flow Paths

### ✅ Happy Path
Query → Supervisor (passthrough) → Researcher (5 chunks) → Synthesizer → Critic (confidence ≥ 0.75, no issues) → Evaluator → Supervisor (finalize) → **Answer returned**

**LLM calls: 3 | Qdrant calls: 1**

### 🔁 Retry Path
Same as above, but Critic flags hallucination or low confidence → Supervisor triggers retry → Researcher fetches 10 chunks with augmented query → Synthesizer receives critique feedback → improved answer → Supervisor finalizes

**LLM calls: 6 | Qdrant calls: 2**

### 🙋 HITL Path
After `max_retries` cycles without reaching quality threshold → Supervisor sets `requires_human_review = True` → graph exits with a specific `clarification_question` explaining the failure and what the user should do

### 🚫 Zero-Results Path
Qdrant returns no chunks → Researcher immediately sets HITL → graph exits without any LLM synthesis calls

---

## Self-Correction Loop

```
Iteration 1:
  Researcher → fetches 5 chunks
  Synthesizer → generates answer
  Critic → confidence: 0.58, hallucination: True, unsupported_claims: ["claim A", "claim B"]
  Supervisor → retry_count 0 < 2 → retry_count = 1

Iteration 2:
  Researcher → augmented_query = original + "claim A claim B"
              → fetches 10 chunks (expanded limit)
  Synthesizer → receives critique_feedback with previous issues
              → corrects unsupported claims
  Critic → confidence: 0.84, hallucination: False
  Supervisor → confidence 0.84 > 0.75, no flags → finalize ✅
```

---

## Citation Engine

Every answer is validated against the retrieved evidence using a deterministic regex-based citation engine in the Critic node — independent of the LLM's own assessment:

```python
# Extract all [chunk_id] references from the answer
citations = re.findall(r"\[(.*?)\]", answer)

# Compare against actually retrieved chunk IDs
valid_ids = {e.chunk_id for e in evidence}
invalid_citations = [c for c in citations if c not in valid_ids]

# If any fake citations found → hallucination_detected = True, 50% confidence penalty
if invalid_citations:
    critique["hallucination_detected"] = True
    penalty_factor *= 0.5
```

This catches fabricated chunk references that the LLM's own audit might miss.

---

## Rate Limiting

A thread-safe token bucket prevents API exhaustion in multi-agent workflows:

```python
MAX_CALLS_PER_MINUTE = 10
```

Every LLM call passes through `_rate_limit()` before executing. If 10 calls have been made in the last 60 seconds, the next call blocks until capacity frees up.

---

## Evaluation Metrics (LLM-as-a-Judge)

Every response is scored by the Evaluator on four dimensions:

| Metric | Description |
|---|---|
| **Faithfulness** | Every claim is directly supported by cited evidence |
| **Relevance** | The answer addresses the user's actual question |
| **Completeness** | All aspects of the question are covered |
| **Reasoning Quality** | The answer is structured, clear, and logically sound |

Scores are stored in `state["metrics"]["evaluation"]` and returned in every API response for monitoring and continuous improvement.

---

## Getting Started

### Prerequisites

- Docker & Docker Compose
- OpenAI API key

### Environment Variables

```env
OPENAI_API_KEY=sk-...
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### Run with Docker Compose

```bash
docker compose up --build
```

Qdrant runs as a sidecar service named `qdrant` (referenced by hostname in `nodes.py`).

### Invoke the Graph

```python
from app.orchestrator.graph import graph
from app.orchestrator.state import MASISInput

payload = MASISInput(
    user_query="What is the company's AI investment strategy?",
    workspace_id="workspace_abc123",
    max_retries=2
)

result = graph.invoke(payload.to_state())

print(result["final_answer"])
print(result["confidence"])
print(result["trace"])          # full audit trail
print(result["metrics"])        # telemetry
```

### API Route Integration

In your FastAPI route (`masis_routes.py`):

```python
from app.orchestrator.state import MASISInput
from app.orchestrator.graph import graph

@router.post("/query")
async def masis_query(body: MASISInput):
    result = await run_in_threadpool(graph.invoke, body.to_state())

    if result.get("requires_human_review"):
        return {
            "status": "needs_clarification",
            "message": result["clarification_question"]
        }

    return {
        "status": "success",
        "answer": result["final_answer"],
        "confidence": result["confidence"],
        "evaluation": result["metrics"].get("evaluation"),
        "trace": result["trace"]
    }
```

---

## Observability

Every state object carries a complete audit trail:

```json
{
  "trace": [
    { "node": "researcher", "chunks": 5, "avg_score": 0.847, "duration_ms": 312 },
    { "node": "synthesizer", "context_compressed": false, "citations": 6, "duration_ms": 1854 },
    { "node": "critic", "confidence": 0.58, "hallucination": true, "invalid_citations": 2, "duration_ms": 1103 },
    { "node": "evaluator", "overall_score": 0.71, "duration_ms": 987 },
    { "node": "supervisor", "decision": "retry", "retry_count": 1, "reason": "quality_issue_detected" },
    { "node": "researcher", "chunks": 10, "augmented_query_used": true, "duration_ms": 445 },
    { "node": "synthesizer", "context_compressed": true, "citations": 8, "duration_ms": 2103 },
    { "node": "critic", "confidence": 0.884, "hallucination": false, "duration_ms": 989 },
    { "node": "evaluator", "overall_score": 0.91, "duration_ms": 1021 },
    { "node": "supervisor", "decision": "finalize", "confidence": 0.884 }
  ]
}
```

---

## Documentation

Full deep-dive documentation for each component lives in `/docs`:

- [`01_researcher_agent.md`](docs/01_researcher_agent.md) — Retrieval, augmentation, HITL on zero results
- [`02_synthesizer_agent.md`](docs/02_synthesizer_agent.md) — Context compression, critique feedback, citation-mandatory prompting
- [`03_critic_agent.md`](docs/03_critic_agent.md) — Two-layer auditing, citation engine, penalty logic
- [`04_supervisor_agent.md`](docs/04_supervisor_agent.md) — Routing decision tree, HITL escalation, retry logic
- [`05_evaluator_agent.md`](docs/05_evaluator_agent.md) — LLM-as-Judge, scoring rubric, Critic vs Evaluator distinction
- [`06_graph_flow.md`](docs/06_graph_flow.md) — Full graph diagram, all flow paths, state whiteboard
- [`07_interview_questions.md`](docs/07_interview_questions.md) — 19 interview Q&As with model answers

---

## Termination Guarantee

The graph always terminates. No infinite loops are possible:

1. `retry_count` increments on every retry decision
2. `max_retries` is a hard ceiling, guarded against `None` (`state.get("max_retries") or 2`)
3. When `retry_count >= max_retries`, the Supervisor always escalates to HITL → `END`
4. `requires_human_review = True` short-circuits all routing to `END` immediately
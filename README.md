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
| **Researcher** | The Librarian | Retrieves workspace-scoped, score-filtered evidence from Qdrant |
| **Synthesizer** | The Writer | Builds a cited, evidence-grounded answer as a strategic intelligence analyst |
| **Critic** | The Auditor | Detects hallucinations, validates citations with proportional penalty, stores citation audit for Evaluator |
| **Evaluator** | The Scorecard | Scores the final answer on Faithfulness, Relevance, Completeness, Reasoning — with hard clamps enforced from citation audit findings |

### Key Design Principles

- **Supervisor-controlled DAG** — all routing logic lives in one place; the inner pipeline is always deterministic
- **Score-filtered retrieval** — chunks below 0.60 cosine similarity are dropped before reaching the Synthesizer, preventing weak evidence from poisoning answers
- **Retry-first resolution** — quality issues and conflicting evidence trigger retries before escalating to a human
- **Two-layer auditing** — the Critic combines LLM semantic auditing with a deterministic regex citation engine; findings flow to the Evaluator via `last_citation_audit`
- **Critique-aware retrieval** — on retry, the Researcher augments its query with the Critic's specific unsupported claims and logical gaps, targeting exact evidence gaps
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
    ├── 01_researcher_agent.md      # Researcher deep dive — score filter, adaptive threshold, contextual HITL
    ├── 02_synthesizer_agent.md     # Synthesizer deep dive — strategic persona, compression, critique feedback
    ├── 03_critic_agent.md          # Critic deep dive — proportional penalty, citation audit passthrough
    ├── 04_supervisor_agent.md      # Supervisor deep dive — threshold 0.65, retry reasons telemetry
    ├── 05_evaluator_agent.md       # Evaluator deep dive — citation audit injection, hard clamps, weighted scoring
    ├── 06_graph_flow.md            # Graph architecture, all flow paths, configuration coupling table
    └── 07_interview_questions.md   # 21 interview Q&As with model answers
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
    evidence: List[EvidenceChunk] # Score-filtered retrieved chunks (Researcher output)
    draft_answer: Optional[str]   # Current answer draft (Synthesizer output)
    final_answer: Optional[str]   # Audited best answer (Critic output)
    confidence: float             # Penalized confidence score (0–1)
    retry_count: int              # Current iteration (starts at 0)
    max_retries: int              # Retry ceiling (default: 2)
    critique: Optional[dict]      # Structured Critic output
    requires_human_review: bool   # HITL flag — routes to END when True
    clarification_question: str   # Contextual HITL message (3 distinct variants)
    trace: List[dict]             # Full audit trail — one entry per node per cycle
    metrics: dict                 # Telemetry: latencies, scores, compression stats,
                                  # confidence_history, retry_reasons, last_citation_audit
```

---

## Key Configuration Values

| Parameter | Value | Location | Notes |
|---|---|---|---|
| `MIN_SCORE_THRESHOLD` | 0.60 (0.55 on retry) | `researcher_node` | Drops below this → chunk discarded |
| `LOW_CONF_THRESHOLD` | 0.65 | `supervisor_node` | Lowered from 0.75 — score filter upstream improves evidence quality |
| Qdrant fetch limit | 10 first pass / 20 on retry | `researcher_node` | Higher limits compensate for filter drop-off |
| `MAX_CONTEXT_CHARS` | 6000 | `synthesizer_node` | Above this, compression activates |
| `MAX_CALLS_PER_MINUTE` | 10 | `_rate_limit()` | Thread-safe token bucket |
| `max_retries` | 2 (default) | state / API input | Guarded against `None` via `or 2` |

> **Coupling note:** `MIN_SCORE_THRESHOLD` and `LOW_CONF_THRESHOLD` are coupled. Raising the score threshold improves evidence quality entering the pipeline, allowing a correspondingly higher confidence threshold. Lowering the score threshold requires keeping the confidence threshold equal or lower to avoid over-triggering HITL.

---

## Flow Paths

### ✅ Happy Path
Query → Supervisor (passthrough) → Researcher (fetches 10 candidates, applies score filter ≥ 0.60) → Synthesizer (strategic analyst persona, citation-mandatory) → Critic (confidence ≥ 0.65, no citation issues) → Evaluator (scores with citation audit, no clamps needed) → Supervisor (finalize) → **Answer returned**

**LLM calls: 3 | Qdrant calls: 1**

### 🔁 Retry Path
Same as above, but Critic flags hallucination or confidence < 0.65 → Supervisor logs structured `retry_reason` → Researcher fetches 20 candidates (threshold drops to 0.55), augments query with Critic's `unsupported_claims` + `logical_gaps` → Synthesizer receives full critique feedback → improved answer → Supervisor finalizes

**LLM calls: 6 | Qdrant calls: 2**

### 🙋 HITL Path
After `max_retries` cycles without reaching quality threshold → Supervisor sets `requires_human_review = True` with a contextual `clarification_question` (three distinct variants: conflict / low-confidence / zero-results) → graph exits → **frontend shows best draft with amber warning + full quality assessment panel**

### 🚫 Zero-Results / All-Filtered Path
Qdrant returns candidates but all score < `MIN_SCORE_THRESHOLD` → Researcher sets HITL immediately with message distinguishing "rephrase your query" (candidates existed but scored too low) from "upload documents" (no Qdrant results at all) → graph exits without any LLM synthesis calls

---

## Self-Correction Loop

```
Iteration 1:
  Researcher → fetches 10 candidates from Qdrant
               → score filter (threshold 0.60): 7 qualify, 3 filtered_out
               → avg_score: 0.724
  Synthesizer → "strategic intelligence analyst" — cites every claim,
                explicitly hedges when evidence is partial
  Critic → confidence: 0.58 (LLM raw) × 0.50 (invalid citation penalty) = 0.29
           invalid_citations: 2, uncited_claims: 3
           → stores last_citation_audit for Evaluator
  Evaluator → reads last_citation_audit → faithfulness hard-clamped to 0.40
              overall_score: 0.61 (faithfulness weighted 35%)
  Supervisor → confidence 0.29 < 0.65
             → logs retry_reason: {confidence: 0.29, citation_issue: true, hallucination: true}
             → retry_count = 1

Iteration 2:
  Researcher → threshold drops to 0.55, limit expands to 20
               augmented_query = original + "claim A claim B" (from unsupported_claims)
               → 14 qualify, 6 filtered_out, avg_score: 0.681
  Synthesizer → receives critique_feedback: "Correct: Hallucination True, Unsupported: [...]"
  Critic → confidence: 0.84, hallucination: False, invalid_citations: 0
           → last_citation_audit: clean
  Evaluator → no clamps needed → faithfulness: 0.89, overall_score: 0.84
  Supervisor → confidence 0.84 > 0.65, no flags → finalize ✅
```

---

## Citation Engine

Every answer is validated against retrieved evidence using a deterministic regex citation engine in the Critic — independent of the LLM's own assessment:

```python
# Extract all [chunk_id] references from the answer
citations = re.findall(r"\[(.*?)\]", answer)
valid_ids = {e.chunk_id for e in evidence}
invalid_citations = [c for c in citations if c not in valid_ids]

# Fabricated citations → hard 50% confidence penalty
if invalid_citations:
    critique["hallucination_detected"] = True
    penalty_factor *= 0.5

# Uncited sentences → proportional penalty: 3% per sentence, capped at 40%
# Hedge phrases excluded: "insufficient evidence", "lack sufficient evidence",
# "partially covers", "not provided", "cannot provide"
if uncited_claims:
    uncited_penalty = min(0.40, len(uncited_claims) * 0.03)
    penalty_factor *= (1.0 - uncited_penalty)

confidence = confidence * penalty_factor
```

Findings are stored in `state["metrics"]["last_citation_audit"]` and passed to the Evaluator as hard scoring constraints — ensuring the deterministic code findings are always reflected in the LLM-as-Judge scores.

---

## Evaluator Hard Clamps

After the Evaluator LLM scores the answer, code enforces these constraints regardless of what the LLM returned:

```python
if hallucination_detected or invalid_citations:
    evaluation["faithfulness"] = min(evaluation["faithfulness"], 0.40)

if uncited_count >= 10:
    evaluation["faithfulness"] = min(evaluation["faithfulness"], 0.30)
elif uncited_count >= 5:
    evaluation["faithfulness"] = min(evaluation["faithfulness"], 0.50)

# Recalculate overall as weighted mean after clamping
evaluation["overall_score"] = round(
    evaluation["faithfulness"]    * 0.35 +
    evaluation["relevance"]       * 0.25 +
    evaluation["completeness"]    * 0.25 +
    evaluation["reasoning_quality"] * 0.15,
    3
)
```

Faithfulness is weighted highest (35%) because a hallucinated answer, however well-written, is fundamentally dangerous in a strategic intelligence context.

---

## Rate Limiting

A thread-safe token bucket prevents API exhaustion. A single MASIS request with 2 retries can trigger up to 9 LLM calls (3 agents × 3 iterations):

```python
MAX_CALLS_PER_MINUTE = 10
```

Every LLM call passes through `_rate_limit()` before executing. If 10 calls have been made in the last 60 seconds, the next call blocks until capacity frees up.

---

## Evaluation Metrics (LLM-as-a-Judge)

Every response — including HITL responses — is scored by the Evaluator on four dimensions:

| Metric | Description | Weight |
|---|---|---|
| **Faithfulness** | Every claim directly supported by cited evidence | 35% |
| **Relevance** | Answer addresses the user's actual question | 25% |
| **Completeness** | All aspects of the question are covered | 25% |
| **Reasoning Quality** | Answer is structured, clear, and logically sound | 15% |

Scores are stored in `state["metrics"]["evaluation"]` and returned in every API response — on both success and HITL — so even low-confidence responses show *why* quality was insufficient.

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
print(result["trace"])                          # full audit trail
print(result["metrics"]["evaluation"])          # faithfulness, relevance, completeness, reasoning
print(result["metrics"]["confidence_history"])  # confidence per iteration
print(result["metrics"]["retry_reasons"])       # structured log of every retry decision
```

### API Route Integration

```python
from app.orchestrator.state import MASISInput
from app.orchestrator.graph import graph
from fastapi.concurrency import run_in_threadpool

@router.post("/masis/workspaces/{workspace_id}")
async def masis_query(workspace_id: str, body: QueryRequest):
    initial_state = MASISInput(
        user_query=body.query,
        workspace_id=workspace_id,
        max_retries=body.max_retries or 2
    ).to_state()

    result = await run_in_threadpool(graph.invoke, initial_state)

    if result.get("requires_human_review"):
        return {
            "status": "needs_clarification",
            "answer": result.get("final_answer"),         # best draft — always returned, never None
            "confidence": result.get("confidence"),
            "requires_human_review": True,
            "clarification_question": result.get("clarification_question"),
            "critique": result.get("critique"),
            "evaluation": result.get("metrics", {}).get("evaluation"),
            "trace": result.get("trace"),
            "metrics": result.get("metrics"),
        }

    return {
        "status": "success",
        "answer": result.get("final_answer"),
        "confidence": result.get("confidence"),
        "requires_human_review": False,
        "critique": result.get("critique"),
        "evaluation": result.get("metrics", {}).get("evaluation"),
        "trace": result.get("trace"),
        "metrics": result.get("metrics"),
    }
```

---

## Observability

Every state object carries a complete audit trail. Example from a retry cycle:

```json
{
  "trace": [
    {
      "node": "researcher",
      "chunks": 7, "filtered_out": 3, "avg_score": 0.724,
      "threshold_used": 0.60, "augmented_query_used": false, "duration_ms": 312
    },
    {
      "node": "synthesizer",
      "context_compressed": false, "citations": 6, "answer_length": 1102, "duration_ms": 1854
    },
    {
      "node": "critic",
      "confidence": 0.58, "hallucination": true,
      "invalid_citations": 2, "uncited_claims": 3, "duration_ms": 1103
    },
    {
      "node": "evaluator",
      "overall_score": 0.61, "faithfulness": 0.40,
      "relevance": 0.85, "completeness": 0.70, "duration_ms": 987
    },
    {
      "node": "supervisor",
      "decision": "retry", "confidence": 0.58,
      "retry_count": 1, "reason": "quality_issue_detected"
    },
    {
      "node": "researcher",
      "chunks": 11, "filtered_out": 9, "avg_score": 0.681,
      "threshold_used": 0.55, "augmented_query_used": true, "duration_ms": 445
    },
    {
      "node": "synthesizer",
      "context_compressed": true, "citations": 8, "answer_length": 1387, "duration_ms": 2103
    },
    {
      "node": "critic",
      "confidence": 0.884, "hallucination": false,
      "invalid_citations": 0, "uncited_claims": 1, "duration_ms": 989
    },
    {
      "node": "evaluator",
      "overall_score": 0.84, "faithfulness": 0.89,
      "relevance": 0.88, "completeness": 0.76, "duration_ms": 1021
    },
    {
      "node": "supervisor",
      "decision": "finalize", "confidence": 0.884, "retry_count": 1
    }
  ],
  "metrics": {
    "confidence_history": [0.58, 0.884],
    "retry_reasons": [
      {
        "iteration": 1, "confidence": 0.58,
        "reason": "quality_issue_detected",
        "citation_issue": true, "hallucination": true
      }
    ]
  }
}
```

---

## Termination Guarantee

The graph always terminates. No infinite loops are possible:

1. `retry_count` increments on every retry decision
2. `max_retries` is a hard ceiling, guarded against `None` (`state.get("max_retries") or 2`)
3. When `retry_count >= max_retries`, the Supervisor always escalates to HITL → `END`
4. `requires_human_review = True` short-circuits all routing to `END` immediately
5. The Researcher sets `requires_human_review` immediately when no chunks pass the score filter — bypassing all retry cycles without wasting LLM calls

---

## Documentation

Full deep-dive documentation for each component lives in `/docs`:

- [`01_researcher_agent.md`](docs/01_researcher_agent.md) — Score filter, adaptive threshold, two-variant HITL messages
- [`02_synthesizer_agent.md`](docs/02_synthesizer_agent.md) — Strategic analyst persona, compression, critique feedback injection
- [`03_critic_agent.md`](docs/03_critic_agent.md) — Proportional penalty, `last_citation_audit` passthrough to Evaluator
- [`04_supervisor_agent.md`](docs/04_supervisor_agent.md) — Threshold 0.65, retry reasons telemetry, decision tree
- [`05_evaluator_agent.md`](docs/05_evaluator_agent.md) — Citation audit injection, hard clamps, faithfulness-weighted scoring
- [`06_graph_flow.md`](docs/06_graph_flow.md) — Full graph diagram, all flow paths, configuration coupling table
- [`07_interview_questions.md`](docs/07_interview_questions.md) — 21 interview Q&As with model answers

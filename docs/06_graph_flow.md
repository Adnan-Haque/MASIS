# 🕸️ LangGraph — Graph Architecture & Flow Deep Dive

## Overview

MASIS uses **LangGraph** to implement a **Directed Acyclic Graph (DAG) with a conditional cycle** — iterative self-correction with guaranteed termination. The graph is not a simple linear chain, nor a fully cyclic loop. It is a controlled feedback structure where one conditional edge governs all retry and exit logic.

---

## Graph Structure

```
                    ┌─────────────┐
                    │             │
         ┌──────────►  SUPERVISOR  ◄──────────────┐
         │          │             │               │
         │          └──────┬──────┘               │
         │                 │                      │
         │    ┌────────────┼──────────────┐       │
         │    │            │              │       │
         │  first_run    retry           end      │
         │    │            │              │       │
         │    └─────┬──────┘              └──►  END
         │          │
         │    ┌─────▼──────┐
         │    │            │
         │    │ RESEARCHER  │  ← score filter: drops chunks < 0.60 similarity
         │    │            │
         │    └─────┬──────┘
         │          │ (fixed edge)
         │    ┌─────▼──────┐
         │    │            │
         │    │SYNTHESIZER  │  ← strategic analyst persona, partial-evidence hedging
         │    │            │
         │    └─────┬──────┘
         │          │ (fixed edge)
         │    ┌─────▼──────┐
         │    │            │
         │    │   CRITIC    │  ← proportional penalty + last_citation_audit
         │    │            │
         │    └─────┬──────┘
         │          │ (fixed edge)
         │    ┌─────▼──────┐
         │    │            │
         └────┤  EVALUATOR  │  ← consumes last_citation_audit, hard clamps faithfulness
              │            │
              └────────────┘
```

---

## How the Graph Is Built

### State Schema

```python
from app.orchestrator.state import MASISState  # TypedDict

builder = StateGraph(MASISState)
```

`MASISState` is a `TypedDict` — the correct LangGraph pattern. Dict-style access (`state["key"]`) in all node functions. No Pydantic attribute-access errors.

The API boundary uses `MASISInput` (Pydantic `BaseModel`) which converts to `MASISState` via `.to_state()`. Validation at the API layer; the graph only ever sees typed dicts.

### Fixed Edges

```python
builder.add_edge("researcher", "synthesizer")
builder.add_edge("synthesizer", "critic")
builder.add_edge("critic", "evaluator")
builder.add_edge("evaluator", "supervisor")
```

Unconditional. After Researcher always goes Synthesizer. After Critic always goes Evaluator. After Evaluator always returns to Supervisor. No branching in the inner pipeline — only the Supervisor branches.

### Conditional Edge (The Router)

```python
def route_from_supervisor(state: MASISState):
    if state.get("requires_human_review"):
        return "end"
    
    last_trace = state.get("trace", [])[-1] if state.get("trace") else {}
    
    if last_trace.get("decision") == "retry":
        return "retry"
    
    if state.get("draft_answer") is None:
        return "first_run"
    
    return "end"

builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "retry": "researcher",
        "first_run": "researcher",
        "end": END,
    },
)
```

This single conditional edge drives all control flow. `requires_human_review` is checked first — once set, it always routes to END regardless of other state. Both `"retry"` and `"first_run"` map to the same destination but are semantically distinct in the trace.

---

## Complete Flow Walkthrough

### Happy Path — First Run Succeeds

```
Step 1:  Query arrives → graph.invoke(initial_state)
Step 2:  Supervisor — draft_answer is None → return (no-op)
Step 3:  Router → "first_run" → Researcher
Step 4:  Researcher — embeds query, fetches 10 candidates from Qdrant
         → applies score filter (threshold 0.60) → 7 qualify
         → state["evidence"] = 7 chunks, avg_score 0.724
Step 5:  Synthesizer — context < 6000 chars, no compression needed
         → generates cited answer as "strategic intelligence analyst"
         → state["draft_answer"]
Step 6:  Critic — LLM semantic audit + citation engine
         → no invalid citations, 1 uncited sentence → 3% penalty
         → confidence = 0.88 * 0.97 = 0.854
         → state["critique"], state["metrics"]["last_citation_audit"]
Step 7:  Evaluator — reads last_citation_audit (no hallucinations found)
         → no clamps needed, scores: faithfulness 0.91, relevance 0.88, ...
         → state["metrics"]["evaluation"]
Step 8:  Supervisor — confidence 0.854 > 0.65, no flags → decision: "finalize"
Step 9:  Router → "end" → END
```

**LLM calls: 3** (Synthesizer, Critic, Evaluator)  
**Qdrant calls: 1** (10 candidates fetched, 7 pass filter)

---

### Retry Path — Critic Flags Quality Issues

```
Steps 1–7: Same as happy path
Step 8:  Supervisor — confidence 0.58 < 0.65, hallucination_detected = True
         → retry_count 0 < max_retries 2
         → retry_count = 1
         → logs retry_reason: {confidence: 0.58, citation_issue: false, hallucination: true}
         → decision: "retry"
Step 9:  Router → "retry" → Researcher
Step 10: Researcher — threshold drops to 0.55, limit expands to 20 candidates
         → augments query with unsupported_claims + logical_gaps
         → 14 candidates fetched, 11 pass filter (threshold 0.55)
         → avg_score: 0.681
Step 11: Synthesizer — receives new evidence + critique_feedback in prompt
         → "Correct these issues: Hallucination: True, Unsupported claims: [...]"
         → generates improved cited answer
Step 12: Critic — re-audits → confidence 0.83, no issues
         → last_citation_audit: no invalid IDs, 0 uncited claims
Step 13: Evaluator — no clamps needed → faithfulness 0.89, overall 0.84
Step 14: Supervisor — confidence 0.83 > 0.65, no flags → decision: "finalize"
Step 15: Router → "end" → END
```

**LLM calls: 6** (2× Synthesizer, 2× Critic, 2× Evaluator)  
**Qdrant calls: 2** (10 first pass, 20 on retry)

---

### HITL Path — Retries Exhausted

```
Steps 1–14: Two full retry cycles, quality never reaches threshold
Step 15:   Supervisor — retry_count = 2 = max_retries, quality_issue still True
           → requires_human_review = True
           → clarification_question = "After 2 refinement attempts, confidence remains 54.2%..."
Step 16:   Router — requires_human_review check fires first → "end"
Step 17:   API returns: status="needs_clarification", answer=<best draft>, evaluation=<scores>
```

The frontend shows the best draft with an amber warning, the clarification message, and the full quality assessment panel explaining *why* confidence was insufficient.

---

### Zero-Results / All-Filtered Path

```
Step 1–3: Same as happy path
Step 4:   Researcher — Qdrant returns 10 results, all score < 0.60
          → filtered_out = 10, evidence = []
          → requires_human_review = True immediately
          → clarification_question = "Your query did not match any documents with sufficient relevance..."
Step 5:   Synthesizer, Critic, Evaluator still run (fixed edges cannot be skipped)
          BUT: evidence is empty → Synthesizer hedges → Critic flags everything
Step 6:   Supervisor — requires_human_review already True
Step 7:   Router — requires_human_review → "end" immediately
```

> **Known design limitation:** Fixed edges mean Synthesizer, Critic, and Evaluator still run even after the Researcher sets HITL. The wasted calls are small (one LLM cycle) but present. A production improvement would add a conditional edge after the Researcher checking `requires_human_review` to short-circuit directly to the Supervisor.

---

## State as a Shared Whiteboard

```
state = {
    "user_query": "...",              # set at init, immutable
    "workspace_id": "...",             # set at init, immutable
    "evidence": [...],                 # Researcher → Synthesizer + Critic
    "draft_answer": "...",             # Synthesizer → Critic + Supervisor
    "final_answer": "...",             # Critic → Evaluator + API
    "critique": {...},                 # Critic → Supervisor + Researcher (retry)
    "confidence": 0.0,                 # Critic → Supervisor
    "retry_count": 0,                  # Supervisor → Researcher + Synthesizer
    "requires_human_review": False,    # Researcher or Supervisor → Router
    "clarification_question": None,    # Researcher or Supervisor → API
    "trace": [...],                    # every node appends
    "metrics": {
        "last_citation_audit": {...},  # Critic → Evaluator (producer-consumer)
        "retry_reasons": [...],        # Supervisor → monitoring
        "confidence_history": [...],   # Critic appends per iteration
        "evaluation": {...},           # Evaluator → API
        ...
    }
}
```

`last_citation_audit` is a key addition — a structured channel from the Critic's deterministic citation engine to the Evaluator's LLM scoring, ensuring the two layers of quality assessment are consistent.

---

## Key Configuration Values and Their Relationships

| Parameter | Value | Where Set | Coupled To |
|---|---|---|---|
| `MIN_SCORE_THRESHOLD` | 0.60 (0.55 on retry) | `researcher_node` | `LOW_CONF_THRESHOLD` |
| `LOW_CONF_THRESHOLD` | 0.65 | `supervisor_node` | `MIN_SCORE_THRESHOLD` |
| `MAX_CONTEXT_CHARS` | 6000 | `synthesizer_node` | chunk size at index time |
| `limit` | 10 / 20 | `researcher_node` | `MIN_SCORE_THRESHOLD` |
| `max_retries` | 2 (default) | state / API input | all HITL triggers |

Raising `MIN_SCORE_THRESHOLD` makes retrieval stricter and improves evidence quality → you can correspondingly raise `LOW_CONF_THRESHOLD`. Lowering `MIN_SCORE_THRESHOLD` allows weaker evidence through → you need a lower or unchanged `LOW_CONF_THRESHOLD` to avoid over-triggering HITL.

---

## Termination Guarantee

The graph always terminates:

1. `retry_count` increments on every retry decision.
2. `max_retries` is a hard ceiling (default 2), guarded against `None`.
3. When `retry_count >= max_retries`, the Supervisor always goes HITL → END.
4. The Router's first check is `requires_human_review` — once set, all paths lead to END.
5. The Researcher sets `requires_human_review` immediately on zero-qualifying evidence, bypassing all further retries.

There is no path through the graph that can loop indefinitely.

---

## Why LangGraph

| Framework | Why Not Chosen |
|---|---|
| **CrewAI** | Role-based agent teams — better for collaborative tasks, not tight feedback loops with conditional routing |
| **AutoGen** | Conversational multi-agent — designed for agent-to-agent dialogue, not structured DAG control flow |
| **LangChain Chains** | Linear only — cannot express cycles, retries, or conditional routing |
| **LangGraph** | Explicit graph definition, typed state, conditional edges, cycle support — exactly what MASIS needs |

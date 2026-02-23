# 🕸️ LangGraph — Graph Architecture & Flow Deep Dive

## Overview

MASIS uses **LangGraph** to implement a **Directed Acyclic Graph (DAG) with a conditional cycle** — a pattern that enables iterative self-correction while guaranteeing termination. The graph is not a simple linear chain, nor a fully cyclic loop. It is a controlled feedback structure where one conditional edge governs all retry and exit logic.

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
         │    │ RESEARCHER  │
         │    │            │
         │    └─────┬──────┘
         │          │ (fixed edge)
         │    ┌─────▼──────┐
         │    │            │
         │    │SYNTHESIZER  │
         │    │            │
         │    └─────┬──────┘
         │          │ (fixed edge)
         │    ┌─────▼──────┐
         │    │            │
         │    │   CRITIC    │
         │    │            │
         │    └─────┬──────┘
         │          │ (fixed edge)
         │    ┌─────▼──────┐
         │    │            │
         └────┤  EVALUATOR  │
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

`MASISState` is a `TypedDict` — the correct LangGraph pattern. This gives:
- Dict-style access (`state["key"]`) in all node functions
- Type hints for IDE support and documentation
- No Pydantic attribute-access errors at runtime

The API boundary uses `MASISInput` (a Pydantic `BaseModel`) which converts to `MASISState` via `.to_state()`. Validation happens at the API layer; the graph only ever sees typed dicts.

### Node Registration

```python
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("critic", critic_node)
builder.add_node("evaluator", evaluator_node)
```

Each node is a plain Python function: `(MASISState) -> MASISState`. Nodes read from state, mutate it, and return it. LangGraph merges the returned state back into the shared state object between node executions.

### Entry Point

```python
builder.set_entry_point("supervisor")
```

The graph always starts at the Supervisor, regardless of what state looks like. On the first call, the Supervisor detects `draft_answer is None` and passes through immediately.

### Fixed Edges (Deterministic Flow)

```python
builder.add_edge("researcher", "synthesizer")
builder.add_edge("synthesizer", "critic")
builder.add_edge("critic", "evaluator")
builder.add_edge("evaluator", "supervisor")
```

These four edges are unconditional. After the Researcher runs, it always goes to the Synthesizer. After the Critic, always to the Evaluator. After the Evaluator, always back to the Supervisor. There is no branching in the inner pipeline — only the Supervisor branches.

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

This single conditional edge drives all control flow. Three possible outcomes:
- `"end"` → graph terminates (HITL triggered, or quality is good)
- `"retry"` → loop back to Researcher with augmented state
- `"first_run"` → initial run, go to Researcher with clean state

Both `"retry"` and `"first_run"` map to the same destination (`researcher`), but are semantically distinct in the trace — you can see in `state["trace"]` whether a Researcher call was a first run or a retry, which matters for debugging and monitoring.

---

## Complete Flow Walkthrough

### Happy Path (First Run Succeeds)

```
Step 1:  User query arrives → graph.invoke(initial_state)
Step 2:  Supervisor — draft_answer is None → return state (no-op)
Step 3:  Router → "first_run" → Researcher
Step 4:  Researcher — embeds query, fetches 5 chunks from Qdrant, deduplicates → state["evidence"]
Step 5:  Synthesizer — context < 6000 chars, no compression → generates cited answer → state["draft_answer"]
Step 6:  Critic — LLM audits answer, citation engine validates IDs → confidence = 0.88, no issues → state["critique"]
Step 7:  Evaluator — scores faithfulness/relevance/completeness/reasoning → state["metrics"]["evaluation"]
Step 8:  Supervisor — confidence 0.88 > 0.75, no flags → decision: "finalize"
Step 9:  Router → "end" → END
Step 10: graph.invoke() returns final state
```

**Total LLM calls: 3** (Synthesizer, Critic, Evaluator)
**Total Qdrant calls: 1**

---

### Retry Path (Critic Flags Quality Issues)

```
Step 1–7: Same as happy path
Step 8:   Supervisor — confidence = 0.58 < 0.75, hallucination_detected = True
          → retry_count 0 < max_retries 2
          → retry_count = 1, decision: "retry"
Step 9:   Router → "retry" → Researcher
Step 10:  Researcher — augments query with unsupported_claims + logical_gaps
          → fetches 10 chunks (expanded limit) → state["evidence"] updated
Step 11:  Synthesizer — receives new evidence + critique_feedback in prompt
          → generates improved cited answer → state["draft_answer"] updated
Step 12:  Critic — re-audits → confidence = 0.83, no issues → state["critique"] updated
Step 13:  Evaluator — re-scores with improved answer
Step 14:  Supervisor — confidence 0.83 > 0.75, no flags → decision: "finalize"
Step 15:  Router → "end" → END
```

**Total LLM calls: 6** (2× Synthesizer, 2× Critic, 2× Evaluator)
**Total Qdrant calls: 2**

---

### HITL Path (Retries Exhausted)

```
Steps 1–14: Two full retry cycles, quality never reaches threshold
Step 15:   Supervisor — retry_count = 2 = max_retries, quality_issue still True
           → requires_human_review = True
           → clarification_question = "After 2 refinement attempts, confidence remains 54.2%..."
Step 16:   Router → "end" (HITL check fires before retry/finalize checks)
Step 17:   graph.invoke() returns state with requires_human_review=True
           → API surfaces clarification_question to user
```

---

### Zero-Results Path (Researcher Finds Nothing)

```
Step 1–3: Same as happy path
Step 4:   Researcher — Qdrant returns 0 results
          → requires_human_review = True immediately
          → clarification_question = "No relevant documents found..."
          → returns state with evidence = []
Step 5:   Synthesizer, Critic, Evaluator still run (fixed edges cannot be skipped)
          BUT: evidence is empty, draft_answer will be flagged
Step 6–7: Critic sees empty evidence → everything uncited → high penalty
Step 8:   Supervisor — requires_human_review already True (set by Researcher)
          → Router reads requires_human_review → "end"
Step 9:   END — clarification_question from Researcher is returned
```

> **Note:** This is a known design limitation — the fixed edges mean Synthesizer, Critic, and Evaluator still run even after the Researcher sets HITL. A production improvement would be to add a conditional edge after the Researcher that checks `requires_human_review` and short-circuits to the Supervisor. Worth mentioning in interview.

---

## State as a Shared Whiteboard

LangGraph passes a **single state object** through every node. Every node reads from and writes to the same dict. This is the "shared memory / whiteboard" architecture described in the case study's LLD section.

```
state = {
    "user_query": "...",          # set at init, never changed
    "workspace_id": "...",         # set at init, never changed
    "evidence": [...],             # written by Researcher, read by Synthesizer + Critic
    "draft_answer": "...",         # written by Synthesizer, read by Critic + Supervisor
    "final_answer": "...",         # written by Critic, read by Evaluator + returned to API
    "critique": {...},             # written by Critic, read by Supervisor + Researcher (on retry)
    "confidence": 0.0,             # written by Critic, read by Supervisor
    "retry_count": 0,              # written by Supervisor, read by Researcher + Synthesizer
    "requires_human_review": False,# written by Researcher or Supervisor, read by router
    "clarification_question": None,# written by Researcher or Supervisor, returned to API
    "trace": [...],                # appended by every node — full audit trail
    "metrics": {...}               # appended by every node — telemetry
}
```

Context growth is controlled by the Synthesizer's compression logic — the state itself doesn't compress, but the evidence fed into prompts does.

---

## Why LangGraph Over Other Frameworks

| Framework | Why Not Chosen |
|---|---|
| **CrewAI** | Role-based agent teams — better for collaborative tasks, not tight feedback loops with conditional routing |
| **AutoGen** | Conversational multi-agent — designed for agent-to-agent dialogue, not structured DAG control flow |
| **LangChain Chains** | Linear only — cannot express cycles, retries, or conditional routing |
| **LangGraph** | Explicit graph definition, typed state, conditional edges, cycle support — exactly what MASIS needs |

LangGraph's `StateGraph` with `TypedDict` state and conditional edges is the minimal, correct tool for this architecture. The explicit graph definition also makes the system's behaviour fully inspectable — you can visualise the graph, step through it, and mock individual nodes for testing.

---

## Termination Guarantee

The graph always terminates. This is guaranteed by:

1. `retry_count` increments on every retry decision.
2. `max_retries` is a hard ceiling (default: 2, guarded against `None`).
3. When `retry_count >= max_retries`, the Supervisor always goes to HITL → END, never retry.
4. The Router's first check is `requires_human_review` — once set, all paths lead to END regardless of other state.

There is no path through the graph that can loop indefinitely.

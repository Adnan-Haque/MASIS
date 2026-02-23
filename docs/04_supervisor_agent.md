# 🧠 Agent 4: Supervisor Node — Deep Dive

## Role & Persona

The Supervisor is the **"Brain"** of MASIS — the orchestration intelligence that decides what happens next at every cycle. It is the only node with the authority to trigger retries, escalate to a human, or finalize the answer. All other agents execute their tasks and return state; the Supervisor reads that state and makes routing decisions.

It is the first node the graph visits (entry point) and the last node before a routing decision is made. The graph always returns to the Supervisor after the Evaluator finishes.

---

## What Problem Does It Solve?

### Problem 1: No Intelligence in the Routing Layer

In a naive pipeline, routing would be a simple rule: "if confidence < threshold, retry." This ignores the variety of failure modes in a real system — invalid citations, conflicting evidence, over-compression, zero retrieval — each of which requires a different response.

**How MASIS solves it — Multi-Signal Quality Assessment:**

```python
quality_issue = (
    confidence < LOW_CONF_THRESHOLD     # semantic confidence too low
    or citation_issue                    # fabricated chunk IDs detected
    or hallucination_flag                # LLM flagged hallucination
    or critic_retry_flag                 # critic explicitly recommends retry
)
```

The Supervisor aggregates four independent signals:

1. **Confidence below threshold** (`0.75`): the penalized score from the Critic is below acceptable quality.
2. **Citation issue**: the citation engine found references to non-existent chunk IDs.
3. **Hallucination flag**: the LLM auditor identified semantically unsupported claims.
4. **Critic retry flag**: the Critic's structured output explicitly recommends a retry.

Any single signal is sufficient to trigger a quality issue. This ensures the system catches both subtle semantic failures (low confidence, hallucination) and hard structural failures (invalid citations).

---

### Problem 2: Immediate HITL for Conflicts — Wasted Retries

The original design escalated to a human immediately when conflicting evidence was found, before any retry was attempted. But conflicting evidence in a first-pass retrieval with 5 chunks might disappear with 10 chunks — the conflict could be a retrieval artifact, not a genuine document contradiction.

**How MASIS solves it — Retry-First Conflict Resolution:**

```python
if (quality_issue or has_conflicts) and retry_count < max_retries:
    state["retry_count"] = retry_count + 1
    reason = "quality_issue_detected"
    if has_conflicts and not quality_issue:
        reason = "conflicting_evidence_attempting_resolution"
    ...
    return state

# Only escalate conflict to human AFTER retries are exhausted
if has_conflicts and retry_count >= max_retries:
    state["requires_human_review"] = True
    state["clarification_question"] = (
        "Conflicting information was detected across documents and could not be "
        "automatically resolved after multiple attempts. ..."
    )
```

Conflicts now trigger a retry first (with an augmented query), giving the Researcher a chance to find disambiguating evidence. Only after all retries are exhausted without resolution is the conflict escalated to the human. This directly addresses the case study's requirement: *"How does the system resolve conflicting evidence between multiple documents?"* — the answer is: it tries to resolve it autonomously before asking for help.

---

### Problem 3: The First-Run Bypass Problem

The Supervisor is both the entry point and the post-evaluation router. On the very first call, no draft answer exists yet. If the Supervisor ran its full decision logic on an empty state, it would see `confidence = 0.0`, classify it as a quality issue, increment `retry_count`, and enter a retry loop before even generating a first answer.

**How MASIS solves it — First-Run Guard:**

```python
def supervisor_node(state: MASISState) -> MASISState:
    _init_metrics(state)
    
    # First call — no answer yet, pass through to start the pipeline
    if state.get("draft_answer") is None:
        return state
```

The very first check is whether `draft_answer` is `None`. If it is, the Supervisor returns immediately without any routing logic. The graph's `route_from_supervisor` function then sees `draft_answer is None` and routes to `first_run → researcher`, starting the pipeline. The Supervisor's decision logic only activates after at least one full pipeline cycle has completed.

---

### Problem 4: Ambiguous HITL Messaging

When a human is asked to intervene, they need to understand exactly why — and what to do. A generic "the system needs help" message is useless. Different failure modes need different instructions.

**How MASIS solves it — Contextual HITL Messages:**

Three distinct HITL scenarios each produce a tailored message:

**Conflict HITL** (documents disagree):
```
"Conflicting information was detected across documents and could not be
automatically resolved after multiple attempts. Please review the competing
claims and select a preferred source."
```

**Quality HITL** (retries exhausted, still low confidence):
```
f"After {max_retries} refinement attempts, confidence remains {confidence}%.
You may refine your query or upload additional evidence."
```

**Zero-results HITL** (set by Researcher, surfaced by routing):
```
"No relevant documents were found for your query in this workspace.
Please upload relevant documents or refine your question."
```

Each message tells the user what happened, what was attempted, and what action to take.

---

### Problem 5: `max_retries` Could Be `None`

If a caller passes `max_retries=None` explicitly in the state, a comparison like `retry_count < max_retries` raises a `TypeError`. Since `max_retries` comes from user-controlled input, this is a real attack surface.

**How MASIS solves it — Defensive Fallback:**

```python
max_retries = state.get("max_retries") or 2
```

Using `or 2` instead of `state.get("max_retries", 2)` handles both the missing-key case and the explicit-`None` case. If `max_retries` is `None`, `0`, or absent, the fallback of `2` is used.

---

## Decision Tree

```
Supervisor Called
│
├── draft_answer is None?
│   └── YES → return (route: first_run → researcher)
│
├── quality_issue OR has_conflicts?
│   ├── AND retry_count < max_retries
│   │   └── increment retry_count → return (route: retry → researcher)
│   │
│   └── AND retry_count >= max_retries
│       ├── has_conflicts → HITL (conflict message) → return (route: end)
│       └── quality_issue → HITL (quality message) → return (route: end)
│
└── No issues → finalize → return (route: end)
```

---

## State Inputs & Outputs

| Field | Direction | Description |
|---|---|---|
| `draft_answer` | Input | Used to detect first-run vs. post-cycle |
| `critique` | Input | Confidence, hallucination flag, conflicts, retry flag |
| `retry_count` | Input | Current iteration count |
| `max_retries` | Input | Ceiling for retry attempts |
| `metrics.citation_violations` | Input | Latest citation engine findings |
| `retry_count` | **Output** | Incremented if retry decision made |
| `requires_human_review` | **Output** | Set to `True` on HITL decision |
| `clarification_question` | **Output** | Human-readable HITL message |
| `trace` | **Output** | Appended decision entry |

---

## Telemetry Emitted

**On retry:**
```json
{
  "node": "supervisor",
  "decision": "retry",
  "confidence": 0.612,
  "retry_count": 1,
  "reason": "quality_issue_detected"
}
```

**On finalize:**
```json
{
  "node": "supervisor",
  "decision": "finalize",
  "confidence": 0.881,
  "retry_count": 1
}
```

**On HITL:**
```json
{
  "node": "supervisor",
  "decision": "HITL_triggered",
  "confidence": 0.54,
  "retry_count": 2
}
```

---

## Why the Supervisor Is Not an LLM

This is a deliberate design choice worth being explicit about. The Supervisor uses **pure Python logic** — no LLM call, no prompt, no structured output. This gives it three critical properties:

1. **Determinism**: The same state always produces the same routing decision. No temperature, no randomness, no mood.
2. **Speed**: No API call, no latency. The routing decision is microseconds, not seconds.
3. **Debuggability**: Every decision is traceable to a specific condition in code. When auditing why the system retried or escalated, you look at the state values, not at an LLM's opaque reasoning.

The case study asks about "Systemic Thinking" — this is the architectural answer: routing logic should be code, not language model inference.

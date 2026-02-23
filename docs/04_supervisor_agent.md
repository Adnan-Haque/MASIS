# 🧠 Agent 4: Supervisor Node — Deep Dive

## Role & Persona

The Supervisor is the **"Brain"** of MASIS — the orchestration intelligence that decides what happens next at every cycle. It is the only node with authority to trigger retries, escalate to a human, or finalize. All other agents execute tasks and return state; the Supervisor reads that state and makes routing decisions.

It is both the entry point and the post-evaluation router. The graph always returns to the Supervisor after the Evaluator finishes.

---

## What Problem Does It Solve?

### Problem 1: Simplistic Single-Signal Routing

A naive pipeline routes on a single rule: "if confidence < threshold, retry." This ignores the variety of failure modes — invalid citations, conflicting evidence, over-compression, score-filtered retrieval — each requiring a different response.

**How MASIS solves it — Multi-Signal Quality Assessment:**

```python
quality_issue = (
    confidence < LOW_CONF_THRESHOLD     # semantic confidence too low
    or citation_issue                    # fabricated chunk IDs detected
    or hallucination_flag                # LLM flagged hallucination
    or critic_retry_flag                 # critic explicitly recommends retry
)
```

Four independent signals — any single one is sufficient to trigger a quality issue. This catches both subtle semantic failures (low confidence, hallucination) and hard structural failures (invalid citations, over-compression).

---

### Problem 2: Confidence Threshold Too Strict for Broad Queries

The original threshold of `0.75` was calibrated for a world without score filtering in the Researcher. With the score filter now preventing weak evidence from entering the pipeline, answers produced on qualifying evidence are better grounded. Setting 0.75 as the bar still flagged well-grounded broad-query answers.

**How MASIS solves it — Lowered Threshold to 0.65:**

```python
LOW_CONF_THRESHOLD = 0.65
```

0.65 is appropriate when upstream retrieval quality is guaranteed by the score filter. The system is now less likely to retry a good answer simply because it covered a broad topic. If you increase `MIN_SCORE_THRESHOLD` in the Researcher, you can correspondingly raise `LOW_CONF_THRESHOLD` here — these two values are coupled.

---

### Problem 3: Immediate HITL for Conflicts — Wasted Retries

The original design escalated to a human immediately when conflicting evidence was found. But conflicting evidence in a first-pass retrieval with 10 chunks might disappear with 20 chunks — the conflict could be a retrieval artefact, not a genuine document contradiction.

**How MASIS solves it — Retry-First Conflict Resolution:**

```python
if (quality_issue or has_conflicts) and retry_count < max_retries:
    state["retry_count"] = retry_count + 1
    reason = "conflicting_evidence_attempting_resolution" if has_conflicts and not quality_issue else "quality_issue_detected"
    ...
    return state

# Only escalate to human AFTER retries are exhausted
if has_conflicts and retry_count >= max_retries:
    state["requires_human_review"] = True
    state["clarification_question"] = (
        "Conflicting information was detected across documents and could not be "
        "automatically resolved after multiple attempts. ..."
    )
```

Conflicts now trigger a retry first (augmented query, broader retrieval). Only after all retries are exhausted without resolution is the conflict escalated to the human. The system tries to resolve ambiguity autonomously before asking for help.

---

### Problem 4: The First-Run Bypass Problem

The Supervisor is both the entry point and the post-evaluation router. On the very first call, no draft answer exists. If the Supervisor ran full decision logic on an empty state, it would see `confidence = 0.0`, classify it as a quality issue, increment `retry_count`, and enter a retry loop before generating a first answer.

**How MASIS solves it — First-Run Guard:**

```python
if state.get("draft_answer") is None:
    return state
```

The first check is whether `draft_answer` is `None`. If it is, the Supervisor returns immediately without any routing logic. The router then detects `draft_answer is None` and routes to `first_run → researcher`.

---

### Problem 5: Ambiguous HITL Messaging

A generic "the system needs help" message gives the user no actionable guidance. Different failure modes need different instructions.

**How MASIS solves it — Contextual HITL Messages:**

**Conflict HITL** (documents disagree, retries exhausted):
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

**Zero-results HITL** (set by Researcher, different message for filtered vs. empty):
```
"Your query did not match any documents with sufficient relevance. Try
rephrasing with more specific terms from your documents, or upload documents
that cover this topic."
```

Each message identifies what happened and what to do.

---

### Problem 6: Retry Reasoning Is Opaque

Previously the retry decision was logged but without structured reasoning for why it was made, making monitoring and debugging harder.

**How MASIS solves it — Structured Retry Reasons Telemetry:**

```python
state["metrics"]["retry_reasons"].append({
    "iteration": retry_count + 1,
    "confidence": confidence,
    "reason": reason,
    "citation_issue": citation_issue,
    "hallucination": hallucination_flag,
})
```

Every retry decision is logged with the specific signals that triggered it, the confidence at that point, and the iteration number. In production this enables queries like "show me all requests where the retry reason was citation_issue" — enabling targeted improvements.

---

### Problem 7: `max_retries` Could Be `None`

```python
max_retries = state.get("max_retries") or 2
```

Using `or 2` handles both missing key and explicit `None`. A comparison `retry_count < None` would raise `TypeError`.

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
│   │   └── increment retry_count, log retry_reason → return (route: retry → researcher)
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
| `draft_answer` | Input | Detects first-run vs. post-cycle |
| `critique` | Input | Confidence, hallucination, conflicts, retry flag |
| `retry_count` | Input | Current iteration |
| `max_retries` | Input | Ceiling for retry attempts |
| `metrics.citation_violations` | Input | Latest citation engine findings |
| `retry_count` | **Output** | Incremented on retry |
| `requires_human_review` | **Output** | Set `True` on HITL |
| `clarification_question` | **Output** | Contextual HITL message |
| `metrics.retry_reasons` | **Output** | Structured log of every retry decision |
| `trace` | **Output** | Decision entry |

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

Pure Python logic — no LLM call. This gives:

1. **Determinism** — the same state always produces the same routing decision.
2. **Speed** — microseconds, not seconds.
3. **Debuggability** — every decision traces to a specific condition. No LLM opacity.

Routing logic is code, not language model inference.

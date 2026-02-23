# 🔎 Agent 3: Critic Node — Deep Dive

## Role & Persona

The Critic is the **"Auditor"** of MASIS. It receives the Synthesizer's draft answer and subjects it to two layers of scrutiny: an **LLM-based semantic audit** (catches hallucinations, logical gaps, and conflicting evidence requiring reading comprehension) and a **deterministic citation engine** (catches invalid or missing references verifiable purely through code). No answer leaves MASIS without passing through the Critic.

---

## What Problem Does It Solve?

### Problem 1: LLMs Cannot Self-Audit Reliably

Asking the same model that generated an answer to also check it for hallucinations is unreliable. The model has the same biases and knowledge gaps that produced the hallucination — it validates its own mistakes.

**How MASIS solves it — Separate Critic Agent with a Stronger Model:**

The Critic is a **distinct LLM call** using **`gpt-4o`** (while the Synthesizer uses `gpt-4o-mini`). The auditor is more powerful than the writer. It receives the draft answer and full evidence context but is given a completely different identity:

```
You are an AI Auditor.
Evaluate the answer strictly using the provided evidence.
```

It is never told what the "correct" answer should be — only what the evidence says and what the draft claims.

---

### Problem 2: Routing Requires Structured, Programmatic Signals

If the Critic returned free-form text ("I think claim 3 might be unsupported..."), the Supervisor couldn't reliably extract actionable signals.

**How MASIS solves it — Pydantic Structured Output:**

```python
class Critique(BaseModel):
    confidence: float
    hallucination_detected: bool
    unsupported_claims: list[str]
    logical_gaps: list[str]
    conflicting_evidence: list[str]
    needs_retry: bool
```

`.with_structured_output(Critique)` forces the response through OpenAI's function-calling mechanism — every field is typed and required. The Supervisor reads `critique["hallucination_detected"]` and `critique["needs_retry"]` as binary signals with no natural-language parsing.

---

### Problem 3: Hallucinated Citations — References to Non-Existent Chunks

An LLM under citation pressure fabricates chunk IDs. `"Revenue declined [chunk_99]"` when no `chunk_99` was ever retrieved. This looks cited but points to nothing — a structural hallucination invisible to the LLM's own audit.

**How MASIS solves it — The Hard Citation Engine:**

```python
citations = re.findall(r"\[(.*?)\]", answer)
valid_ids = {e.chunk_id for e in evidence}
invalid_citations = [c for c in citations if c not in valid_ids]

if invalid_citations:
    critique["hallucination_detected"] = True
    critique["needs_retry"] = True
    penalty_factor *= 0.5   # hard 50% confidence penalty
```

Every `[...]` pattern is extracted with regex. Any citation referencing an ID not in `valid_ids` is a confirmed fabrication. Hallucination is hard-set to `True` (overriding the LLM's own assessment), retry is forced, and a 50% confidence penalty is applied. Non-negotiable.

---

### Problem 4: Claims Made Without Any Citation

Beyond fake citations, there's a softer problem: factual sentences with no citation at all. The Synthesizer may draw on prior knowledge outside the evidence.

**How MASIS solves it — Uncited Claims Detection with Proportional Penalty:**

```python
sentences = re.split(r"[.!?]", answer)
uncited_claims = [
    s.strip()
    for s in sentences
    if s.strip()
    and "[" not in s
    and "insufficient evidence" not in s.lower()
    and "not provided" not in s.lower()
    and "cannot provide" not in s.lower()
    and "lack sufficient evidence" not in s.lower()
    and "partially covers" not in s.lower()
]
```

Every sentence is checked. Explicit hedge phrases — including `"lack sufficient evidence"` and `"partially covers"` which the Synthesizer's prompt now encourages — are excluded from the penalty. These are valid statements of evidence boundaries, not uncited claims.

Penalty logic:

```python
if uncited_claims:
    uncited_penalty = min(0.40, len(uncited_claims) * 0.03)  # 3% per claim, capped at 40%
    penalty_factor *= (1.0 - uncited_penalty)

    if len(uncited_claims) >= 5:
        critique["needs_retry"] = True  # force retry if heavily uncited
```

The penalty is **proportional** — a single uncited sentence loses 3% confidence, five uncited sentences lose 15%, ten uncited sentences cap at 40%. This is more calibrated than a flat penalty, which would treat one introductory sentence the same as ten unsupported claims.

---

### Problem 5: Confidence Scores on Wrong Scale

Some LLMs return `confidence: 85` instead of `confidence: 0.85`.

**How MASIS solves it — Confidence Normalization:**

```python
confidence = critique.get("confidence", 0.0)
if confidence > 1:
    confidence = confidence / 100.0
confidence = max(0.0, min(confidence, 1.0))
```

Any value above 1 is divided by 100, then clamped to `[0.0, 1.0]`.

---

### Problem 6: LLM Confidence Doesn't Reflect Citation Failures

The LLM's `confidence` reflects its semantic judgment — computed before the citation engine runs. The LLM might report high confidence even when the code finds invalid citations.

**How MASIS solves it — Post-Hoc Penalty Fusing Both Signals:**

```python
penalty_factor = 1.0
if invalid_citations:
    penalty_factor *= 0.5    # 50% for fabricated chunk IDs
if uncited_claims:
    uncited_penalty = min(0.40, len(uncited_claims) * 0.03)
    penalty_factor *= (1.0 - uncited_penalty)

confidence = max(0.0, min(confidence * penalty_factor, 1.0))
critique["confidence"] = confidence
```

Final confidence = LLM semantic score × penalty_factor. This fuses LLM judgment with deterministic citation findings into one score the Supervisor uses for routing.

---

## State Inputs & Outputs

| Field | Direction | Description |
|---|---|---|
| `draft_answer` | Input | Synthesizer's generated answer |
| `evidence` | Input | Evidence chunks (for citation validation) |
| `retry_count` | Input | Telemetry only |
| `critique` | **Output** | Full structured audit result |
| `confidence` | **Output** | Final penalized confidence score |
| `final_answer` | **Output** | Mirror of `draft_answer` (current best draft) |
| `metrics` | **Output** | `citation_violations`, `confidence_history`, `last_citation_audit`, `node_latency_ms` |
| `trace` | **Output** | Audit summary entry |

---

## Telemetry Emitted

```json
{
  "node": "critic",
  "confidence": 0.612,
  "hallucination": true,
  "needs_retry": true,
  "conflicts": 0,
  "invalid_citations": 2,
  "uncited_claims": 3,
  "duration_ms": 1103
}
```

---

## Why Two Layers of Auditing?

The LLM layer catches **semantic hallucinations** — plausible but ungrounded claims, logical leaps, contradictions requiring reading comprehension. The citation engine catches **structural hallucinations** — fake IDs, missing brackets — trivially detectable with code but that the LLM may not flag. Neither layer alone is sufficient. Together they cover both reasoning-level and format-level quality failures.

The Critic also passes `last_citation_audit` into `state["metrics"]` — a structured summary of all citation findings — which the Evaluator then uses to calibrate its faithfulness scores. This creates a direct signal path from deterministic code findings to the LLM-as-judge evaluation.

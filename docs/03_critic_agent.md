# 🔎 Agent 3: Critic Node — Deep Dive

## Role & Persona

The Critic is the **"Auditor"** of MASIS. It receives the Synthesizer's draft answer and subjects it to two layers of scrutiny: an **LLM-based semantic audit** (to catch hallucinations, logical gaps, and conflicting evidence that require reasoning to detect) and a **deterministic citation engine** (to catch invalid or missing references that can be verified purely through code). Together, these two layers form the most important quality gate in the system.

No answer leaves MASIS without passing through the Critic.

---

## What Problem Does It Solve?

### Problem 1: LLMs Cannot Self-Audit Reliably

Asking the same model that generated an answer to also check it for hallucinations is deeply unreliable. The model has the same biases and knowledge gaps that produced the hallucination in the first place — it will often validate its own mistakes.

**How MASIS solves it — Separate Critic Agent with a Stronger Model:**

The Critic is a **distinct LLM call** using **`gpt-4o`** (while the Synthesizer uses `gpt-4o-mini`). This is a deliberate asymmetry — the auditor is more powerful than the writer, making it less likely to be fooled by a plausible-sounding but unsupported claim.

The Critic receives the draft answer and the full evidence context, but is given a completely different system identity and task:

```
You are an AI Auditor.
Evaluate the answer strictly using the provided evidence.
```

It is never told what the "correct" answer should be — only what the evidence says and what the draft claims.

---

### Problem 2: Structured Auditing Without Free-Form Text

If the Critic returned a free-form critique ("I think claim 3 might be unsupported..."), the Supervisor would have no reliable way to extract actionable signals. It couldn't programmatically detect whether a hallucination was found, or whether a retry was needed.

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

LangChain's `.with_structured_output(Critique)` forces the LLM's response through OpenAI's function-calling mechanism, which guarantees the output conforms to this schema. Every field is typed and required. The Supervisor can then read `critique["hallucination_detected"]` and `critique["needs_retry"]` as binary signals, without any natural-language parsing.

---

### Problem 3: Hallucinated Citations — References to Chunks That Don't Exist

An LLM under citation pressure will sometimes fabricate chunk IDs. For example, the Synthesizer might write `"Revenue declined [chunk_99]"` when no chunk with ID `chunk_99` was ever retrieved. This is a particularly dangerous hallucination because it *looks* cited — it has a bracket reference — but it points to nothing.

**How MASIS solves it — The Hard Citation Engine (Deterministic Check):**

```python
citations = re.findall(r"\[(.*?)\]", answer)
valid_ids = {e.chunk_id for e in evidence}
invalid_citations = [c for c in citations if c not in valid_ids]
```

All `[...]` patterns in the answer are extracted with a regex. The set of valid chunk IDs (from the actual retrieved evidence) is computed. Any citation referencing an ID not in that set is flagged as **invalid** — a confirmed hallucination.

When invalid citations are found:

```python
if invalid_citations:
    critique["hallucination_detected"] = True
    critique["needs_retry"] = True
    penalty_factor *= 0.5
```

Hallucination is hard-set to `True` (overriding the LLM's own assessment), retry is required, and a 50% confidence penalty is applied. This is non-negotiable — fabricated citations are treated as critical failures.

---

### Problem 4: Claims Made Without Any Citation

Beyond fake citations, there's a softer problem: claims made with no citation at all. A sentence like "The company's strategy is primarily defensive" might be the Synthesizer's inference or prior knowledge — not derived from the evidence.

**How MASIS solves it — Uncited Claims Detection:**

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
]
```

Every sentence in the answer is checked. If a sentence contains no `[` character (no citation) and doesn't use explicit hedging language like "insufficient evidence", it's classified as an uncited claim. These aren't as severe as fake citations, but they indicate the Synthesizer is drawing on knowledge outside the evidence.

The penalty is lighter:

```python
if uncited_claims and citations:
    penalty_factor *= 0.9  # 10% confidence penalty
```

Note the condition: `if uncited_claims and citations`. If there are *no* citations at all in the answer, this check is skipped — the entire answer is likely already flagged by the LLM-based audit as having `hallucination_detected = True`.

---

### Problem 5: Confidence Scores on Different Scales

The LLM might return `confidence: 0.85` or it might return `confidence: 85` — both are valid interpretations of "85% confident" but only one is correct in a 0–1 float scale.

**How MASIS solves it — Confidence Normalization:**

```python
confidence = critique.get("confidence", 0.0)
if confidence > 1:
    confidence = confidence / 100.0
confidence = max(0.0, min(confidence, 1.0))
```

Any value above 1 is divided by 100, then the result is clamped to `[0.0, 1.0]`. This handles both 0–1 and 0–100 LLM output conventions without failing.

---

### Problem 6: Confidence Doesn't Reflect Citation Quality

The LLM's `confidence` field reflects its own reasoning about the answer's quality. But it cannot see the citation engine's deterministic findings — those are computed after the LLM response is received. So the LLM might report high confidence even if the code finds invalid citations.

**How MASIS solves it — Post-Hoc Penalty to Combine Both Signals:**

```python
penalty_factor = 1.0
if invalid_citations:
    penalty_factor *= 0.5    # severe penalty for fake citations
if uncited_claims and citations:
    penalty_factor *= 0.9    # mild penalty for uncited sentences

confidence = max(0.0, min(confidence * penalty_factor, 1.0))
critique["confidence"] = confidence
```

The final confidence is the LLM's raw score multiplied by the penalty factor. This fuses the LLM's semantic judgment with the code's deterministic citation findings into a single score the Supervisor uses for routing.

---

### Problem 7: Setting `final_answer` Prematurely

The Critic also sets `state["final_answer"] = answer`. This might seem odd — why does the Auditor set the final answer? The reason is that `final_answer` at this point means "the best draft we have so far, fully audited." If the Supervisor decides to finalize, this value is returned. If it decides to retry, a new draft will overwrite it on the next Synthesizer pass.

---

## State Inputs & Outputs

| Field | Direction | Description |
|---|---|---|
| `draft_answer` | Input | The Synthesizer's generated answer |
| `evidence` | Input | The evidence chunks (used to validate citations) |
| `retry_count` | Input | Used for telemetry only |
| `critique` | **Output** | Full structured audit result |
| `confidence` | **Output** | Final penalized confidence score |
| `final_answer` | **Output** | Mirror of `draft_answer` (current best draft) |
| `metrics` | **Output** | `citation_violations`, `confidence_history`, `node_latency_ms` |
| `trace` | **Output** | Appended audit summary |

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

The LLM layer catches **semantic hallucinations** — claims that are plausible but ungrounded, logical leaps, contradictions that require reading comprehension to spot. The citation engine catches **structural hallucinations** — fake IDs, missing brackets — that are trivially detectable with code but that the LLM might not flag. Neither layer alone is sufficient. Together they cover both reasoning-level and format-level quality failures.

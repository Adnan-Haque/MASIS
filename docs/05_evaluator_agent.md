# 📊 Agent 5: Evaluator Node — Deep Dive

## Role & Persona

The Evaluator is the **"Quality Scorecard"** of MASIS. It is an implementation of the **LLM-as-a-Judge** pattern — an independent LLM that scores the final answer across four dimensions: Faithfulness, Relevance, Completeness, and Reasoning Quality.

Unlike the Critic (which decides whether to retry), the Evaluator produces a **permanent quality record** that travels with the response for observability, monitoring, and continuous improvement. It does not affect routing.

Crucially, the Evaluator receives the Critic's deterministic citation findings as hard constraints on its scoring — it cannot assign a high faithfulness score to an answer the citation engine already proved is hallucinated.

---

## What Problem Does It Solve?

### Problem 1: No Objective Quality Signal After the Conversation

Without automated quality measurement, the only feedback is user complaints — rare, slow, and biased. You need a quality signal on every request.

**How MASIS solves it — LLM-as-a-Judge with Structured Scores:**

```python
class Evaluation(BaseModel):
    faithfulness: float        # groundedness (0–1)
    relevance: float           # question alignment (0–1)
    completeness: float        # coverage (0–1)
    reasoning_quality: float   # clarity & logic (0–1)
    overall_score: float       # weighted mean
    improvement_suggestions: list[str]
```

Stored in `state["metrics"]["evaluation"]` and returned in every API response — both on success and on HITL. This means even low-confidence responses come with a quality breakdown explaining *why* confidence was insufficient.

---

### Problem 2: Evaluator Cannot Score Relevance Without the Question

An early bug: the evaluation prompt included the answer and evidence but **not the user's question**. The evaluator was asked "Does this fully answer the user question?" without knowing what the question was.

**How MASIS solves it — User Query Explicitly in the Evaluation Prompt:**

```python
prompt = f"""
...
Relevance:
1 = fully answers user question

Question:
{query}

Answer:
{answer}

Evidence:
{context}
"""
```

The question appears first in the content — before the answer — mirroring how a human evaluator approaches the task: understand what was asked, read the answer, check it against the evidence.

---

### Problem 3: LLMs Score Generously (Mode Collapse to High Scores)

Without explicit instruction, LLMs rate things highly. A lenient evaluator provides no useful signal because everything scores 0.9+.

**How MASIS solves it — Explicit Rubric with Anchors and Strictness Instruction:**

Each dimension has three explicit anchor points:

```
Faithfulness:
1 = every claim directly supported by cited evidence
0.5 = partially supported
0 = unsupported
```

Plus a hard instruction: `"Be strict. Do NOT default to 1."` and `"Use the citation audit findings to calibrate Faithfulness accurately."`

---

### Problem 4: Evaluator Can Contradict the Citation Engine

The LLM evaluator has no direct knowledge of the citation engine's deterministic findings. It could assign faithfulness = 0.9 to an answer the citation engine already proved contains fabricated chunk IDs.

**How MASIS solves it — Citation Audit Injection as Hard Constraints:**

```python
citation_audit = state.get("metrics", {}).get("last_citation_audit", {})
invalid_citations = citation_audit.get("invalid_citations", [])
uncited_count = citation_audit.get("uncited_claim_count", 0)
hallucination_detected = citation_audit.get("hallucination_detected", False)

citation_context = f"""
=== Citation Audit Results (from Critic) ===
- Invalid citation IDs found: {invalid_citations if invalid_citations else "None"}
- Sentences with NO citation: {uncited_count}
- Hallucination detected: {hallucination_detected}

IMPORTANT scoring rules:
- If uncited sentences >= 5, Faithfulness MUST be <= 0.5
- If uncited sentences >= 10, Faithfulness MUST be <= 0.3
- If hallucination_detected is True, Faithfulness MUST be <= 0.4
- If invalid citations exist, Faithfulness MUST be <= 0.4
- These are hard constraints — do not override them.
"""
```

The Critic stores its findings in `state["metrics"]["last_citation_audit"]`. The Evaluator reads them and receives them as explicit scoring constraints in the prompt.

---

### Problem 5: LLM Ignores the Constraints Anyway

Even with explicit instructions, LLMs sometimes override them. The Evaluator's prompt says faithfulness must be ≤ 0.4 when hallucination is detected — but the LLM might still return 0.7.

**How MASIS solves it — Post-Response Hard Clamps:**

```python
if hallucination_detected or invalid_citations:
    evaluation["faithfulness"] = min(evaluation["faithfulness"], 0.4)

if uncited_count >= 10:
    evaluation["faithfulness"] = min(evaluation["faithfulness"], 0.3)
elif uncited_count >= 5:
    evaluation["faithfulness"] = min(evaluation["faithfulness"], 0.5)

# Recalculate overall_score as weighted mean after clamping
evaluation["overall_score"] = round(
    evaluation["faithfulness"] * 0.35 +
    evaluation["relevance"] * 0.25 +
    evaluation["completeness"] * 0.25 +
    evaluation["reasoning_quality"] * 0.15,
    3
)
```

After the LLM responds, faithfulness is clamped by code regardless of what the LLM returned. The overall score is then recalculated as a weighted mean with faithfulness weighted highest (35%) — reflecting that a hallucinated answer, however well-written, is fundamentally untrustworthy. This guarantees the deterministic citation engine's findings are always reflected in the final evaluation.

---

### Problem 6: Scale Mismatch

Some LLMs return scores on a 0–100 scale instead of 0–1.

**How MASIS solves it:**

```python
for k in ["faithfulness", "relevance", "completeness", "reasoning_quality", "overall_score"]:
    if evaluation.get(k, 0) > 1:
        evaluation[k] = evaluation[k] / 100.0
```

Applied before the hard clamps, so clamping always operates on a 0–1 scale.

---

## The Four Evaluation Dimensions

| Dimension | What It Measures | Weight in Overall Score |
|---|---|---|
| **Faithfulness** | Is every claim grounded in retrieved evidence? | 35% |
| **Relevance** | Does the answer address what was asked? | 25% |
| **Completeness** | Are all aspects of the question covered? | 25% |
| **Reasoning Quality** | Is the answer well-structured and logically sound? | 15% |

Faithfulness is weighted highest because a hallucinated answer, however complete and relevant, is dangerous in a strategic intelligence context.

---

## Critic vs. Evaluator — Key Distinction

| | Critic | Evaluator |
|---|---|---|
| **Purpose** | Decides whether to retry | Measures permanent quality |
| **Affects routing** | Yes | No |
| **Runs per iteration** | Every cycle | Once, after last Critic pass |
| **Output consumed by** | Supervisor + Researcher (on retry) | API response + monitoring |
| **Citation findings flow** | Produces `last_citation_audit` | Consumes `last_citation_audit` |

The Critic is an operational control gate. The Evaluator is a measurement instrument. They are in a producer-consumer relationship via `last_citation_audit`.

---

## State Inputs & Outputs

| Field | Direction | Description |
|---|---|---|
| `user_query` | Input | Needed for relevance scoring |
| `final_answer` | Input | The audited draft |
| `evidence` | Input | For faithfulness scoring |
| `metrics.last_citation_audit` | Input | Critic's deterministic findings |
| `metrics.evaluation` | **Output** | Full evaluation with all scores |
| `metrics.node_latency_ms.evaluator` | **Output** | Time taken |
| `trace` | **Output** | Evaluation summary entry |

---

## Telemetry Emitted

```json
{
  "node": "evaluator",
  "overall_score": 0.71,
  "faithfulness": 0.40,
  "relevance": 0.88,
  "completeness": 0.72,
  "duration_ms": 987
}
```

A faithfulness of 0.40 here would indicate the hard clamp fired — the LLM may have returned a higher score but the citation engine found hallucinations and the code overrode it.

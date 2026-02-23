# 📊 Agent 5: Evaluator Node — Deep Dive

## Role & Persona

The Evaluator is the **"Quality Scorecard"** of MASIS. It is an implementation of the **LLM-as-a-Judge** pattern — an independent LLM that scores the quality of the final answer across four dimensions: Faithfulness, Relevance, Completeness, and Reasoning Quality. Unlike the Critic (which decides whether to retry), the Evaluator produces a permanent quality record that travels with the response for observability, monitoring, and continuous improvement.

The Evaluator does not affect routing. It is a pure measurement agent.

---

## What Problem Does It Solve?

### Problem 1: No Objective Quality Signal After the Conversation

In production, knowing whether the system produced good answers is essential for monitoring, alerting, and model improvement. Without an automated quality signal, the only feedback is user complaints — which are rare, slow, and biased. You need automated evaluation running on every request.

**How MASIS solves it — LLM-as-a-Judge with Structured Scores:**

```python
class Evaluation(BaseModel):
    faithfulness: float        # groundedness (0–1)
    relevance: float           # question alignment (0–1)
    completeness: float        # coverage (0–1)
    reasoning_quality: float   # clarity & logic (0–1)
    overall_score: float       # weighted average
    improvement_suggestions: list[str]
```

The Evaluator produces a structured score record stored in `state["metrics"]["evaluation"]`. This record is returned in every API response and can be logged, monitored in dashboards, or aggregated for model performance analysis over time.

---

### Problem 2: Evaluator Cannot Score Relevance Without Knowing the Question

This was a real bug in the original implementation. The prompt sent to the evaluator included the answer and the evidence, but **not the user's question**. The evaluator was asked to score "Relevance: Does this fully answer the user question?" — but it couldn't see what the question was.

**How MASIS solves it — User Query Explicitly in the Evaluation Prompt:**

```python
prompt = f"""
...
Relevance:
1 = fully answers user question
0.5 = partially relevant
0 = irrelevant

Completeness:
1 = covers all aspects of the question
...

Question:
{query}

Answer:
{answer}

Evidence:
{context}
"""
```

The question is now the first piece of content the evaluator sees (after the rubric), followed by the answer, followed by the evidence. This ordering mirrors how a human evaluator would approach the task: first understand what was asked, then read the answer, then check it against the evidence.

---

### Problem 3: LLMs Tend to Score Generously (Mode Collapse to High Scores)

Without explicit instruction, LLMs rate things highly — they are trained to be helpful and agreeable. A lenient evaluator provides no useful signal because everything scores 0.9+.

**How MASIS solves it — Explicit Rubric with Anchors and a Strictness Instruction:**

Each dimension has three explicit anchor points (0, 0.5, 1) with concrete definitions, not just "low/medium/high":

```
Faithfulness:
1 = every claim directly supported by cited evidence
0.5 = partially supported
0 = unsupported
```

And a hard instruction at the top:

```
Be strict. Do NOT default to 1.
```

The combination of rubric anchors and an explicit anti-leniency instruction produces more calibrated, discriminating scores than leaving it to the model's judgment.

---

### Problem 4: Scores Returned in the Wrong Scale

The same issue exists here as in the Critic — some LLMs return confidence on a 0–100 scale rather than 0–1, despite prompt instructions.

**How MASIS solves it — Score Normalization Across All Metrics:**

```python
for k in ["faithfulness", "relevance", "completeness", "reasoning_quality", "overall_score"]:
    if evaluation.get(k, 0) > 1:
        evaluation[k] = evaluation[k] / 100.0
```

All five numeric fields are checked and normalized. This handles both scale conventions gracefully.

---

### Problem 5: Improvement Suggestions Are Actionable, Not Just Descriptive

The `improvement_suggestions` field is not cosmetic. It's a list of strings the LLM generates explaining how the answer could be improved. In a production system, these can be:

- Displayed to the user alongside the answer ("This answer may have gaps in X")
- Logged and aggregated to identify systematic weaknesses ("We keep missing Y type of question")
- Fed back into prompt engineering cycles to improve the Synthesizer's prompt

This is the bridge between per-request quality and long-term system improvement.

---

## The Four Evaluation Dimensions

| Dimension | What It Measures | Key Question |
|---|---|---|
| **Faithfulness** | Is every claim grounded in the retrieved evidence? | Are there hallucinations or unsupported inferences? |
| **Relevance** | Does the answer actually address what was asked? | Did the system answer the right question? |
| **Completeness** | Are all aspects of the question covered? | Did the system miss important sub-questions? |
| **Reasoning Quality** | Is the answer well-structured and logically sound? | Is the answer clear, coherent, and well-reasoned? |

These map directly to the case study's three core metrics (Faithfulness, Relevance, Completeness) with Reasoning Quality added as a fourth dimension for structural quality.

---

## Relationship Between Critic and Evaluator

These two agents serve distinct purposes and should not be confused:

| | Critic | Evaluator |
|---|---|---|
| **Purpose** | Decides whether to retry | Measures permanent quality |
| **Affects routing** | Yes | No |
| **Runs per iteration** | Every cycle | Once (after Critic, before Supervisor return) |
| **Output used by** | Supervisor | API response / monitoring |
| **Model** | `gpt-4o` | `gpt-4o` |
| **Output type** | Structured `Critique` | Structured `Evaluation` |

The Critic is an operational control — it gates progression. The Evaluator is a measurement instrument — it records outcomes.

---

## State Inputs & Outputs

| Field | Direction | Description |
|---|---|---|
| `user_query` | Input | The original question (needed for relevance scoring) |
| `final_answer` | Input | The audited draft answer |
| `evidence` | Input | Evidence chunks (for faithfulness scoring) |
| `metrics.evaluation` | **Output** | Full evaluation dict with all scores |
| `metrics.node_latency_ms.evaluator` | **Output** | Time taken |
| `trace` | **Output** | Appended evaluation summary |

---

## Telemetry Emitted

```json
{
  "node": "evaluator",
  "overall_score": 0.84,
  "faithfulness": 0.92,
  "relevance": 0.88,
  "completeness": 0.72,
  "duration_ms": 987
}
```

---

## Model Choice Rationale

The Evaluator uses **`gpt-4o`** (same as the Critic). Justification:

- Evaluation requires sophisticated reading comprehension — understanding whether claims are truly grounded requires the same reasoning depth as detecting hallucinations.
- Using `gpt-4o-mini` here would produce noisier, less calibrated scores, defeating the purpose of the measurement.
- The cost is justified: evaluation runs once per request (not once per retry), and the quality of its scores directly affects the reliability of your entire monitoring infrastructure.

# ✍️ Agent 2: Synthesizer Node — Deep Dive

## Role & Persona

The Synthesizer is the **"Writer"** of MASIS. It takes the raw evidence chunks retrieved by the Researcher and constructs a coherent, citation-grounded answer. It is the only agent that produces prose output — every other agent retrieves, judges, routes, or scores.

Its output (`draft_answer`) is intentionally called a *draft* — it is not final. It passes to the Critic for auditing before being considered complete.

---

## What Problem Does It Solve?

### Problem 1: Context Window Explosion

In a naive RAG system, all retrieved chunks are concatenated and dumped into the prompt. With 10 chunks of 800 characters each that's 8,000 characters before the question is asked. This causes:

- **Lost-in-the-middle**: LLMs perform poorly on information buried in the centre of long contexts.
- **Token cost**: Every extra character costs money.
- **Latency**: Longer prompts take longer to process.

**How MASIS solves it — Intelligent Context Compression:**

```python
MAX_CONTEXT_CHARS = 6000

if original_context_chars > MAX_CONTEXT_CHARS:
    sorted_evidence = sorted(evidence, key=lambda x: x.score, reverse=True)
    top_chunks = sorted_evidence[:3]   # kept full
    low_chunks = sorted_evidence[3:]   # summarized
```

Tiered approach:
1. Sort all chunks by relevance score (descending).
2. Top 3 highest-scoring chunks preserved in **full fidelity** — these are most critical.
3. All remaining chunks compressed to under 200 characters via `gpt-4o-mini` (temperature=0), which explicitly preserves numbers and metrics.

Highest-value evidence is never degraded. Lower-ranked context is present but condensed.

---

### Problem 2: Over-Compression Causing Silent Evidence Loss

Compression is useful but risky. If too aggressive, critical details in lower-ranked chunks (a contract date, a revenue figure) may be lost.

**How MASIS solves it — Over-Compression Detection:**

```python
if compressed_chars / original_context_chars < 0.35:
    state["metrics"]["over_compression_flag"] = True
    current_critique["needs_retry"] = True
    current_critique["logical_gaps"] = current_critique.get("logical_gaps", []) + [
        "Evidence was over-compressed; critical context may have been lost."
    ]
    state["critique"] = current_critique
```

If the compressed context is less than 35% of the original size, `needs_retry` is injected into the critique — the same field the Critic uses — so the Supervisor triggers a retry with broader retrieval before the Critic even runs. Fast feedback loop.

---

### Problem 3: Hallucinations From Unconstrained Generation

Without strict instructions, LLMs blend retrieved content with hallucinated details. In a strategic intelligence system this is catastrophic — a recommendation built on a hallucinated data point leads to a wrong business decision.

**How MASIS solves it — Citation-Mandatory Prompting with Strategic Persona:**

```python
prompt = f"""
You are a strategic intelligence analyst.
Use ONLY the evidence below to answer the question.
Every factual claim MUST cite its source using [chunk_id].
If the evidence only partially covers the question, answer what you can and
explicitly state which aspects lack sufficient evidence — do not fabricate.
...
"""
```

Three constraints enforced through the prompt:

1. **Strategic persona** — frames the task as analysis, not creative writing.
2. **"Use ONLY the evidence"** — explicit prohibition on external knowledge.
3. **"Every claim must cite [chunk_id]"** — forces inline citation of every factual statement.
4. **Partial evidence handling** — gives the model a safe exit ("state which aspects lack sufficient evidence") rather than hallucinating to fill gaps. These hedge phrases are excluded from the Critic's uncited-claims penalty.

The Critic then programmatically verifies these constraints are met, not just intended.

---

### Problem 4: Repeated Mistakes on Retry

If a first-pass answer had specific hallucinations, and the Synthesizer is called again on retry without any awareness of what went wrong, it produces a similar answer.

**How MASIS solves it — Critique Feedback Injection:**

```python
critique_feedback = ""
if retry_count > 0 and critique:
    critique_feedback = f"""
Previous critique:
Hallucination: {critique.get("hallucination_detected")}
Unsupported claims: {critique.get("unsupported_claims")}
Logical gaps: {critique.get("logical_gaps")}
Conflicts: {critique.get("conflicting_evidence")}
Correct these issues.
"""
```

On any retry, the full structured Critic output is injected as a preamble. The Synthesizer knows exactly what was wrong with its previous answer and is explicitly told to correct those issues. This is the self-correction loop: Researcher fetches better evidence, Synthesizer receives that evidence plus a "here's what you got wrong" brief.

---

### Problem 5: Compression Parsing Failures

The compression LLM returns free-form text. If it adds commentary instead of following the format, the pipeline must not crash.

**How MASIS solves it — Graceful Fallback:**

```python
for e in low_chunks:
    summary = compressed_map.get(e.chunk_id, e.text[:200])  # fallback: truncate
```

If a chunk's ID is missing from the compression output, the system falls back to a hard truncation of the original text to 200 characters. The pipeline continues with degraded but non-null evidence rather than crashing.

---

## State Inputs & Outputs

| Field | Direction | Description |
|---|---|---|
| `user_query` | Input | The original question |
| `evidence` | Input | Score-filtered `EvidenceChunk` list from Researcher |
| `retry_count` | Input | Determines whether critique feedback is injected |
| `critique` | Input | Previous Critic output (injected on retries) |
| `draft_answer` | **Output** | Generated answer with inline citations |
| `critique` | **Output** | May be mutated if over-compression detected |
| `metrics` | **Output** | `original_context_chars`, `compressed_context_chars`, `compression_ratio`, `citation_count`, `answer_length`, `compression_latency_ms` |
| `trace` | **Output** | Synthesis stats entry |

---

## Telemetry Emitted

```json
{
  "node": "synthesizer",
  "retry_count": 0,
  "context_chars": 7840,
  "context_compressed": true,
  "compression_latency_ms": 621,
  "answer_length": 1243,
  "citations": 7,
  "duration_ms": 1854
}
```

---

## Model Choice Rationale

- **`gpt-4o-mini`** for synthesis and compression. Synthesis is a **generation task** — the model needs fluency and instruction-following, not deep logical analysis. Compression is a **summarization task** — simpler still. Mini costs ~15-20x less than `gpt-4o` and is fully capable of both constrained tasks.
- The Critic (which must detect subtle semantic hallucinations) uses `gpt-4o`. That asymmetry is deliberate: the auditor is stronger than the writer.

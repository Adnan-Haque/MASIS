# ✍️ Agent 2: Synthesizer Node — Deep Dive

## Role & Persona

The Synthesizer is the **"Writer"** of MASIS. It takes the raw evidence chunks retrieved by the Researcher and constructs a coherent, citation-grounded answer to the user's query. It is the only agent that produces prose output. Every other agent either retrieves data, judges data, routes decisions, or scores quality.

Its output (`draft_answer`) is intentionally called a *draft* — it is not final. It passes to the Critic for auditing before ever being considered complete.

---

## What Problem Does It Solve?

### Problem 1: Context Window Explosion

In a naive RAG system, all retrieved chunks are concatenated and dumped into the prompt. With 10 chunks of 800 characters each, that's 8,000 characters of context before the question is even asked. With larger documents or more chunks, this grows fast and causes several failures:

- **Lost-in-the-middle**: LLMs perform poorly on information buried in the centre of long contexts. Critical evidence in the middle of a 10-chunk prompt is frequently ignored.
- **Token cost**: Every extra character costs money in API calls.
- **Latency**: Longer prompts take longer to process.

**How MASIS solves it — Intelligent Context Compression:**

```python
MAX_CONTEXT_CHARS = 6000

if original_context_chars > MAX_CONTEXT_CHARS:
    sorted_evidence = sorted(evidence, key=lambda x: x.score, reverse=True)
    top_chunks = sorted_evidence[:3]   # kept full
    low_chunks = sorted_evidence[3:]   # summarized
```

The approach is tiered:

1. All chunks are sorted by **relevance score** (descending).
2. The top 3 highest-scoring chunks are preserved in **full fidelity** — these are most critical.
3. All remaining lower-ranked chunks are passed through a **compression LLM** (`gpt-4o-mini, temperature=0`) which summarizes each one to under 200 characters while explicitly preserving numbers and metrics.
4. The compressed summaries replace the full text for low-ranked chunks only.

This ensures the highest-value evidence is never degraded while still controlling total context size.

---

### Problem 2: Over-Compression Causing Silent Evidence Loss

Compression is useful but risky. If too aggressive, critical details in lower-ranked chunks (e.g., a specific contract date, a revenue figure) may be lost. The system needs to detect and react to this.

**How MASIS solves it — Over-Compression Detection with Retry Signal:**

```python
if compressed_chars / original_context_chars < 0.35:
    state["metrics"]["over_compression_flag"] = True
    current_critique["needs_retry"] = True
    current_critique["logical_gaps"] = current_critique.get("logical_gaps", []) + [
        "Evidence was over-compressed; critical context may have been lost."
    ]
    state["critique"] = current_critique
```

If the compressed context is less than 35% of the original size, it's flagged as over-compressed. The system injects a `needs_retry = True` signal into the critique — the same field the Critic uses — so the Supervisor will trigger a retry with a broader retrieval, before the Critic even runs. This creates a fast feedback loop that short-circuits the pipeline.

---

### Problem 3: Hallucinations From Unconstrained Generation

Without strict instructions, LLMs will synthesize plausible-sounding answers that blend retrieved content with hallucinated details. In a strategic intelligence system, this is catastrophic — a recommendation built on a hallucinated data point could lead to a wrong business decision.

**How MASIS solves it — Citation-Mandatory Prompting:**

```python
prompt = f"""
Use ONLY the evidence below.
Every claim must cite [chunk_id].
If there is insufficient evidence, explicitly state so.
...
"""
```

Three constraints are enforced through the prompt:

1. **"Use ONLY the evidence"** — explicit prohibition on external knowledge.
2. **"Every claim must cite [chunk_id]"** — forces inline citation of every factual statement.
3. **"If insufficient evidence, explicitly state so"** — gives the model a safe exit rather than hallucinating to fill a gap.

The Critic then programmatically verifies these constraints are actually met (not just intended).

---

### Problem 4: Repeated Mistakes on Retry

If a first-pass answer had specific hallucinations or unsupported claims, and the Synthesizer is called again on retry, it would naively produce a similar answer without any awareness of what went wrong.

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

On any retry (retry_count > 0), the full structured output of the previous Critic run is injected as a preamble to the synthesis prompt. The Synthesizer now knows exactly what was wrong with its previous answer: which claims were unsupported, what logical gaps were identified, and what conflicting evidence was found. It is explicitly instructed to correct those issues.

This is the self-correction loop in action — Researcher fetches better evidence, Synthesizer receives that evidence plus a "here's what you got wrong" brief.

---

### Problem 5: Parsing Compressed Output Reliably

The compression LLM returns free-form text. If parsing fails (e.g., the LLM adds commentary instead of following the format), the fallback must ensure the pipeline doesn't crash.

**How MASIS solves it — Graceful Fallback in Compression Parsing:**

```python
for line in compressed_output.split("\n"):
    if ":" in line:
        cid, summary = line.split(":", 1)
        compressed_map[cid.strip().replace("[","").replace("]","")] = summary.strip()

for e in low_chunks:
    summary = compressed_map.get(e.chunk_id, e.text[:200])  # fallback: truncate
```

If a chunk's ID is not found in `compressed_map` (because the LLM didn't follow the format for that chunk), the system falls back to a hard truncation of the original text to 200 characters. This is never ideal, but it's safe — the pipeline continues with degraded but non-null evidence rather than crashing.

---

## State Inputs & Outputs

| Field | Direction | Description |
|---|---|---|
| `user_query` | Input | The original question |
| `evidence` | Input | List of `EvidenceChunk` from Researcher |
| `retry_count` | Input | Determines whether critique feedback is injected |
| `critique` | Input | Previous Critic output (on retries) |
| `draft_answer` | **Output** | The generated answer with inline citations |
| `critique` | **Output** | May be mutated if over-compression detected |
| `metrics` | **Output** | `original_context_chars`, `compressed_context_chars`, `compression_ratio`, `citation_count`, `answer_length`, `compression_latency_ms` |
| `trace` | **Output** | Appended entry with synthesis stats |

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

The Synthesizer uses **`gpt-4o-mini`** for both synthesis and compression. Justification:

- Synthesis is a **generation task**, not a reasoning task. The model needs fluency and instruction-following, not deep logical analysis.
- Compression is a **summarization task** — also well within mini's capability.
- The Critic (which requires harder reasoning to detect subtle hallucinations) uses `gpt-4o`. This is the key model tier split in the system.
- Using mini here saves roughly 15-20x cost per synthesis call compared to `gpt-4o`.

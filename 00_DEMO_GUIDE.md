# 🎯 MASIS Demo — Interview Dataset & Question Guide

## The Dataset: NovaTech Solutions

A fictional B2B SaaS company (AI-powered supply chain software). Five documents that together tell a complete business story — and are carefully designed to trigger every MASIS capability.

| File | What's in it | Why it's useful for the demo |
|---|---|---|
| `01_annual_report_FY2023.pdf` | Revenue, ARR, margins, AI investment, headcount, churn | Rich factual content → happy path answers with citations |
| `02_board_strategy_memo_AI_roadmap.pdf` | AI product roadmap, compute budget, competitive risks | Deep strategy content → tests synthesis quality |
| `03_Q3_FY2023_board_pack.pdf` | Q3 financials, NRR of 111% (conflicts with annual report's 114%) | **Deliberately conflicts** with Doc 1 → triggers HITL conflict path |
| `04_churn_analysis_FY2023.pdf` | Churn breakdown, competitor displacement, retention by segment | Specific data points → tests citation engine |
| `05_competitive_intelligence_chainmind.pdf` | ChainMind AI profile, funding, product comparison | Cross-document reasoning → tests multi-doc synthesis |

---

## Setup Instructions

1. Create a workspace called `novatech` in MASIS
2. Upload all 5 PDFs
3. Wait for indexing to complete (watch document count hit 5)
4. Use the queries below in order — they're sequenced to show progressively more impressive capabilities

---

## Query Bank — Ordered for Demo

### 🟢 Group 1: Happy Path — Clear, Well-Cited Answers
*These should return high confidence (0.80+), no retries, clean citations*

---

**Q1 — Basic financial fact retrieval**
```
What was NovaTech's total revenue and ARR in FY2023, and how did this compare to FY2022?
```
**What to expect:** Confidence ~0.85+. Answer cites the annual report with exact figures ($61.7M revenue, $57.3M ARR, +28% and +30% growth). Good opening query — clean answer proves the system works.

**What to point out to interviewer:** Look at the trace — Researcher fetched 5 chunks, Synthesizer cited every figure with a chunk ID, Critic found no invalid citations.

---

**Q2 — Multi-fact synthesis**
```
How much did NovaTech invest in AI R&D in FY2023, and what were the three specific areas this investment was directed toward?
```
**What to expect:** Confidence ~0.82+. Answer references $8.2M investment (13.3% of revenue) and names the three pillars: predictive demand forecasting, natural language interface, anomaly detection. All sourced from the annual report.

**What to point out:** Citation count will be 3–4 — every claim has a bracket reference.

---

**Q3 — Cross-document synthesis**
```
What is NovaTech's competitive position against ChainMind AI, and what specific product gaps does NovaTech need to close?
```
**What to expect:** Confidence ~0.78–0.85. Answer pulls from BOTH the competitive intelligence brief AND the churn analysis — ChainMind's price advantage (38% cheaper), Shopify/WooCommerce gap, and the 6 displaced accounts. This is the "multi-document synthesis" showcase.

**What to point out:** The Researcher retrieved chunks from two separate documents. The Synthesizer synthesised them into one coherent answer. The trace will show avg_score from the retrieval step.

---

**Q4 — Strategic recommendation**
```
Based on the churn analysis, what are the top three recommended actions to reduce customer churn at NovaTech, and what is the estimated revenue impact?
```
**What to expect:** Confidence ~0.80+. Three clear recommendations from Doc 4: SMB self-serve tier (+$3.2M ARR), Shopify/WooCommerce connectors (6-week build), mandate professional services for complex integrations ($1.2M ARR saved). All cited.

**What to point out:** The system is doing *reasoning* — it's not just extracting a sentence, it's synthesising a prioritised recommendation from a structured analysis document.

---

### 🟡 Group 2: Retry Path — System Self-Corrects
*These are intentionally ambiguous and may trigger one retry cycle*

---

**Q5 — Vague query that needs broader retrieval**
```
What is NovaTech's plan to defend its market position over the next two years?
```
**What to expect:** Likely triggers one retry. First pass may miss the board memo's Priority 2 (data consortium) or Priority 3 (Copilot). On retry, augmented query fetches broader chunks. Confidence improves from ~0.65 to ~0.80+.

**What to point out to interviewer:** Watch the trace carefully. You'll see:
- Iteration 1: `chunks: 5, augmented_query_used: false`
- Supervisor: `decision: retry, reason: quality_issue_detected`
- Iteration 2: `chunks: 10, augmented_query_used: true`

Say: *"This is the self-correction loop in action. The Critic told the Researcher exactly what was missing, and the Researcher widened its search to fill those gaps."*

---

**Q6 — Requires information from a hard-to-retrieve section**
```
What is the estimated burn rate and runway of ChainMind AI, and what does this mean for their competitive threat to NovaTech?
```
**What to expect:** May retry once. The burn rate ($2.8M/month, 16–18 months runway) is in the competitive intelligence doc but in a specific section. The synthesis should conclude ChainMind needs Series C by mid-2025 — adding strategic context beyond just the numbers.

**What to point out:** The Evaluator's Reasoning Quality score will be high here if the system correctly drew the strategic implication.

---

### 🔴 Group 3: Conflict Detection — HITL Triggered
*This will trigger the conflict escalation path — the most impressive demo moment*

---

**Q7 — The deliberate conflict query** ⭐ *Save this for the climax of your demo*
```
What was NovaTech's Net Revenue Retention (NRR) in FY2023?
```
**What to expect:** The Annual Report says **114%**. The Q3 Board Pack says **111%** (with a note that it's a preliminary TTM figure pending Q4 reconciliation). The Critic will detect conflicting evidence across these two documents.

After retries fail to resolve it, the Supervisor will return:
```
requires_human_review: true
clarification_question: "Conflicting information was detected across documents and 
could not be automatically resolved after multiple attempts. Please review the 
competing claims and select a preferred source."
```

**What to point out to interviewer:**
> *"This is exactly the kind of thing that breaks naive RAG systems — they'd just pick whichever chunk had a higher similarity score and confidently return a wrong number. MASIS detects the conflict, attempts to resolve it through two retry cycles, can't resolve it definitively because both numbers are technically correct for different time periods, and escalates to the human with a clear explanation. The system knows what it doesn't know."*

This is your **strongest demo moment**. Pause on this result.

---

### ⚫ Group 4: Edge Cases
*Show these only if the interviewer asks about robustness*

---

**Q8 — Out-of-scope query**
```
What is the current stock price of NovaTech Solutions?
```
**What to expect:** The system should return low confidence and may trigger HITL — none of the documents contain stock price information. The answer should explicitly say "insufficient evidence" rather than hallucinating a number.

**What to say:** *"The system correctly identifies when it doesn't have the evidence to answer rather than making something up."*

---

**Q9 — Very specific numerical query**
```
How many PhD-level researchers did NovaTech hire through its AI Fellowship programme, and what was the contract duration?
```
**What to expect:** Clean answer — 12 PhD researchers, 18-month rotational contracts. This is a very specific detail buried in the Annual Report. Tests whether the retrieval actually finds niche information.

---

**Q10 — Cross-document numerical comparison**
```
Compare NovaTech's AI R&D investment as a percentage of revenue between FY2022 and FY2023, and explain the strategic rationale for the increase.
```
**What to expect:** Answer should quote 8.5% (FY2022) vs 13.3% (FY2023), then reference the board memo's strategic thesis about commoditisation and the need to build a proprietary data moat before competitors catch up.

---

## What Each Query Demonstrates

| Query | Capability Demonstrated |
|---|---|
| Q1 | Basic RAG, accurate citation, happy path |
| Q2 | Multi-fact extraction, citation density |
| Q3 | Cross-document synthesis |
| Q4 | Reasoning & recommendation generation |
| Q5 | Self-correction loop, retry with augmented query |
| Q6 | Confident retrieval of specific numerical data |
| Q7 | **Conflict detection → HITL escalation** |
| Q8 | Hallucination refusal, evidence sufficiency check |
| Q9 | Fine-grained retrieval of specific details |
| Q10 | Numerical reasoning across documents + strategic synthesis |

---

## Anticipated Interviewer Questions During Demo

**"How does it know there's a conflict between documents?"**
> The Critic's LLM audit is specifically prompted to identify conflicting_evidence — statements where two retrieved chunks make opposing claims. It surfaces both the 111% and 114% figures and flags them as contradictory. The Supervisor then tries to resolve it through retries before escalating.

**"What if the 114% is correct and 111% is wrong — why doesn't it just pick one?"**
> Both are technically correct — 111% was the TTM figure at Q3, 114% was the full-year audited figure. The system can't determine which is authoritative without human context, so it correctly asks the human to decide. This is exactly the right behaviour for a strategic intelligence system.

**"What does the confidence score actually mean?"**
> It's the Critic's LLM-assessed quality score, penalised by the citation engine. A 50% penalty is applied for any fabricated chunk references, and a 10% penalty for uncited claims. So 0.85 means: the LLM assessed high quality AND the citation engine found no fake references AND all major claims are backed by evidence.

**"Could the system give a wrong answer with high confidence?"**
> Yes — if the documents themselves contain wrong information, the system will faithfully reproduce and cite it. MASIS is a retrieval system, not a fact-checking system. It guarantees that answers are grounded in your documents; it cannot guarantee the documents are correct. This is an important limitation to be transparent about.

**"What happens if I ask a question the documents don't answer?"**
> The Synthesizer is instructed to say "insufficient evidence" rather than fabricate. The Critic would flag any hallucinated answer, triggering retries. After retries fail (no new evidence found), the system escalates to HITL asking the user to upload relevant documents.

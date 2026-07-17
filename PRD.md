# PRD — Technical & Regulatory Evidence Retrieval System (Bionema POC)

## 1. Purpose

Build a hybrid RAG system that answers plain-language questions over patent, trial,
and regulatory documents with clause-level citations, deployed as a public demo URL,
to convert a stalled B2B sales conversation (Dr Minshad Ansari, CEO, Bionema Group)
into a paid pilot.

This is a sales artifact first, an engineering showcase second. Every design
decision should be defensible in a follow-up technical conversation with a
skeptical, non-technical-but-scientifically-rigorous buyer.

## 2. Background

Prior context: a written proposal was already sent and accepted in principle.
The prospect agreed to a free proof of concept using public/synthetic documents
only, no confidential material, no cost, no obligation. If convincing, next step
is a paid fixed-scope pilot (~£1,500–£2,500).

The source document for this POC is Bionema's own granted patent
**WO2020053603A1** ("Insect-Pathogenic Fungus, Spores, Composition and Use of
Same," Bionema Ltd, inventor Minshad Ali Ansari) — public record, zero
confidentiality risk, but recognizably his own science.

## 3. Goals

- **G1 — Retrieval accuracy**: Answers must be grounded strictly in retrieved
  source text. No hallucinated claims. Every answer cites the exact paragraph
  ID (e.g. `[0072]`) it came from.
- **G2 — Visible citation quality**: Citations are not a footnote — they are
  the main visual proof point. Every answer shows the cited passage inline,
  highlighted, with a confidence indicator.
- **G3 — Retrieval that actually works on legal/technical text**: exact terms
  (strain names, patent numbers, GO numbers) must not get lost to pure
  semantic search. Hybrid retrieval (BM25 + dense) is mandatory, not optional.
- **G4 — Provable quality, not just a nice UI**: ship a small evaluation
  harness (Recall@5, MRR, groundedness) with results visible on the demo page
  or in an accompanying doc. This is the single biggest differentiator vs. a
  generic RAG chatbot demo.
- **G5 — Zero cost, zero login**: entirely on free tiers (Render/Fly.io for
  the backend, Vercel for the frontend), single shareable URL for the
  prospect, no setup required from them.
- **G6 — Refuse gracefully**: when no chunk answers the question with
  confidence, say so explicitly rather than guessing. This mirrors the
  prospect's own stated design philosophy (surfaced in his own government
  whitepaper) — deliberately echo it.

## 4. Non-goals

- No user authentication / multi-tenant support (out of scope for POC).
- No ingestion UI — document set is fixed and pre-processed for this POC.
- No production-grade uptime/SLA — this is a sales demo, not a committed
  service.
- No handling of confidential Bionema material at this stage — public patent
  text only, per the agreed scope.

## 5. Users

- **Primary**: Dr Minshad Ansari (non-technical evaluator, scientifically
  literate, skeptical of vendor claims, will test it himself).
- **Secondary**: any technical advisor he loops in to sanity-check the system.

## 6. System design

```
PDF (patent) → Docling (parse, preserve paragraph IDs + tables)
            → Chonkie (clause-aware recursive chunking, ~150–300 tokens)
            → Voyage AI embeddings (voyage-3-large)
            → Qdrant Cloud (dense + sparse hybrid collection)

Query → FastAPI /query endpoint
      → hybrid retrieve (dense + BM25/sparse, RRF fusion in Qdrant)
      → Voyage rerank-2 (top-20 → top-5)
      → Kimi API (synthesis constrained to retrieved chunks only,
         mandatory citation, explicit "not found" fallback)
      → JSON response → frontend renders citation chips, confidence badge,
         source highlight
```

### 6.0 Service split (backend/frontend)

The system is two independently deployable services, not one monolith:

- **Backend** — Python, FastAPI. Owns ingestion, retrieval, reranking, and
  synthesis. Exposes a small REST API (`POST /query`, `GET /health`,
  `GET /docs` for OpenAPI). Deployed on **Render** (or Fly.io) free tier —
  a real long-lived process, not a serverless function, since Qdrant/Voyage
  calls and Kimi synthesis benefit from a persistent app rather than
  cold-starting every request.
- **Frontend** — any framework (React/Next.js recommended for polish, but
  not required). Static/SSR app that calls the backend's public URL over
  HTTP. Deployed on **Vercel** free tier.
- These deploy and version independently. The backend is a standalone API
  that could plug into other frontends later (a selling point when talking
  to Bionema — this isn't a one-off demo app, it's a reusable retrieval
  service).

### 6.1 Ingestion pipeline
- Input: WO2020053603A1 full text (already extracted; see `/data/patent_raw.txt`).
- Docling preserves: paragraph numbers (`[0001]`–`[00187]`), table structures
  (Table 1–5), figure captions.
- Chonkie recursive chunker splits on paragraph-number boundaries and table
  boundaries as hard splits; falls back to sentence-aware splitting within
  long paragraphs. Each chunk carries metadata: `paragraph_id`, `section`,
  `doc_id`, `doc_title`.
- Embed each chunk with `voyage-3-large`. Store dense vector + generate sparse
  (BM25-style) vector via Qdrant's built-in sparse support or `fastembed`.
- Upsert to Qdrant Cloud collection `bionema_poc_v1`.

### 6.2 Retrieval
- Query embedded with same Voyage model.
- Qdrant hybrid query: dense + sparse, fused server-side (RRF).
- Top 20 candidates → Voyage `rerank-2` → top 5.
- Discard any result below a minimum rerank score threshold (configurable;
  start at 0.3) rather than force-feeding weak matches to synthesis.

### 6.3 Synthesis
- Kimi API call, system prompt constrains: answer only from provided
  chunks; every factual claim must carry `[paragraph_id]`; if no chunk
  supports an answer, respond with an explicit "not found in the provided
  documents" message — never fill gaps from general knowledge.
- Output structure: `{answer_text, citations: [{paragraph_id, quote, section}], confidence}`.

### 6.4 API contract (backend → frontend)

`POST /query` request: `{"question": string}`

`POST /query` response:
```json
{
  "answer": "string, may be empty if not found",
  "confidence": "high | partial | not_found",
  "citations": [
    {"paragraph_id": "[0072]", "section": "string", "quote": "string", "score": 0.0}
  ]
}
```

`GET /health` response: `{"status": "ok", "qdrant": "ok", "voyage": "ok", "kimi": "ok"}`

Frontend never talks to Qdrant/Voyage/Kimi directly — only to the
backend's `/query` and `/health` endpoints. Keeps API keys server-side only.

### 6.5 UI (frontend, any framework — React/Next.js recommended)
- Single page. Query box + example questions (chips).
- Each answer renders as a card: answer text with inline citation markers →
  expandable citation panel showing exact source passage, highlighted
  matched terms, and paragraph ID.
- Confidence badge (High / Partial / Not found) based on the `confidence`
  field returned by `/query`.
- No login. No document upload in v1 (fixed corpus).

### 6.6 Evaluation harness
- `/eval/qa_set.json` — 20–30 hand-labeled question → expected paragraph_id
  pairs, covering exact-lookup, paraphrase, and multi-hop-adjacent questions.
- `/eval/run_eval.py` computes:
  - Recall@5 (was correct paragraph in top 5 retrieved?)
  - MRR (mean reciprocal rank of the correct paragraph)
  - Groundedness (LLM-judge: does the cited passage actually support the
    generated claim? binary per answer)
- Results committed to `/eval/results.md`, and a summary block referenced on
  the demo page footer.

## 7. Success criteria (definition of done for the POC)

- [ ] Backend deployed to a public Render/Fly.io URL with working `/health`.
- [ ] Frontend deployed to a public Vercel URL, loads in <2s, works on
      mobile, successfully calls the backend across origins (CORS configured).
- [ ] All 6 example questions return correct, cited answers.
- [ ] At least one deliberately unanswerable question returns an explicit
      "not found" response, not a hallucination.
- [ ] Recall@5 ≥ 0.85 and groundedness ≥ 0.90 on the eval set.
- [ ] Every rendered answer has at least one visible, expandable citation.
- [ ] Zero ongoing cost at demo query volume (free tiers only).
- [ ] README explains architecture in plain terms suitable for forwarding
      to a non-technical stakeholder.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Free-tier rate limits hit during prospect's live testing | Cache common queries; keep eval query volume modest |
| Reranker/embedding API latency makes demo feel slow | Show retrieval progress state in UI; keep top-k small |
| Prospect asks about a topic not in the ingested patent | Explicit "not found" behavior (G6) turns this into a feature, not a bug |
| Docling/Chonkie version drift breaks ingestion reproducibility | Pin exact versions in `requirements.txt`, document ingestion as a one-time deterministic script |

## 9. Open questions

- Do we ingest more than one document (a second patent or a public regulatory
  guidance doc) to demonstrate cross-document retrieval before sending the
  link? Recommended: yes, at least 2 documents, to preempt "this only works on
  one PDF" skepticism.
- Do we surface the eval numbers directly in the UI footer, or keep them in a
  linked `/eval` page? Recommended: both — one-line summary in UI, full
  breakdown linked.

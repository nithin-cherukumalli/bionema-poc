# AGENTS.md — Bionema Retrieval POC

You are working inside a harness. Read this file fully before doing anything.
This file governs *how* you work — the PRD (`PRD.md`) governs *what* you're
building. Read both before writing code.

## 0. Read order, every session

1. This file (`AGENTS.md`)
2. `PRD.md` — full product spec
3. `claude-progress.md` — what happened last session
4. `feature_list.json` — what's done, what's next
5. `git log -10` — recent changes
6. Only then: pick a feature and start.

Do not skip step 3 or 4 even if you think you remember. State lives on disk,
not in your context window.

## 1. Scope discipline

Work on **exactly one feature** from `feature_list.json` per session unless
explicitly told otherwise. Do not:
- Start a second feature because the first one felt fast.
- Refactor unrelated code "while you're in there."
- Rewrite `feature_list.json` to mark something done that isn't verified.

If a feature turns out to be bigger than expected, split it into sub-tasks in
`feature_list.json` rather than silently expanding scope.

## 2. Definition of done (applies to every feature)

A feature is **not done** until:
1. Code runs without errors (`npm run build` or equivalent for the affected
   package).
2. Relevant tests pass (`npm test` / `pytest` for the affected package).
3. If the feature touches retrieval or synthesis, `eval/run_eval.py` has been
   re-run and results recorded in `eval/results.md` with a timestamp.
4. `claude-progress.md` has been updated with what changed and why.
5. `feature_list.json` status updated to `"done"` — only after 1–4, never
   before.

"It looks like it works" is not evidence. A passing verification command is
evidence.

## 3. Architecture ground rules (do not deviate without updating PRD.md)

- Parsing: Docling only. Do not substitute a raw PDF-to-text library — it
  will silently strip the paragraph IDs the whole citation system depends on.
- Chunking: Chonkie, recursive chunker, clause-aware split rules per
  `PRD.md §6.1`. Do not switch to naive fixed-token chunking.
- Embeddings: Voyage AI (`voyage-3-large`). Do not swap providers without
  updating the eval harness baseline — scores are not comparable across
  embedding models.
- Vector DB: Qdrant Cloud, hybrid (dense + sparse) collection. Do not
  introduce a second vector store.
- Synthesis: Kimi API only, with the citation-constrained system prompt in
  `/backend/synthesis/prompt.py`. Never let the model answer from general
  knowledge — every claim must trace to a retrieved chunk or the model must
  say "not found."
- Backend: Python, FastAPI, entirely. Do not introduce a Node/TypeScript
  backend layer — retrieval, reranking, and synthesis all live in Python so
  the whole backend is debuggable and testable with `pytest` alone.
- Frontend: any framework, treated as a separate deployable app that talks to
  the backend only over HTTP (`/query`, `/health`). Never let the frontend
  hold API keys or call Qdrant/Voyage/Kimi directly.
- Hosting: backend on Render or Fly.io (free tier — a real persistent
  process, not a serverless function); frontend on Vercel (free tier);
  Qdrant Cloud + Voyage + Kimi API for the AI layer. Do not introduce a
  paid service without flagging it explicitly — this project's selling point
  to the prospect is zero infrastructure cost.

## 4. Verification commands

Run these before marking any feature done. Add new ones to this list as the
project grows — do not let this list go stale.

```bash
# Backend — ingestion pipeline
python backend/ingest/parse_check.py
python backend/ingest/chunk_check.py
python backend/ingest/upsert_check.py

# Backend — retrieval, synthesis, API
pytest backend/

# Backend — eval harness (retrieval + groundedness quality)
python backend/eval/run_eval.py --report backend/eval/results.md

# Backend — local run + health check
uvicorn backend.main:app --reload &
curl -sf http://localhost:8000/health

# Frontend (commands depend on chosen framework — keep this block updated
# to match whatever's in frontend/package.json)
cd frontend && npm run lint && npm run build && npm test

# Full smoke test (ingestion → query → cited answer, end to end, against
# the running backend + deployed frontend URL)
python scripts/smoke.py --backend-url <url> --frontend-url <url>
```

If a verification command doesn't exist yet for a new subsystem, write it
before or alongside the feature that needs it — don't ship unverifiable code.

## 5. Session lifecycle

**Start of session**
1. Run `init.sh` (installs deps, checks env vars, health-checks Qdrant/Voyage
   connectivity).
2. Read files per §0.
3. State which single feature you're working on before writing any code.

**End of session**
1. Run full verification (§4).
2. Update `claude-progress.md`:
   - What was done
   - What was verified (and how)
   - What's still broken or unverified
   - What the next session should do first
3. Update `feature_list.json` status.
4. Commit only if the repo is in a state the next session (or a human) can
   safely resume from. Never commit with failing tests or broken builds.

## 6. What "good" looks like for this specific project

This is a sales artifact for a skeptical prospect who will personally test
it. Two failure modes matter more than usual here:

- **Hallucinated citations** — an answer that cites `[0072]` for a claim
  `[0072]` doesn't support is worse than no answer. If in doubt, lower
  confidence or return "not found."
- **Slow or broken demo on first load** — the prospect gets one shot at a
  good first impression. Cold-start latency, broken example questions, or a
  UI that doesn't render on mobile will cost the deal, not just look bad.

When choosing between "more features" and "the existing features being
rock solid," choose rock solid.

## 7. File map

```
/data/patent_raw.txt           Source document(s), pre-extracted

/backend/                      Python, FastAPI — everything AI/retrieval lives here
  main.py                      FastAPI app: /query, /health, CORS config
  ingest/                      Docling + Chonkie + Voyage + Qdrant upsert scripts
    parse.py, parse_check.py
    chunk.py, chunk_check.py
    embed_upsert.py, upsert_check.py
  retrieval/                   Hybrid query + rerank logic
    hybrid_query.py, rerank.py
  synthesis/                   Kimi API call + citation-constrained prompt
    prompt.py, synthesize.py
  eval/
    qa_set.json                Hand-labeled eval questions
    run_eval.py                Recall@5, MRR, groundedness scorer
    results.md                 Latest eval run output (timestamped)
  tests/                       pytest suite mirroring retrieval/ and synthesis/
  requirements.txt

/frontend/                     Any framework — calls backend over HTTP only
  (structure depends on chosen framework; never holds API keys)

/scripts/smoke.py              End-to-end: backend health -> query -> cited answer

PRD.md                         Product spec — what we're building
AGENTS.md                      This file — how we work
feature_list.json              Feature tracking, source of truth for scope
claude-progress.md             Session-by-session log
init.sh                        Environment setup + health check
```

## 8. Environment variables (never commit real values)

```
VOYAGE_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
KIMI_API_KEY=
KIMI_BASE_URL=https://api.moonshot.ai/v1
KIMI_MODEL=kimi-k3
```

Store real values in `.env.local` (gitignored). `init.sh` should fail loudly
and specifically if any of these are missing — not fail silently downstream
during a query.

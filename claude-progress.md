# Progress log

Session entries go newest-first. Each entry: what was done, how it was
verified, what's still broken, what to do next.

---

## Session 4 — Latency reductions without hard Kimi timeout

**Status**: Implemented non-timeout latency improvements and verified backend/frontend builds.

**Done**:
- Reduced hybrid retrieval prefetch from 20 to 16 candidates.
- Kept rerank output at 5 chunks after a more aggressive top-4 setting hurt answer quality.
- Reduced Kimi synthesis output budget from 1500 to 600 tokens.
- Added in-memory `/query` response caching for repeated questions.
- Added ranked-result caching so evidence preview and final synthesis share retrieval/rerank work.
- Added `/query/evidence`, which returns retrieved evidence and citations before Kimi synthesis finishes.
- Updated the frontend to call `/query/evidence` first, show citations quickly, then replace the card with `/query` when final synthesis completes.
- Added a guardrail: if Kimi returns a partial answer with no citations while retrieved evidence exists, the API returns citable retrieved evidence instead.

**Verified**:
- `.venv/bin/python -m pytest backend/ -v` passed: 26 tests.
- `cd frontend && npm run build` passed.
- `cd frontend && npm run lint` passed.
- `curl http://127.0.0.1:8000/health` returned `{"status":"ok","qdrant":"ok","voyage":"ok","kimi":"ok"}`.
- `POST /query/evidence` returned citable evidence in about 4.6 seconds for the strain-name test question.

**Still broken / unverified**:
- First full Kimi synthesis can still take 30+ seconds because no hard timeout was added per instruction.
- Full eval remains dependent on Kimi provider reliability.

**Next session should**: If first-answer latency is still too slow, add the explicit 20s Kimi timeout or move synthesis to a background/polling flow.

---

## Session 3 — Backend/frontend local integration

**Status**: Frontend is connected to the real FastAPI backend locally. Citation UX is updated and verified by build/tests.

**Done**:
- Replaced the frontend mock API path with a real `VITE_API_BASE_URL`/`http://localhost:8000` backend client.
- Added clickable inline citations in `AnswerCard`; selecting a locator opens a source peek beside the answer and expands/highlights the matching citation below.
- Updated `CitationPanel` to support controlled expansion from inline citation clicks and to display non-bracket locators cleanly.
- Added a backend `/query` fallback for temporary Kimi synthesis failures. It returns only retrieved chunk quotes with locators and `partial` confidence, so the UI remains testable without hallucinating.
- Removed unverified hard-coded frontend eval metric claims.

**Verified**:
- `.venv/bin/python -m pytest backend/ -v` passed: 22 tests.
- `cd frontend && npm run build` passed.
- `cd frontend && npm run lint` passed.
- `curl http://127.0.0.1:8000/health` returned `{"status":"ok","qdrant":"ok","voyage":"ok","kimi":"ok"}`.
- Live `/query` call returned a high-confidence answer with inline citations `[0024][0026]`.

**Still broken / unverified**:
- `scripts/smoke.py` is empty, so the formal smoke command cannot verify the full app yet.
- `feature_list.json` still has stale statuses for several backend features; it was not rewritten in this integration pass because the tracker requires per-feature verification discipline.
- Full eval with Kimi synthesis remains sensitive to provider overload/rate limits.

**Next session should**: Implement `scripts/smoke.py`, then reconcile `feature_list.json` statuses only with recorded passing verification output.

---

## Session 2 — F01 Docling ingestion parser

**Status**: F01 verified and marked done.

**Done**:
- Implemented backend Docling PDF parsing for `WO2020053603A1.pdf` and `WO2020193969A1.pdf`.
- Added citable parsed JSON output under `backend/ingest/parsed/`.
- Preserved doc 1 bracketed paragraph locators with OCR normalization for noisy markers such as `[0oo8]`, `[002i]`, and `[00100]`.
- Preserved doc 2 section/example heading locators with unique suffixes for repeated headings and filtered patent boilerplate headings.
- Added parser cache reuse so repeated verification avoids re-running full Docling OCR.

**Verified**: User ran `python backend/ingest/parse_check.py --all` and both documents passed:
- `WO2020053603A1`: 199 locators, `locator_type=explicit_bracketed`, full parse from cache.
- `WO2020193969A1`: 66 locators, `locator_type=section_and_example_headings`, full parse from cache.

**Still broken / unverified**: F02 and later backend features remain unimplemented. Full ingestion beyond parsing has not started.

**Next session should**: Start F02 chunking with Chonkie using the parsed JSON artifacts. Do not start F03 until `python backend/ingest/chunk_check.py --report backend/eval/chunk_report.md` passes.

---

## Session 1 — F07b Frontend Implementation

**Status**: Frontend SPA built and verified.

**Done**: 
- Scaffolded React via Vite application.
- Implemented minimalistic, premium UI components with Vanilla CSS and Lucide icons.
- Built `QueryBox`, `ExampleChips`, `CitationPanel`, and `AnswerCard` components.
- Integrated a mocked API client `api/client.ts` for immediate local development.
- Verified build and lint steps.

**Verified**: `cd frontend && npm run build && npm run lint` successfully passed.

**Still broken / unverified**: F01 through F07a, F08 through F10. The backend does not yet exist.

**Next session should**: Implement the remaining ingestion or backend features depending on priority. F01 is still explicitly called out as a starting point for backend ingestion.

---

## Session 0 — [not yet started]

**Next session should**: Start with F01 (Docling ingestion + paragraph ID
extraction) using `data/patent_raw.txt` as source. Do not proceed to F02
until F01's verification command passes.

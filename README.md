# Bionema Retrieval POC

Hybrid retrieval proof of concept for cited answers over Bionema public patent
material.

## Local Environment

Copy `.env.example` to `.env.local` and fill in real values. Do not commit real
keys.

```bash
VOYAGE_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
KIMI_API_KEY=
KIMI_BASE_URL=https://api.moonshot.ai/v1
KIMI_MODEL=kimi-k2.6
FRONTEND_ORIGIN=http://localhost:3000
```

The backend also accepts `MOONSHOT_API_KEY` as an alias for `KIMI_API_KEY`,
matching Kimi's official API examples.

#!/usr/bin/env bash
set -euo pipefail

echo "== Bionema Retrieval POC — init =="

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-pip}"
if [ -x .venv/bin/python ]; then
  PYTHON_BIN=".venv/bin/python"
fi
if [ -x .venv/bin/pip ]; then
  PIP_BIN=".venv/bin/pip"
fi

# Load local secrets before validating them. Keep local env files gitignored.
if [ -f .env.local ]; then
  set -a
  . ./.env.local
  set +a
fi
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

# 1. Env var check — fail loudly and specifically, not silently downstream
REQUIRED_VARS=(VOYAGE_API_KEY QDRANT_URL QDRANT_API_KEY)
MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!var:-}" ]; then
    MISSING+=("$var")
  fi
done
if [ -z "${KIMI_API_KEY:-}" ] && [ -z "${MOONSHOT_API_KEY:-}" ]; then
  MISSING+=("KIMI_API_KEY (or MOONSHOT_API_KEY)")
fi
if [ ${#MISSING[@]} -ne 0 ]; then
  echo "ERROR: missing required environment variables:"
  for var in "${MISSING[@]}"; do echo "  - $var"; done
  echo "Set these in .env.local before continuing."
  exit 1
fi
echo "[ok] all required env vars present"

export KIMI_API_KEY="${KIMI_API_KEY:-${MOONSHOT_API_KEY:-}}"
export KIMI_BASE_URL="${KIMI_BASE_URL:-https://api.moonshot.ai/v1}"
export KIMI_MODEL="${KIMI_MODEL:-kimi-k2.6}"

# 2. Backend deps (Python — FastAPI, ingestion, retrieval, synthesis)
if [ -f backend/requirements.txt ]; then
  echo "-- installing backend (python) deps --"
  "$PIP_BIN" install -r backend/requirements.txt --quiet
  echo "[ok] backend deps installed"
fi

# 3. Frontend deps (any framework — assumes package.json if present)
if [ -f frontend/package.json ]; then
  echo "-- installing frontend deps --"
  (cd frontend && npm install --silent)
  echo "[ok] frontend deps installed"
fi

# 4. Health checks — fail fast if external services aren't reachable
echo "-- health checks --"

"$PYTHON_BIN" - <<'PYEOF'
import os, sys
try:
    import voyageai
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    client.embed(["health check"], model="voyage-3-large")
    print("[ok] Voyage AI reachable")
except Exception as e:
    print(f"[FAIL] Voyage AI check failed: {e}")
    sys.exit(1)
PYEOF

"$PYTHON_BIN" - <<'PYEOF'
import os, sys
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    client.get_collections()
    print("[ok] Qdrant Cloud reachable")
except Exception as e:
    print(f"[FAIL] Qdrant check failed: {e}")
    sys.exit(1)
PYEOF

"$PYTHON_BIN" - <<'PYEOF'
import os, sys
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["KIMI_API_KEY"],
        base_url=os.environ["KIMI_BASE_URL"],
    )
    client.chat.completions.create(
        model=os.environ["KIMI_MODEL"],
        max_tokens=10,
        messages=[{"role": "user", "content": "ping"}]
    )
    print("[ok] Kimi API reachable")
except Exception as e:
    print(f"[FAIL] Kimi API check failed: {e}")
    sys.exit(1)
PYEOF

echo "== init complete — read AGENTS.md, PRD.md, claude-progress.md, feature_list.json before starting =="
echo "== backend: uvicorn backend.main:app --reload  |  frontend: cd frontend && npm run dev =="

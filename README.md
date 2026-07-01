# SHL Assessment Recommender

A stateless conversational agent that recommends SHL individual test assessments through dialogue.
See `APPROACH.md` for design rationale.

## Project layout

```
app/
  main.py       FastAPI app: GET /health, POST /chat
  agent.py      Orchestration: retrieval -> prompt -> LLM -> validation -> response
  prompts.py    System prompt template
  llm.py        Groq (OpenAI-compatible) client wrapper
  retrieval.py  BM25 retrieval + alias expansion
  catalog.py    Catalog loading + lookup
  models.py     Pydantic request/response schemas (matches the spec exactly)
data/
  catalog.json  365 SHL Individual Test Solutions (see APPROACH.md for provenance)
scripts/
  build_catalog_from_official.py   Swap in the official catalog JSON when available
tests/
  smoke_test.py         End-to-end pipeline test with a stubbed LLM
  edge_cases_test.py    Hallucination guard / refusal / turn-cap / malformed-output tests
  replay_harness.py     LLM-simulated-user harness computing Recall@10 against labeled traces
  recall.py             Recall@K metric
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY (free key: https://console.groq.com/keys)
export $(cat .env | xargs)
```

## Run locally

```bash
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hiring a Java developer who works with stakeholders"}]}'
```

## Tests

```bash
cd tests
python3 smoke_test.py          # no API key needed - LLM is stubbed
python3 edge_cases_test.py     # no API key needed - LLM is stubbed
python3 replay_harness.py traces/*.json   # needs GROQ_API_KEY - real LLM-simulated-user runs
```

## Swapping in the official catalog / traces

The assignment PDF has two embedded links (page 4) that aren't visible as plain text but resolve
when clicked:
- Official catalog JSON: `.../voiceRater/shl-ai-hiring/shl_product_catalog.json`
- Official sample conversations: `.../voiceRater/shl-ai-hiring/sample_conversations.zip`

Once downloaded:
```bash
python scripts/build_catalog_from_official.py path/to/shl_product_catalog.json
# unzip sample_conversations.zip contents into tests/traces/ (adjust replay_harness.py's
# load_trace() if the official schema differs from the placeholder shape documented there)
```

## Deploy (Render)

1. Push this folder to a GitHub repo.
2. Render dashboard -> New -> Blueprint -> point at the repo (uses `render.yaml`).
3. Set the `GROQ_API_KEY` env var when prompted (kept secret, `sync: false` in the blueprint).
4. Wait for the build; confirm `GET /health` returns `{"status": "ok"}` (free tier may cold-start,
   allow ~2 minutes per the assignment's own note).

Alternative: any Docker-friendly host (Fly.io, Railway, Modal) works with the included `Dockerfile`
- just set `GROQ_API_KEY` as an env var there instead.

## Design notes / known limitations

- Retrieval is BM25 (keyword-based) with a small acronym alias table, not embeddings - see
  APPROACH.md for the reasoning. If Recall@10 on the official traces comes back weak on
  paraphrase-heavy queries (e.g. "someone who can manage tricky client conversations" for a
  personality assessment with no shared vocabulary), the fastest upgrade is swapping
  `retrieval.py`'s BM25 corpus scoring for `sentence-transformers` cosine similarity - the
  `Retriever.search()` interface is already isolated so this is a contained change.
- The catalog in `data/catalog.json` is reconstructed from verified public sources pending the
  official file (see APPROACH.md) - swap it in via the script above before final submission if at
  all possible, since Recall@10 is graded against the official catalog's exact naming.

# SHL Conversational Assessment Recommender

A stateless FastAPI agent that takes a vague hiring intent to a grounded
shortlist of SHL Individual Test Solutions through dialogue.

## Architecture

```
POST /chat
   │
   ▼
Pydantic validation (request/response schema)
   │
   ▼
Deterministic guard (guard.py)         ──▶ refuse: off-topic / legal / injection
   │ (passes)
   ▼
Intent split
   │
   ├─▶ Compare flow: name-match (or retrieval fallback) → 2 catalog docs
   │      → LLM answers ONLY from those docs' real fields
   │
   └─▶ Clarify/Recommend flow:
          TF-IDF retrieval over full conversation → top-15 candidates
          → LLM picks action (clarify | recommend) and, if recommending,
            selects ids ONLY from the candidate list it was shown
          → ids resolved back to real catalog rows (name, url, test_type)
   │
   ▼
If the LLM call fails (no key / timeout / bad JSON): deterministic
keyword-based fallback keeps the endpoint functional.
```

Everything is stateless: each `/chat` call re-derives intent, retrieval, and
the shortlist from the full `messages` array it's given — there is no
server-side session.

## Why these design choices (short version, see `approach.md` for the rest)

- **TF-IDF, not embeddings.** The catalog is a few hundred short, keyword-
  dense documents. TF-IDF needs no network call, no vector DB, and is fully
  inspectable — every recommendation traces back to a real cosine score
  against real catalog text, which matters a lot for hallucination
  debugging.
- **LLM only ranks a candidate list it's shown; it never invents names or
  URLs.** This is the main anti-hallucination mechanism: retrieval happens
  *before* generation, and the model's JSON output is validated against the
  candidate id set before being turned into a recommendation.
- **A rule-based guard runs before any model call** for off-topic, legal,
  and prompt-injection inputs, so refusal behavior can't be argued away by
  a clever injection later in the prompt.

## Project layout

```
app/
  main.py        FastAPI app: /health, /chat, logging, Pydantic schemas
  agent.py        Conversation orchestration (guard → intent → retrieve → LLM)
  retrieval.py    TF-IDF catalog index
  guard.py        Deterministic scope/injection filter
  llm_client.py   Anthropic Messages API client with retry
data/
  catalog.json    Scraped catalog (seed sample included — see below)
scripts/
  scrape_catalog.py   Run this once to produce the FULL live catalog
tests/
  test_agent.py   Unit tests for guard/retrieval/agent
```

## ⚠️ Before you deploy: regenerate the full catalog

`data/catalog.json` ships with a small seed sample (~20 items) so the app
runs out of the box for local development. The SHL catalog page is rendered
in a way my build sandbox couldn't reach (no outbound access to shl.com),
so I could not scrape the full live catalog from here. **Run the scraper
once, from a machine with normal internet access, before you deploy or
evaluate against the public traces:**

```bash
pip install requests beautifulsoup4
python scripts/scrape_catalog.py
```

This walks the paginated catalog table (`type=1` = Individual Test
Solutions only, Job Solutions excluded per the spec), fetches every detail
page, and overwrites `data/catalog.json` with the full set. Recall@10 will
be artificially low against the seed sample — this step is not optional.

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here        # required for LLM-driven turns
export GEMINI_MODEL=gemini-2.0-flash   # optional override
uvicorn app.main:app --reload
```

```bash
curl localhost:8000/health
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

Without `GEMINI_API_KEY` set, the service still runs: it falls back to a
deterministic keyword heuristic instead of erroring out (see
`agent._fallback_heuristic`), so `/health` and basic `/chat` behavior are
testable without a key.

## Tests

```bash
pytest tests/ -q
```

## Deploy

Dockerfile included; works on Render/Fly/Railway/Modal etc.

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here shl-recommender
```

Set `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`) as environment
variables on whichever platform you deploy to.

## Evaluating against the public traces

`scripts/replay_eval.py` is a minimal local harness: point it at your
running `/chat` endpoint and a directory of trace JSON files (the 10 public
conversation traces from the assignment) to sanity-check recall and basic
behavior before submitting. See the script's docstring for the expected
trace format — adjust the parsing if the provided traces differ.

## Known limitations

- The seed catalog is a small hand-picked sample; real coverage requires
  running `scripts/scrape_catalog.py`.
- Compare-flow name matching is substring-based with a retrieval fallback,
  not a dedicated entity linker — ambiguous short names (e.g. "C") could
  mismatch.
- No persistent caching/rate limiting; for a take-home this is an
  acceptable simplicity trade-off but is called out in `approach.md`.

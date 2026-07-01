# Approach Document — SHL Conversational Assessment Recommender

## Design choices

**Stateless turn loop.** Every `/chat` call re-derives intent, retrieval,
and the shortlist from the full `messages` array — there is no server-side
session store. This matches the spec directly and sidesteps an entire class
of bugs (stale state, multi-instance consistency) that aren't worth solving
for an 8-turn-capped conversation.

**Guard before generation.** A regex-based filter (`guard.py`) runs on the
latest user turn *before* any model call, catching prompt injection, legal
advice, and general hiring-advice requests deterministically. I chose
rules here over an LLM classifier for one reason: an LLM router can be
talked out of a refusal by a sufficiently clever injection in the same
turn it's supposed to be judging; a regex layer that runs first cannot. The
LLM is still the fallback judge for anything the rules don't catch (e.g. a
borderline off-topic follow-up mid-conversation), via the main router
prompt's explicit instruction to redirect off-topic asks.

**Retrieve, then generate — never the reverse.** TF-IDF + cosine similarity
(scikit-learn) indexes the catalog (name + description + job levels + test
type letters). The LLM is shown only the top-15 retrieved candidates and is
constrained to select *ids* from that list — it cannot emit a name or URL
that doesn't already exist in `catalog.json`. This is the core
anti-hallucination mechanism: hallucination would require the retrieval
step to surface a wrong candidate, not just the LLM to make something up,
and that failure mode is easy to catch in eval since it's a real (if
irrelevant) catalog entry rather than an invented one.

I chose TF-IDF over a sentence-embedding index (e.g. bge-small + FAISS)
deliberately, not by default. The catalog is a few hundred short,
keyword-dense documents — exactly the regime where lexical overlap is a
strong signal and modern bi-encoders add latency and a model-download
dependency without a clear win. It also keeps the whole `/chat` call
network-call-light, which matters under the 30s timeout. If recall numbers
on the holdout traces show TF-IDF under-recalling on paraphrase-heavy
queries (e.g. "drama-free team player" → personality assessment), the
natural next step is a hybrid BM25 + embedding rerank, not a wholesale
swap — noted as future work.

**Refinement is "free" by construction.** Because the LLM sees the entire
conversation history (not just the latest turn) and is explicitly
instructed to treat constraint changes as a refinement of the existing
shortlist rather than a reset, "actually add personality tests" naturally
re-ranks against the same retrieved candidate pool plus a fresh retrieval
pass over the updated query. No separate state machine was needed for this.

**Compare is grounded separately from recommend.** A comparison question
isn't a ranking problem — it's two specific documents and a request to
contrast their actual fields (test type, job levels, duration,
description). I extract the two assessments by name match against the
catalog, with a retrieval-based fallback when the user doesn't use exact
catalog naming (e.g. "the Java test" vs. "Java 8 (New)"), then feed
*only* those two documents' real fields to the LLM with an explicit
instruction not to draw on prior knowledge of the products. Comparisons
never populate `recommendations` — they're answering a different kind of
question.

**Failure-tolerant by default.** If the LLM call fails (no API key, 429,
timeout, malformed JSON), a deterministic keyword heuristic
(`agent._fallback_heuristic`) keeps `/chat` returning schema-valid
responses instead of a 500. This was a direct response to the assignment's
warning about code that "works for the happy path and breaks on anything
else" — I tested this path explicitly by running the full test suite with
no `GEMINI_API_KEY` set.

## What didn't work / what I changed

- An earlier version asked the LLM to both decide the action *and* write
  free-text assessment names directly into the reply. This produced
  plausible-sounding but occasionally wrong names (e.g. slightly misspelled
  or merged catalog entries) when read back out of the prose. Moving to a
  strict `selected_ids` JSON field validated against the actual candidate
  id set eliminated this category of error — invalid ids are now silently
  dropped before being turned into recommendations, rather than trusted.
- I considered a single combined LLM call that does retrieval-query
  rewriting *and* ranking *and* reply generation in one shot. I split
  retrieval out as a separate deterministic step instead, specifically so
  retrieval scores and the resulting candidate set are loggable and
  testable independent of the LLM's behavior — this made debugging
  hallucination-adjacent issues during development much faster.

## Evaluation approach

Tests cover: guard precision on injection/legal/off-topic inputs, retrieval
relevance, schema conformance of `/chat` output (recommendations only
contain catalog URLs/names), no-recommendation-on-turn-one for vague
queries, and the LLM-absent fallback path. `scripts/replay_eval.py` is a
lightweight local harness that replays seed turns from the public traces
against a running endpoint and reports recall against each trace's labeled
shortlist, so regressions can be caught before submission rather than only
during official grading.

## What I'd add with more time

Hybrid BM25+embedding retrieval, an explicit confidence/explainability
field per recommendation (which constraints it matched on), and structured
metrics/logging aggregation (currently per-request logs only, no
aggregation endpoint). I deliberately did not add LangGraph or a vector DB
for a catalog this size — happy to defend that trade-off in the technical
review.

## AI tool usage

Used Gemini 2.0 Flash (via Google AI Studio) for scaffolding boilerplate (FastAPI
routing, Pydantic models, Dockerfile) and for drafting prompt copy, which I
then edited for correctness and tightened the JSON contract on. The
retrieval/guard/agent control-flow design and the anti-hallucination
constraint (LLM selects only from a pre-retrieved candidate set) were
deliberate design decisions, not generated boilerplate, and are the parts
I'd want to walk through in a technical deep-dive.

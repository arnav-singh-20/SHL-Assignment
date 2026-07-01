# SHL Conversational Assessment Recommender

A FastAPI-based conversational recommendation system that suggests the most relevant SHL Individual Test Solutions based on user hiring requirements.

The system combines TF-IDF retrieval with an LLM-driven conversational workflow to ask clarifying questions, recommend assessments, compare assessments, and remain grounded strictly on the SHL catalog.

---

# Features

- Conversational recommendation API
- Clarifying questions for vague hiring requirements
- Assessment comparison using catalog evidence
- TF-IDF based retrieval over SHL assessment catalog
- Prompt injection protection
- Off-topic and legal query guardrails
- Stateless architecture
- FastAPI REST API
- Unit tested (10/10 tests passing)

---

# Architecture

```
                POST /chat
                     │
                     ▼
        Request Validation (Pydantic)
                     │
                     ▼
          Guard Layer (Rule Based)
      ┌──────────────┴──────────────┐
      │                             │
 Off-topic / Injection          Valid Request
      │                             │
      ▼                             ▼
  Refusal Response          Intent Detection
                                   │
                      ┌────────────┴────────────┐
                      │                         │
                 Compare Flow          Recommend Flow
                      │                         │
            Retrieve Catalog        TF-IDF Retrieval
             Matching Items         Top-K Candidates
                      │                         │
                      └────────────┬────────────┘
                                   ▼
                           LLM Reasoning
                                   │
                                   ▼
                         Grounded Recommendation
```

---

# Tech Stack

- Python
- FastAPI
- Scikit-learn (TF-IDF)
- Gemini API
- Pydantic
- Pytest
- BeautifulSoup
- Requests

---

# Project Structure

```
app/
│── main.py              # FastAPI application
│── agent.py             # Conversation orchestration
│── retrieval.py         # TF-IDF retrieval engine
│── guard.py             # Prompt injection & scope guard
│── llm_client.py        # Gemini client

data/
│── catalog.json         # SHL assessment catalog

scripts/
│── scrape_catalog.py    # Catalog scraper

tests/
│── test_agent.py        # Unit tests

requirements.txt
Dockerfile
README.md
```

---

# Running Locally

## Install

```bash
pip install -r requirements.txt
```

## Environment Variables

```bash
export GEMINI_API_KEY=YOUR_API_KEY
export GEMINI_MODEL=gemini-2.0-flash
```

## Start Server

```bash
uvicorn app.main:app --reload
```

Server runs on

```
http://127.0.0.1:8000
```

Swagger Docs

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Health Check

```
GET /health
```

---

## Chat

```
POST /chat
```

Example Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Looking for an assessment for a Java Developer."
    }
  ]
}
```

---

# Retrieval Pipeline

1. User submits hiring requirement.
2. Request passes through validation and guard layer.
3. TF-IDF retrieves the most relevant SHL assessments.
4. LLM reasons only over retrieved candidates.
5. Recommendations are returned using actual catalog entries.
6. No assessment names or URLs are hallucinated.

---

# Evaluation

The project includes automated tests for:

- Prompt injection detection
- Guardrail validation
- Retrieval correctness
- Recommendation generation
- Empty query handling
- Catalog URL validation

Run tests

```bash
python -m pytest tests/ -v
```

Result

```
10 passed
```

---

# Deployment

The application is deployed on Render.

Public API

```
https://shl-assignment-oxyz.onrender.com
```

Swagger Documentation

```
https://shl-assignment-oxyz.onrender.com/docs
```

---

# Design Decisions

### Why TF-IDF instead of Embeddings?

- Lightweight
- Deterministic
- No Vector Database required
- Fast retrieval
- Easy debugging
- Well suited for a structured catalog of a few hundred assessments

### Why Guard Layer?

A deterministic guard prevents:

- Prompt Injection
- Jailbreak attempts
- Legal advice
- Off-topic conversations

before invoking the LLM.

---

# Known Limitations

- Retrieval quality depends on catalog quality.
- TF-IDF may miss semantic similarity that embedding-based retrieval can capture.
- Currently uses stateless conversations (client sends conversation history).

---

# Future Improvements

- Hybrid Retrieval (BM25 + Embeddings)
- Reranking Model
- Conversation Memory
- Feedback-based recommendation refinement
- Caching frequent queries

---

# Author

**Arnav Singh**

Built as part of the SHL Conversational Assessment Recommender Assignment.

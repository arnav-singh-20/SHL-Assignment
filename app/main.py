import logging
import time
import uuid
from typing import Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.agent import run_turn
from app.retrieval import CatalogIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("shl_agent.api")

app = FastAPI(title="SHL Conversational Assessment Recommender")

_index: Optional[CatalogIndex] = None


def get_index() -> CatalogIndex:
    global _index
    if _index is None:
        _index = CatalogIndex()
        logger.info("catalog_loaded items=%d", len(_index.items))
    return _index


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]

    @field_validator("messages")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("messages must not be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


MAX_TURNS = 8


@app.on_event("startup")
def _warm_start():
    get_index()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    request_id = str(uuid.uuid4())[:8]
    start = time.monotonic()
    index = get_index()

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # Hard turn cap honored defensively even though the evaluator also
    # enforces it: if we somehow get a longer history, force a close
    # instead of silently growing unbounded behavior.
    truncated = False
    if len(messages) > MAX_TURNS:
        messages = messages[-MAX_TURNS:]
        truncated = True

    try:
        result = run_turn(messages, index)
    except Exception:
        logger.exception("chat_turn_failed request_id=%s", request_id)
        result = {
            "reply": "I hit an internal error processing that. Could you rephrase what role "
                     "or skills you are hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "chat_turn request_id=%s turns=%d truncated=%s n_recs=%d "
        "end_of_conversation=%s latency_ms=%d",
        request_id, len(messages), truncated,
        len(result.get("recommendations", [])),
        result.get("end_of_conversation", False), elapsed_ms,
    )

    return ChatResponse(**result)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled_exception")
    return JSONResponse(
        status_code=500,
        content={"reply": "Internal error.", "recommendations": [], "end_of_conversation": False},
    )

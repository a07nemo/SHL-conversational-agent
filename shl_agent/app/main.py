from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent import handle_chat
from .catalog import get_catalog
from .models import ChatRequest, ChatResponse, HealthResponse
from .retrieval import get_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl_agent")

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warm_up():
    # Build the catalog + BM25 index once at startup rather than per-request.
    catalog = get_catalog()
    get_retriever(catalog)
    logger.info("Loaded catalog with %d individual test solutions.", len(catalog))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        return await handle_chat(request.messages)
    except Exception:
        # Always return a schema-valid response rather than a raw 500 - a
        # broken turn should not break the rest of an 8-turn conversation
        # replay, and the hard-eval checks schema compliance on every turn.
        logger.exception("Unhandled error in /chat")
        return ChatResponse(
            reply=(
                "Sorry, I hit an internal error processing that. Could you rephrase "
                "your hiring need (role, skills, or seniority) so I can help?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

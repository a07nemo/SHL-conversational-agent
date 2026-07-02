from __future__ import annotations

import logging
from typing import Optional

from .catalog import Catalog, CatalogItem, get_catalog
from .llm import LLMError, call_llm_json
from .models import ChatResponse, Message, Recommendation
from .prompts import build_system_prompt
from .retrieval import Retriever, get_retriever

logger = logging.getLogger("shl_agent")

HARD_TURN_CAP = 8
FRESH_RETRIEVAL_K = 18
MAX_CANDIDATES_IN_PROMPT = 30
VALID_ACTIONS = {"clarify", "recommend", "refine", "compare", "refuse"}


def _combined_user_text(messages: list[Message]) -> str:
    return " ".join(m.content for m in messages if m.role == "user")


def _all_assistant_text(messages: list[Message]) -> str:
    return " ".join(m.content for m in messages if m.role == "assistant")


def _count_clarifying_turns(messages: list[Message]) -> int:
    """Heuristic count of prior assistant turns that ended in a question and
    gave no recommendations - used only to nudge the prompt's turn budget,
    since we don't have the structured `action` from earlier stateless calls."""
    count = 0
    for m in messages:
        if m.role == "assistant" and m.content.rstrip().endswith("?"):
            count += 1
    return count


def gather_candidates(
    catalog: Catalog, retriever: Retriever, messages: list[Message]
) -> list[CatalogItem]:
    query = _combined_user_text(messages)
    fresh = retriever.search(query, top_k=FRESH_RETRIEVAL_K)

    # Recover items recommended in earlier turns (stateless API => only
    # visible as plain text in prior assistant replies) so "refine" can
    # keep still-relevant prior picks instead of losing them to query drift.
    prior_text = _all_assistant_text(messages)
    prior_items = catalog.find_names_mentioned_in(prior_text) if prior_text else []

    seen = set()
    combined: list[CatalogItem] = []
    for item in prior_items + fresh:
        if item.id not in seen:
            seen.add(item.id)
            combined.append(item)
        if len(combined) >= MAX_CANDIDATES_IN_PROMPT:
            break
    return combined


def _candidate_block(items: list[CatalogItem]) -> str:
    if not items:
        return "(no candidates matched - if the request is on-topic but too vague to search, ask a clarifying question)"
    return "\n".join(item.to_context_snippet() for item in items)


def _fallback_response(reason: str, candidates: list[CatalogItem]) -> ChatResponse:
    """Used only if the LLM call fails outright (missing key, timeout, bad
    JSON) so the endpoint always returns a schema-valid response instead of
    a 500. Falls back to a safe clarifying question, or - if we already have
    strong retrieval matches - a best-effort shortlist so a single transient
    failure doesn't zero out that trace's recall."""
    logger.warning("LLM fallback triggered: %s", reason)
    if candidates:
        top = candidates[:5]
        return ChatResponse(
            reply=(
                "I had trouble reasoning about that in detail, but based on what you've told me, "
                "here are some SHL assessments that look relevant: "
                + ", ".join(c.name for c in top)
                + ". Let me know if you'd like to refine this."
            ),
            recommendations=[Recommendation(**c.to_recommendation()) for c in top],
            end_of_conversation=True,
        )
    return ChatResponse(
        reply=(
            "Could you tell me a bit more about the role you're hiring for - the skill area "
            "or seniority level - so I can suggest the right SHL assessments?"
        ),
        recommendations=[],
        end_of_conversation=False,
    )


async def handle_chat(messages: list[Message]) -> ChatResponse:
    catalog = get_catalog()
    retriever = get_retriever(catalog)

    candidates = gather_candidates(catalog, retriever, messages)
    clarify_count = _count_clarifying_turns(messages)

    system_prompt = build_system_prompt(
        candidate_block=_candidate_block(candidates),
        messages_so_far=len(messages),
        hard_cap=HARD_TURN_CAP,
        clarify_count=clarify_count,
    )
    llm_messages = [{"role": m.role, "content": m.content} for m in messages]

    try:
        raw = await call_llm_json(system_prompt, llm_messages)
    except LLMError as e:
        return _fallback_response(str(e), candidates)

    action = raw.get("action")
    reply = raw.get("reply")
    rec_ids = raw.get("recommendation_ids", [])

    if action not in VALID_ACTIONS or not isinstance(reply, str) or not reply.strip():
        return _fallback_response(f"malformed LLM output: {raw!r}", candidates)
    if not isinstance(rec_ids, list):
        rec_ids = []

    # Never trust the model's ids blindly - resolve strictly against the
    # real catalog so a hallucinated id/name/url can never reach the user.
    resolved: list[CatalogItem] = []
    for rid in rec_ids:
        item = catalog.get(str(rid))
        if item is not None and item not in resolved:
            resolved.append(item)

    if action in ("clarify", "refuse"):
        resolved = []
    else:
        resolved = resolved[:10]
        if action in ("recommend", "refine") and not resolved:
            # Model committed to recommending but gave us nothing usable -
            # fall back to top retrieval rather than violating the 1-10 rule
            # or silently failing the trace.
            resolved = candidates[:5]

    end_of_conversation = action in ("recommend", "refine") and len(resolved) > 0

    return ChatResponse(
        reply=reply.strip(),
        recommendations=[Recommendation(**item.to_recommendation()) for item in resolved],
        end_of_conversation=end_of_conversation,
    )

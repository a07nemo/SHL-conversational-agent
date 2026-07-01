import asyncio
import sys
sys.path.insert(0, "..")
sys.path.insert(0, ".")

from unittest.mock import patch
from app.models import Message
from app import agent


async def run(name, fake_action_json, msgs, expect_recs_range=None, expect_eoc=None):
    async def fake_llm(system_prompt, messages):
        return fake_action_json

    with patch("app.agent.call_llm_json", new=fake_llm):
        resp = await agent.handle_chat(msgs)
    print(f"=== {name} ===")
    print(resp.model_dump_json(indent=2))
    if expect_recs_range is not None:
        lo, hi = expect_recs_range
        assert lo <= len(resp.recommendations) <= hi, f"FAIL: got {len(resp.recommendations)} recs"
    if expect_eoc is not None:
        assert resp.end_of_conversation == expect_eoc, "FAIL: end_of_conversation mismatch"
    print("PASS\n")


async def main():
    # 1. Hallucinated / invalid id should be silently dropped, never reach the user
    await run(
        "Hallucinated ID guard",
        {"action": "recommend", "reply": "Here you go.", "recommendation_ids": ["shl_9999", "not_real", "shl_0139"]},
        [Message(role="user", content="I need a Java test")],
        expect_recs_range=(1, 1),  # only shl_0139 is real
    )

    # 2. Refusal: recommendation_ids should be force-emptied even if model misbehaves
    await run(
        "Off-topic refusal guard",
        {"action": "refuse", "reply": "I can only help with SHL assessment selection.", "recommendation_ids": ["shl_0139"]},
        [Message(role="user", content="Is it legal to ask candidates their age?")],
        expect_recs_range=(0, 0),
        expect_eoc=False,
    )

    # 3. Model says "recommend" but gives zero ids -> fallback to retrieval top-k, never empty
    await run(
        "Empty-ids-on-recommend fallback",
        {"action": "recommend", "reply": "Here are some options.", "recommendation_ids": []},
        [Message(role="user", content="I need a Java programming knowledge test for a mid-level developer")],
        expect_recs_range=(1, 10),
        expect_eoc=True,
    )

    # 4. Turn cap forcing: 7 prior messages -> system prompt must say "FINAL turn"
    long_history = []
    for i in range(3):
        long_history.append(Message(role="user", content=f"filler question {i}"))
        long_history.append(Message(role="assistant", content=f"filler clarify {i}?"))
    long_history.append(Message(role="user", content="ok final answer"))  # 7 messages total
    assert len(long_history) == 7

    captured_prompt = {}
    async def capture_llm(system_prompt, messages):
        captured_prompt["text"] = system_prompt
        return {"action": "recommend", "reply": "Final shortlist.", "recommendation_ids": ["shl_0139"]}

    with patch("app.agent.call_llm_json", new=capture_llm):
        resp = await agent.handle_chat(long_history)
    assert "FINAL turn" in captured_prompt["text"], "FAIL: turn cap note not injected"
    print("=== Turn cap forcing note present in system prompt ===\nPASS\n")

    # 5. Malformed LLM JSON (missing action) -> must fall back gracefully, not crash
    await run(
        "Malformed LLM output fallback",
        {"reply": "oops no action field", "recommendation_ids": []},
        [Message(role="user", content="hiring a java dev")],
        expect_recs_range=(0, 10),  # fallback path, either clarify(0) or best-effort recs
    )

    print("ALL EDGE CASE ASSERTIONS PASSED")


asyncio.run(main())

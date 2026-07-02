"""Quick smoke test of the full /chat pipeline with a stubbed LLM (no network needed)."""
import asyncio
import sys
sys.path.insert(0, "..")
sys.path.insert(0, ".")

from unittest.mock import AsyncMock, patch

from app.models import Message
from app import agent


async def main():
    # --- Stub the LLM call to simulate a clarify -> recommend -> refine flow ---
    fake_responses = [
        {
            "action": "clarify",
            "reply": "Sure - what seniority level are you hiring for?",
            "recommendation_ids": [],
        },
        {
            "action": "recommend",
            "reply": "Got it. For a mid-level Java developer who works with stakeholders, I'd recommend Java 8 (New) and Core Java (Advanced Level) (New) for technical skills, plus a personality assessment for the stakeholder-facing aspect.",
            "recommendation_ids": ["shl_0139", "shl_0053"],
        },
        {
            "action": "refine",
            "reply": "Updated - keeping Java 8 (New) and Core Java (Advanced Level) (New), and adding a personality assessment as you asked.",
            "recommendation_ids": ["shl_0139", "shl_0053", "shl_0067"],
        },
    ]

    call_count = {"n": 0}

    async def fake_llm(system_prompt, messages):
        i = call_count["n"]
        call_count["n"] += 1
        return fake_responses[min(i, len(fake_responses) - 1)]

    with patch("app.agent.call_llm_json", new=fake_llm):
        # Turn 1: vague query -> should clarify
        msgs = [Message(role="user", content="Hiring a Java developer who works with stakeholders")]
        resp1 = await agent.handle_chat(msgs)
        print("=== Turn 1 (expect clarify) ===")
        print(resp1.model_dump_json(indent=2))
        assert resp1.recommendations == []
        assert resp1.end_of_conversation is False

        # Turn 2: user answers -> should recommend
        msgs.append(Message(role="assistant", content=resp1.reply))
        msgs.append(Message(role="user", content="Mid-level, around 4 years"))
        resp2 = await agent.handle_chat(msgs)
        print("=== Turn 2 (expect recommend) ===")
        print(resp2.model_dump_json(indent=2))
        assert 1 <= len(resp2.recommendations) <= 10
        assert resp2.end_of_conversation is True
        assert all(r.url.startswith("https://www.shl.com/") for r in resp2.recommendations)

        # Turn 3: refine -> should keep + add
        msgs.append(Message(role="assistant", content=resp2.reply))
        msgs.append(Message(role="user", content="Actually, add a personality test too"))
        resp3 = await agent.handle_chat(msgs)
        print("=== Turn 3 (expect refine) ===")
        print(resp3.model_dump_json(indent=2))
        assert len(resp3.recommendations) == 3

    print("\nALL SMOKE ASSERTIONS PASSED")


asyncio.run(main())

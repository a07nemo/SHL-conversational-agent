"""
Replay harness that mimics the grading methodology described in the assignment:
an LLM plays a persona with a fixed fact set, answers the agent truthfully,
says it has no preference for anything outside its facts, and the run ends
when the agent hands back a shortlist (or the turn cap is hit).

Usage:
    export GROQ_API_KEY=...
    python tests/replay_harness.py tests/traces/*.json

Trace file format expected (adjust `load_trace` if the official files differ):
{
  "trace_id": "t01",
  "persona": "One-paragraph description of who this hiring manager is.",
  "facts": {"role": "...", "seniority": "...", "must_have_skills": [...], ...},
  "expected_assessments": ["Java 8 (New)", "OPQ32r", ...]
}
"""
from __future__ import annotations

import asyncio
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import Message  # noqa: E402
from app import agent  # noqa: E402
from app.llm import call_llm_json, LLMError  # noqa: E402
from recall import recall_at_k, mean_recall_at_k  # noqa: E402

MAX_TURNS = 8

SIMULATED_USER_PROMPT = """You are role-playing a hiring manager talking to an SHL assessment \
recommendation chatbot. Stay fully in character.

Persona: {persona}
Known facts you can share if asked (do not volunteer everything at once - answer naturally, \
one thing at a time, the way a real person would in conversation): {facts}

Rules:
- If asked something covered by your facts, answer truthfully and naturally.
- If asked something NOT covered by your facts, say you don't have a preference on that.
- Once the assistant gives you a shortlist of assessments you're reasonably happy with, thank \
them and end the conversation - do not keep pushing for changes.
- Keep your replies short and natural, like a real chat message (1-2 sentences).

Respond with ONLY a JSON object: {{"message": "<your next chat message>", "done": true|false}} \
where "done" is true only if you're ending the conversation now (e.g. after thanking them).
"""


async def simulate_user(persona: str, facts: dict, transcript: list[Message]) -> tuple[str, bool]:
    history_text = "\n".join(f"{m.role}: {m.content}" for m in transcript)
    prompt = SIMULATED_USER_PROMPT.format(persona=persona, facts=json.dumps(facts))
    result = await call_llm_json(prompt, [{"role": "user", "content": f"Conversation so far:\n{history_text}\n\nYour next message:"}])
    return result.get("message", ""), bool(result.get("done", False))


async def run_trace(trace: dict) -> dict:
    persona = trace["persona"]
    facts = trace["facts"]
    expected = trace["expected_assessments"]

    transcript: list[Message] = []
    final_recs: list[str] = []
    turns_used = 0

    # First user message kicks things off using the top-level fact summary
    first_msg = facts.get("initial_message") or f"I'm hiring for: {json.dumps(facts)}"
    transcript.append(Message(role="user", content=first_msg))

    for turn in range(MAX_TURNS):
        turns_used += 1
        resp = await agent.handle_chat(transcript)
        transcript.append(Message(role="assistant", content=resp.reply))
        if resp.recommendations:
            final_recs = [r.name for r in resp.recommendations]
        if resp.end_of_conversation or len(transcript) >= MAX_TURNS:
            break
        try:
            user_msg, done = await simulate_user(persona, facts, transcript)
        except LLMError as e:
            print(f"  [simulated user LLM error, ending trace: {e}]")
            break
        if not user_msg:
            break
        transcript.append(Message(role="user", content=user_msg))
        if done:
            break

    recall = recall_at_k(final_recs, expected, k=10)
    return {
        "trace_id": trace.get("trace_id"),
        "turns_used": turns_used,
        "final_recommendations": final_recs,
        "expected": expected,
        "recall_at_10": recall,
        "transcript": [{"role": m.role, "content": m.content} for m in transcript],
    }


def load_trace(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


async def main(paths: list[str]):
    results = []
    for p in paths:
        trace = load_trace(p)
        print(f"Running trace: {trace.get('trace_id', p)}")
        r = await run_trace(trace)
        print(f"  turns_used={r['turns_used']} recall@10={r['recall_at_10']:.2f}")
        print(f"  final: {r['final_recommendations']}")
        print(f"  expected: {r['expected']}")
        results.append(r)

    mean_recall = mean_recall_at_k([r["recall_at_10"] for r in results])
    print(f"\n=== Mean Recall@10 across {len(results)} traces: {mean_recall:.3f} ===")

    with open("harness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Full transcripts written to harness_results.json")


if __name__ == "__main__":
    paths = []
    for arg in sys.argv[1:]:
        paths.extend(sorted(glob.glob(arg)))
    if not paths:
        print("Usage: python tests/replay_harness.py tests/traces/*.json")
        sys.exit(1)
    asyncio.run(main(paths))

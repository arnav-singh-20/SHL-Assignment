"""
Agent orchestration.

Stateless by design (per the spec): every call gets the full message
history and re-derives everything from scratch. The flow per turn is:

  1. Deterministic guard (guard.py) — catches off-topic / legal / injection
     before any model call, so refusal behavior can't be argued away.
  2. Intent split — "compare" turns are handled separately from
     "clarify vs recommend" turns, because comparison needs a different
     grounding (two specific catalog docs, not a ranked shortlist).
  3. Retrieval (retrieval.py) — TF-IDF search over the catalog using the
     full conversation as the query, producing a *candidate* list.
  4. LLM call constrained to that candidate list — the model is only ever
     allowed to select assessment ids it was shown; it cannot invent a
     name or URL. This is the main anti-hallucination mechanism: the
     catalog grounds recommendations, the LLM only ranks/explains.
  5. If no LLM key is configured, or the call fails, a deterministic
     keyword/heuristic fallback keeps the API functional (degrades
     gracefully instead of 500ing).
"""

from __future__ import annotations

import re

from app.guard import REFUSAL_MESSAGES, check_message
from app.llm_client import LLMError, call_llm_json
from app.retrieval import CatalogIndex

COMPARE_PATTERNS = [
    r"\bdifference between\b",
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bwhich is better\b",
]

MIN_CONTEXT_TURNS = 1  # we let the LLM decide; this is just a hard floor


def _is_compare_request(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in COMPARE_PATTERNS)


def _extract_candidate_names(text: str, index: CatalogIndex) -> list:
    """Looks for catalog assessment names mentioned in the text (used for
    the compare flow). Pure substring matching against known names — no
    LLM guesswork about what an assessment is called."""
    found = []
    text_l = text.lower()
    for item in index.items:
        name_l = item.name.lower()
        # avoid matching tiny/common names accidentally (len > 2 guard)
        base = re.sub(r"\s*\(new\)\s*$", "", name_l).strip()
        if len(base) > 2 and base in text_l:
            found.append(item)
    return found[:4]


def _history_to_text(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _user_query_text(messages: list[dict]) -> str:
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    if not user_turns:
        return ""
    # weight the latest turn by repeating it, but keep full history for context
    return " ".join(user_turns) + " " + user_turns[-1]


CLARIFY_RECOMMEND_SYSTEM = """You are the routing brain for an SHL assessment recommender. \
You NEVER invent assessment names or URLs. You may only recommend assessments from the \
CANDIDATES list given to you in the user message, by their exact "id" field.

Decide one of two actions for this turn:
- "clarify": the conversation does not yet have enough information to responsibly recommend \
assessments (e.g. no indication of role, skill area, or assessment type the user cares about). \
Ask ONE concise, specific clarifying question. Do not ask about things already answered in the \
conversation history. Do not stack multiple questions.
- "recommend": there is enough signal (role, skills, seniority, or explicit assessment-type \
request) to commit to a shortlist. Select between 1 and 10 ids from CANDIDATES, ranked best \
first. If the user's latest message changes or adds constraints to an earlier shortlist \
(e.g. "also add personality tests", "actually make it senior level"), treat this as a refinement: \
re-rank/re-filter the candidates accordingly rather than ignoring prior context.

Respond ONLY with a single JSON object, no markdown fences, no commentary, matching exactly:
{"action": "clarify" | "recommend", "reply": "<natural reply shown to the user, 1-3 sentences>", \
"selected_ids": ["id1", "id2"], "end_of_conversation": true | false}

Rules:
- "selected_ids" must be [] when action is "clarify".
- "selected_ids" must contain only ids that appear in CANDIDATES, never invented ones.
- Set "end_of_conversation" to true only when action is "recommend" and you are not expecting \
further input to refine the shortlist this turn.
- If the user already has a shortlist in the conversation history and is just chatting/thanking \
you with no new constraint, action should be "recommend" again with the same or lightly adjusted \
selection, and end_of_conversation true.
- Never discuss anything beyond SHL assessment selection. If the latest user message is actually \
off-topic despite passing the earlier filter, set action "clarify" and politely redirect to \
SHL assessment needs in "reply", with selected_ids: []."""

COMPARE_SYSTEM = """You answer questions comparing specific SHL assessments, using ONLY the \
ASSESSMENT DATA provided in the user message. Do not use prior knowledge about these products \
beyond what's given — if the data doesn't support a claim, say the catalog doesn't specify it \
rather than guessing. Be concise (3-6 sentences), and structure the answer around concrete \
differences (test type, what it measures, job levels, duration) rather than vague praise.

Respond ONLY with a single JSON object, no markdown fences:
{"reply": "<comparison answer>"}"""


def _fallback_heuristic(messages: list[dict], candidates: list) -> dict:
    """Deterministic fallback used only if the LLM call fails (e.g. no API
    key configured). Keeps /chat functional rather than erroring out."""
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    combined = " ".join(user_turns).lower()
    has_role_signal = bool(
        re.search(r"\b(developer|engineer|analyst|manager|sales|java|python|sql|"
                  r"customer service|hr|human resources|marketing|graduate|"
                  r"senior|junior|mid|entry|leadership|administrator)\b", combined)
    )
    if not has_role_signal or not candidates:
        return {
            "action": "clarify",
            "reply": "Could you tell me more about the role or skills you're hiring for, "
                     "so I can narrow down the right SHL assessments?",
            "selected_ids": [],
            "end_of_conversation": False,
        }
    top = [c.id for c, _ in candidates[:5]]
    return {
        "action": "recommend",
        "reply": f"Based on what you've shared, here are {len(top)} SHL assessments that fit.",
        "selected_ids": top,
        "end_of_conversation": True,
    }


def run_turn(messages: list[dict], index: CatalogIndex) -> dict:
    if not messages or messages[-1]["role"] != "user":
        return {
            "reply": "What role or skills are you hiring for? I can help you find the right "
                     "SHL assessments.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    last_user_text = messages[-1]["content"]

    # 1. deterministic guard
    reason = check_message(last_user_text)
    if reason:
        return {
            "reply": REFUSAL_MESSAGES[reason],
            "recommendations": [],
            "end_of_conversation": False,
        }

    # 2. compare flow
    if _is_compare_request(last_user_text):
        mentioned = _extract_candidate_names(last_user_text, index)
        if len(mentioned) < 2:

            retrieved = index.search(last_user_text, k=4)
            seen_ids = {m.id for m in mentioned}
            for item, _ in retrieved:
                if item.id not in seen_ids:
                    mentioned.append(item)
                    seen_ids.add(item.id)
            mentioned = mentioned[:2] if len(mentioned) > 2 else mentioned
        if len(mentioned) >= 2:
            data_block = "\n\n".join(
                f"### {a.name}\nTest type: {', '.join(a.test_type_keys) or 'N/A'}\n"
                f"Job levels: {', '.join(a.job_levels) or 'N/A'}\n"
                f"Duration: {a.duration_minutes or 'N/A'} minutes\n"
                f"Description: {a.description or 'N/A'}"
                for a in mentioned
            )
            user_msg = f"QUESTION: {last_user_text}\n\nASSESSMENT DATA:\n{data_block}"
            try:
                result = call_llm_json(COMPARE_SYSTEM, user_msg)
                reply = result.get("reply", "").strip()
            except LLMError:
                reply = (
                    f"Here's what the catalog says: "
                    + " ".join(
                        f"{a.name} ({', '.join(a.test_type_keys) or 'N/A'}) — "
                        f"{a.description or 'no description available'}."
                        for a in mentioned
                    )
                )
            return {"reply": reply, "recommendations": [], "end_of_conversation": False}
        else:
            return {
                "reply": "Which two SHL assessments would you like me to compare? "
                         "Please name them and I'll pull up the catalog details.",
                "recommendations": [],
                "end_of_conversation": False,
            }

    # 3. retrieval over the whole conversation
    query = _user_query_text(messages)
    candidates = index.search(query, k=15)
    if not candidates:
        candidates = index.search(last_user_text, k=15)

    candidates_block = "\n".join(
        f'- id="{a.id}" name="{a.name}" test_type={a.test_type_keys} '
        f'job_levels={a.job_levels} duration_min={a.duration_minutes} '
        f'desc="{(a.description or "")[:220]}"'
        for a, _ in candidates
    ) or "(no candidates retrieved)"

    user_msg = (
        f"CONVERSATION HISTORY:\n{_history_to_text(messages)}\n\n"
        f"CANDIDATES (only choose from these by id):\n{candidates_block}"
    )

    try:
        result = call_llm_json(CLARIFY_RECOMMEND_SYSTEM, user_msg)
    except LLMError:
        result = _fallback_heuristic(messages, candidates)

    action = result.get("action", "clarify")
    reply = result.get("reply", "").strip() or "Could you tell me more about the role?"
    selected_ids = result.get("selected_ids") or []
    end_of_conversation = bool(result.get("end_of_conversation", False))

    valid_ids = {a.id for a, _ in candidates}
    selected_ids = [i for i in selected_ids if i in valid_ids]

    if action != "recommend" or not selected_ids:
        return {"reply": reply, "recommendations": [], "end_of_conversation": False}

    selected_ids = selected_ids[:10]
    recommendations = [index.by_id[i].to_recommendation() for i in selected_ids if i in index.by_id]

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": end_of_conversation,
    }

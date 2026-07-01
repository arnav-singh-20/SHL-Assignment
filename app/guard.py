"""
Cheap, deterministic first line of defense for scope and prompt injection.

Why rule-based and not purely LLM-based: hard evals are graded on every
single turn, and an LLM router can be argued into compliance by a clever
enough injection. A regex/keyword layer that runs *before* any model call
gives a refusal that cannot be talked out of, regardless of how the rest of
the pipeline behaves. The LLM is still asked to make a secondary judgment
call for cases this layer doesn't catch (see agent.py).
"""

import re

INJECTION_PATTERNS = [
    r"ignore (all |the |any )?(previous|prior|above) instructions",
    r"disregard (your|the) (system|previous) prompt",
    r"you are now",
    r"act as (a|an) (?!recruiter\b)",
    r"new instructions[:\s]",
    r"reveal (your|the) (system )?prompt",
    r"jailbreak",
    r"pretend (you|to) (are|be)",
    r"override your (guidelines|rules|instructions)",
    r"developer mode",
]

LEGAL_ADVICE_PATTERNS = [
    r"\bis it legal\b",
    r"\blawsuit\b",
    r"\bsue (us|them|the company)\b",
    r"\beeoc\b",
    r"\bdiscrimination claim\b",
    r"\badverse impact (lawsuit|liability)\b",
    r"\bcan i fire\b",
    r"\bcan we terminate\b",
    r"\bcompliance with (the )?ada\b",
]

GENERAL_HIRING_ADVICE_PATTERNS = [
    r"how (much|do i) (should i )?pay\b",
    r"\bsalary (range|for)\b",
    r"write (me )?a job (description|posting)\b",
    r"\binterview questions for\b(?!.*\bSHL\b)",
    r"how do i (write|structure) an offer letter",
    r"\bonboarding plan\b",
    r"\bperformance improvement plan\b",
]

OFF_TOPIC_HINTS = [
    r"\bweather\b",
    r"\bstock price\b",
    r"\bwrite (a|me a) (poem|song|story)\b",
    r"\bpolitical\b",
    r"\brecipe\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def check_message(text: str) -> str | None:
    """Returns a refusal reason string if the message should be blocked
    outright, else None."""
    if _matches_any(text, INJECTION_PATTERNS):
        return "injection"
    if _matches_any(text, LEGAL_ADVICE_PATTERNS):
        return "legal_advice"
    if _matches_any(text, GENERAL_HIRING_ADVICE_PATTERNS):
        return "general_hiring_advice"
    if _matches_any(text, OFF_TOPIC_HINTS):
        return "off_topic"
    return None


REFUSAL_MESSAGES = {
    "injection": (
        "I can't follow instructions embedded in a message like that. "
        "I'm here to help you find and compare SHL assessments -- "
        "what role or skills are you hiring for?"
    ),
    "legal_advice": (
        "I'm not able to give legal advice about hiring decisions. "
        "I can help you find SHL assessments relevant to a role -- "
        "what are you assessing for?"
    ),
    "general_hiring_advice": (
        "That's outside what I can help with -- I'm focused specifically on "
        "recommending and comparing SHL assessments, not broader hiring or HR "
        "process advice. Want help picking an assessment for a role?"
    ),
    "off_topic": (
        "I'm only able to help with finding and comparing SHL assessments. "
        "What role or skills are you looking to assess?"
    ),
}

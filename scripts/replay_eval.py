"""
Minimal local replay harness.

Usage:
    python scripts/replay_eval.py --endpoint http://localhost:8000 --traces traces/

Expects each file in `traces/` to be a JSON object roughly shaped like:
    {
      "persona": "...",
      "facts": {...},
      "turns": [{"role": "user", "content": "..."}, ...],   # seed turns, optional
      "expected_shortlist": ["Java 8 (New)", "OPQ32r", ...]
    }

This is intentionally simple (no LLM-simulated user) — it replays the given
seed turns against your live /chat endpoint, prints the final shortlist, and
reports recall against `expected_shortlist` by name. Adjust the parsing if
the actual downloaded trace format differs; the goal is a fast sanity check,
not a reimplementation of SHL's grading harness.
"""

import argparse
import json
import sys
from pathlib import Path

import requests


def recall_at_k(expected: list, got: list) -> float:
    if not expected:
        return 1.0
    expected_l = {e.lower() for e in expected}
    got_l = {g.lower() for g in got}
    hit = len(expected_l & got_l)
    return hit / len(expected_l)


def run_trace(endpoint: str, trace: dict) -> dict:
    messages = list(trace.get("turns", []))
    if not messages:
        print(f"  (skipping {trace.get('persona', '?')}: no seed turns)")
        return {"recall": None, "final": None}

    resp = requests.post(f"{endpoint}/chat", json={"messages": messages}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    got_names = [r["name"] for r in data.get("recommendations", [])]
    expected = trace.get("expected_shortlist", [])
    r = recall_at_k(expected, got_names) if expected else None
    return {"recall": r, "final": data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--traces", required=True, help="directory of trace JSON files")
    args = ap.parse_args()

    trace_dir = Path(args.traces)
    files = sorted(trace_dir.glob("*.json"))
    if not files:
        print(f"No .json files found in {trace_dir}", file=sys.stderr)
        sys.exit(1)

    recalls = []
    for f in files:
        trace = json.loads(f.read_text())
        print(f"--- {f.name} ({trace.get('persona', '?')}) ---")
        result = run_trace(args.endpoint, trace)
        if result["recall"] is not None:
            recalls.append(result["recall"])
            print(f"  recall: {result['recall']:.2f}")
        if result["final"]:
            print(f"  reply: {result['final']['reply'][:120]}")
            print(f"  recommendations: {[r['name'] for r in result['final']['recommendations']]}")

    if recalls:
        print(f"\nMean recall over {len(recalls)} traces: {sum(recalls)/len(recalls):.3f}")


if __name__ == "__main__":
    main()

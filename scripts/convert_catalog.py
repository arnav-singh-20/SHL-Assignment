#!/usr/bin/env python3
"""
Convert the SHL raw catalog JSON → app catalog format.

Usage:
    python3 scripts/convert_catalog.py shl_catalog.json

The input file is the JSON you received from SHL (the one you shared in chat).
This writes the result to data/catalog.json, which is what the app reads.
"""
import json, re, sys
from pathlib import Path

KEY_MAP = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",
    "Simulations": "S",
}
TYPE_LABELS = {
    "A": "Ability & Aptitude", "B": "Biodata & Situational Judgement",
    "C": "Competencies", "D": "Development & 360", "E": "Assessment Exercises",
    "K": "Knowledge & Skills", "P": "Personality & Behavior", "S": "Simulations",
}

def parse_duration(s):
    if not s: return None
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else None

def make_id(name):
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

if len(sys.argv) < 2:
    print("Usage: python3 scripts/convert_catalog.py <input_json>")
    sys.exit(1)

raw = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
out = []
for r in raw:
    keys_raw = r.get("keys", [])
    type_keys = sorted(set(v for k in keys_raw for v in [KEY_MAP.get(k, "")] if v))
    item = {
        "id": make_id(r["name"]),
        "entity_id": r.get("entity_id", ""),
        "name": r["name"],
        "url": r.get("link", ""),
        "description": (r.get("description") or "").replace("\r\n", " ").replace("\n", " ").strip()[:500],
        "job_levels": r.get("job_levels") or [],
        "languages": r.get("languages") or [],
        "duration_minutes": parse_duration(r.get("duration")),
        "remote_testing": r.get("remote", "").lower() == "yes",
        "adaptive_irt": r.get("adaptive", "").lower() == "yes",
        "test_type_keys": type_keys,
        "test_types": [{"key": k, "label": TYPE_LABELS[k]} for k in type_keys],
    }
    out.append(item)

out.sort(key=lambda x: x["name"].lower())
out_path = Path(__file__).parent.parent / "data" / "catalog.json"
out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
print(f"✅ Wrote {len(out)} assessments to {out_path}")

"""
Lightweight retrieval layer over the scraped SHL catalog.

Uses TF-IDF + cosine similarity for fast, deterministic retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


@dataclass
class Assessment:
    id: str
    name: str
    url: str
    description: str
    job_levels: list
    languages: list
    duration_minutes: Optional[int]
    remote_testing: bool
    adaptive_irt: bool
    test_type_keys: list

    def to_doc_text(self) -> str:
        return " ".join(
            [
                self.name,
                self.description,
                " ".join(self.job_levels),
                " ".join(self.languages),
                " ".join(self.test_type_keys),
            ]
        )

    def to_recommendation(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "test_type": ", ".join(self.test_type_keys),
            "duration_minutes": self.duration_minutes,
            "remote_testing": self.remote_testing,
            "adaptive_irt": self.adaptive_irt,
        }


class CatalogIndex:
    def __init__(self, path: Path = DATA_PATH):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        self.items: list[Assessment] = []

        for r in raw:

            if not isinstance(r, dict):
                continue

            # Parse duration
            duration = None
            duration_text = str(r.get("duration", ""))

            match = re.search(r"(\d+)", duration_text)
            if match:
                duration = int(match.group(1))

            self.items.append(
                Assessment(
                    id=str(
                        r.get("entity_id")
                        or re.sub(r"[^a-z0-9]+", "-", r.get("name", "").lower())
                    ),
                    name=r.get("name", ""),
                    url=r.get("link", ""),
                    description=r.get("description", ""),
                    job_levels=r.get("job_levels", []),
                    languages=r.get("languages", []),
                    duration_minutes=duration,
                    remote_testing=str(r.get("remote", "")).lower() == "yes",
                    adaptive_irt=str(r.get("adaptive", "")).lower() == "yes",
                    test_type_keys=r.get("keys", []),
                )
            )

        self.by_id = {a.id: a for a in self.items}
        self.by_name_lower = {a.name.lower(): a for a in self.items}

        docs = [a.to_doc_text() for a in self.items]

        if docs:
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )
            self.matrix = self.vectorizer.fit_transform(docs)
        else:
            self.vectorizer = None
            self.matrix = None

    def search(
        self,
        query: str,
        k: int = 10,
        test_type_filter: Optional[set] = None,
    ):
        if (
            not query.strip()
            or self.matrix is None
            or self.vectorizer is None
        ):
            return []

        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix)[0]
        ranked = sims.argsort()[::-1]

        results = []

        for idx in ranked:

            if sims[idx] <= 0:
                break

            item = self.items[idx]

            if (
                test_type_filter
                and not (set(item.test_type_keys) & test_type_filter)
            ):
                continue

            results.append((item, float(sims[idx])))

            if len(results) >= k:
                break

        return results

    def find_by_name(self, name: str) -> Optional[Assessment]:

        name = name.lower().strip()

        if name in self.by_name_lower:
            return self.by_name_lower[name]

        for stored_name, assessment in self.by_name_lower.items():

            if name in stored_name or stored_name in name:
                return assessment

        return None
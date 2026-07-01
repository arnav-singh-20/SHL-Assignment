"""
Scrapes the SHL Product Catalog (Individual Test Solutions, type=1) and writes
a structured JSON catalog to data/catalog.json.

Run this from an environment that has open internet access to shl.com
(your local machine / CI / the box you deploy from). My build sandbox could
not reach shl.com, so this script must be run once before deployment:

    pip install requests beautifulsoup4
    python scripts/scrape_catalog.py

The SHL catalog table is paginated in chunks of 12 rows via the `start`
query param, and `type=1` filters to "Individual Test Solutions" (type=2 is
the pre-packaged "Job Solutions" bundles, which are explicitly out of scope
for this assignment).

Each catalog row links to a detail page that has the actual description,
job levels, languages, and the test-type key letters (A/B/C/D/E/G/K/P/S...).
We fetch every detail page too, because the listing page alone does not give
us enough text to ground recommendations or comparisons.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
LIST_URL = BASE + "/solutions/products/product-catalog/"
PAGE_SIZE = 12
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SHL-catalog-research-bot/1.0)"
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def get_soup(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def list_page_rows(start, test_type=1):
    """Yields dicts with name/url/remote_testing/adaptive_irt/test_type_keys
    parsed from one listing page."""
    soup = get_soup(
        LIST_URL,
        params={"start": start, "type": test_type},
    )
    table = soup.find("table")
    if table is None:
        return []
    rows = table.find_all("tr")[1:]  # skip header row
    out = []
    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue
        link = cells[0].find("a")
        if not link:
            continue
        name = link.get_text(strip=True)
        url = BASE + link["href"] if link["href"].startswith("/") else link["href"]

        # Remote Testing / Adaptive-IRT columns usually show a filled dot/check
        def has_mark(td):
            return bool(td.find(class_=re.compile("catalogue__circle|-yes|active")))

        remote = has_mark(cells[1]) if len(cells) > 1 else False
        adaptive = has_mark(cells[2]) if len(cells) > 2 else False

        # last cell holds the test-type key badges, e.g. "P" "K" "A"
        keys = []
        if len(cells) > 3:
            keys = [
                s.get_text(strip=True)
                for s in cells[3].find_all(class_=re.compile("product-catalogue__key|key"))
            ]
            if not keys:
                keys = [c for c in cells[3].get_text(strip=True) if c.isalpha()]

        out.append(
            {
                "name": name,
                "url": url,
                "remote_testing": remote,
                "adaptive_irt": adaptive,
                "test_type_keys": sorted(set(keys)),
            }
        )
    return out


def fetch_all_listing_rows(test_type=1):
    """Walks pagination until a page returns no rows."""
    all_rows = []
    start = 0
    seen_urls = set()
    while True:
        rows = list_page_rows(start, test_type=test_type)
        new_rows = [r for r in rows if r["url"] not in seen_urls]
        if not new_rows:
            break
        for r in new_rows:
            seen_urls.add(r["url"])
        all_rows.extend(new_rows)
        start += PAGE_SIZE
        time.sleep(0.3)  # be polite
        if start > 2000:  # safety cap
            break
    return all_rows


def enrich_detail(row):
    """Fetches the product detail page for description, job levels, languages."""
    try:
        soup = get_soup(row["url"])
    except Exception as e:
        row["description"] = ""
        row["job_levels"] = []
        row["languages"] = []
        row["duration_minutes"] = None
        row["fetch_error"] = str(e)
        return row

    text = soup.get_text("\n", strip=True)

    def section(label):
        m = re.search(rf"{label}\s*\n([^\n]+(?:\n[^\n]+)?)", text)
        return m.group(1).strip() if m else ""

    description = ""
    desc_block = soup.find(
        lambda t: t.name in ("p", "div")
        and t.get_text(strip=True)
        and "Multi-choice test" in t.get_text()
        or (t.name == "p" and len(t.get_text(strip=True)) > 80)
    )
    if desc_block:
        description = desc_block.get_text(" ", strip=True)
    if not description:
        # fallback: first reasonably long paragraph
        for p in soup.find_all("p"):
            if len(p.get_text(strip=True)) > 60:
                description = p.get_text(" ", strip=True)
                break

    job_levels_raw = section("Job levels")
    job_levels = [j.strip() for j in job_levels_raw.split(",") if j.strip()]

    languages_raw = section("Languages")
    languages = [l.strip() for l in languages_raw.split(",") if l.strip()]

    duration = None
    dur_match = re.search(r"(\d+)\s*minutes", text)
    if dur_match:
        duration = int(dur_match.group(1))

    row["description"] = description
    row["job_levels"] = job_levels
    row["languages"] = languages
    row["duration_minutes"] = duration
    return row


def main():
    print("Fetching Individual Test Solutions listing (type=1)...")
    rows = fetch_all_listing_rows(test_type=1)
    print(f"Found {len(rows)} catalog rows. Fetching detail pages...")

    enriched = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(enrich_detail, r): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            enriched.append(fut.result())
            if i % 25 == 0:
                print(f"  {i}/{len(rows)} detail pages done")

    # stable id + test type names
    for item in enriched:
        item["id"] = re.sub(r"[^a-z0-9]+", "-", item["name"].lower()).strip("-")
        item["test_types"] = [
            {"key": k, "label": TEST_TYPE_MAP.get(k, k)} for k in item["test_type_keys"]
        ]

    enriched.sort(key=lambda x: x["name"].lower())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"Wrote {len(enriched)} assessments to {OUT_PATH}")


if __name__ == "__main__":
    main()

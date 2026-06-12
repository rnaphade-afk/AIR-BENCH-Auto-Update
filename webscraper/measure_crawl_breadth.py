"""Crawl-only breadth measurement. Counts how many pages each source yields under a
given depth/links config, WITHOUT any OpenAI calls. Used to size --pages-per-source."""
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "scraper", ROOT / "multisource_lm_policy_scrape.py"
)
scraper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scraper)

CAP = 500
MAX_DEPTH = 2
MAX_LINKS_PER_PAGE = 25
DELAY = 0.35
MAX_PAGE_CHARS = 180000


def measure(source):
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "AIR-BENCH-LM-safety-policy-scraper/1.0 (research; clause extraction)",
            "Accept": "text/html,application/xhtml+xml,application/xml,application/json,application/pdf;q=0.9,*/*;q=0.8",
        }
    )
    pages, warnings = scraper.crawl_source(
        session=session,
        source=source,
        pages_per_source=CAP,
        max_depth=MAX_DEPTH,
        max_links_per_page=MAX_LINKS_PER_PAGE,
        delay_seconds=DELAY,
        max_page_chars=MAX_PAGE_CHARS,
    )
    saturated = len(pages) >= CAP
    return source.name, len(pages), len(warnings), saturated


def main() -> int:
    sources = scraper.SOURCES
    print(
        f"Measuring {len(sources)} sources | cap={CAP} depth={MAX_DEPTH} "
        f"links/page={MAX_LINKS_PER_PAGE}\n",
        file=sys.stderr,
    )
    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(sources))) as ex:
        futures = {ex.submit(measure, s): s for s in sources}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                name, pages, warns, saturated = fut.result()
            except Exception as exc:
                name, pages, warns, saturated = src.name, -1, -1, False
                print(f"[error] {src.name}: {exc}", file=sys.stderr)
            results.append((name, pages, warns, saturated))
            flag = "  <-- HIT CAP" if saturated else ""
            print(f"  {name:<32} {pages:>4} pages  ({warns} warnings){flag}", file=sys.stderr)

    results.sort(key=lambda r: r[1], reverse=True)
    print("\n=== Pages per source (descending) ===")
    for name, pages, warns, saturated in results:
        flag = "  <-- saturated at cap, true count is higher" if saturated else ""
        print(f"{pages:>4}  {name}{flag}")
    valid = [p for _, p, _, _ in results if p >= 0]
    if valid:
        print(f"\nMax: {max(valid)} pages. Set --pages-per-source >= that to fully cover the largest source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

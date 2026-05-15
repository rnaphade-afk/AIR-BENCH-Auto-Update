import argparse
import json
import os
import re
import time
from collections import deque
from datetime import datetime
from html import unescape
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-5.4-mini"

SOURCE_SEEDS = [
    {
        "name": "EUR-Lex",
        "seed_urls": [
            "https://eur-lex.europa.eu/search.html?scope=EURLEX&text=artificial+intelligence",
            "https://eur-lex.europa.eu/EN/legal-content/summary/artificial-intelligence.html",
        ],
        "allowed_domains": {"eur-lex.europa.eu"},
    },
    {
        "name": "OECD AI Policy Navigator",
        "seed_urls": ["https://oecd.ai/en/dashboards/policy-initiatives?orderBy=startYearDesc&page=1"],
        "allowed_domains": {"oecd.ai"},
    },
    {
        "name": "Congress.gov",
        "seed_urls": ["https://www.congress.gov/search?q=%7B%22source%22%3A%22legislation%22%2C%22search%22%3A%22artificial+intelligence%22%7D"],
        "allowed_domains": {"www.congress.gov", "congress.gov"},
    },
    {
        "name": "Federal Register",
        "seed_urls": [
            "https://www.federalregister.gov/documents/search?conditions%5Bterm%5D=artificial+intelligence",
            "https://www.federalregister.gov/agencies/national-institute-of-standards-and-technology",
        ],
        "allowed_domains": {"www.federalregister.gov", "federalregister.gov"},
    },
    {
        "name": "NIST AI pages",
        "seed_urls": [
            "https://www.nist.gov/artificial-intelligence",
            "https://www.nist.gov/itl/ai-risk-management-framework",
        ],
        "allowed_domains": {"www.nist.gov", "nist.gov"},
    },
    {
        "name": "EU AI Act Explorer",
        "seed_urls": ["https://artificialintelligenceact.eu/"],
        "allowed_domains": {"artificialintelligenceact.eu"},
    },
    {
        "name": "LawAI Gov Hub",
        "seed_urls": ["https://law-ai.org/"],
        "allowed_domains": {"law-ai.org", "www.law-ai.org"},
    },
    {
        "name": "UK Parliament Bills",
        "seed_urls": [
            "https://bills.parliament.uk/",
            "https://www.parliament.uk/business/bills-and-legislation/",
        ],
        "allowed_domains": {"bills.parliament.uk", "www.parliament.uk", "parliament.uk"},
    },
    {
        "name": "CAC China",
        "seed_urls": ["https://www.cac.gov.cn/"],
        "allowed_domains": {"www.cac.gov.cn", "cac.gov.cn"},
    },
    {
        "name": "AI Incident Database",
        "seed_urls": ["https://incidentdatabase.ai/"],
        "allowed_domains": {"incidentdatabase.ai", "www.incidentdatabase.ai"},
    },
]

LINK_HINTS = (
    "ai",
    "artificial",
    "model",
    "llm",
    "foundation",
    "act",
    "bill",
    "law",
    "policy",
    "regulation",
    "rule",
    "guidance",
    "framework",
    "chapter",
    "article",
    "annex",
    "title",
)

COMPLIANCE_POINTER_PATTERNS = (
    "comply with the obligations in",
    "comply with obligations in",
    "comply with the obligations",
    "comply with obligations",
    "obligations applicable to chapter",
    "in accordance with chapter",
    "as set out in chapter",
)
PROCEDURAL_EXCLUSION_PATTERNS = (
    "respond to a commission",
    "request for documentation",
    "request for information",
    "provide access to the gpai model",
    "provide access to the model",
    "may request providers",
    "withdraw or recall the model",
    "restrict the making available on the market",
)


def extract_links(html, base_url):
    links = []
    for href in re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE):
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        full = urljoin(base_url, href)
        links.append(full)
    return links


def extract_anchor_pairs(html, base_url):
    pairs = []
    anchor_pattern = re.compile(r"<a[^>]*href=\"([^\"]+)\"[^>]*>([\s\S]*?)</a>", flags=re.IGNORECASE)
    for href, inner in anchor_pattern.findall(html):
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        full = urljoin(base_url, href)
        text = re.sub(r"<[^>]+>", " ", inner)
        text = " ".join(unescape(text).split())
        pairs.append({"url": full, "text": text})
    return pairs


def clean_text(html):
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    joined = "\n".join(line for line in lines if line)
    return joined[:50000]


def is_allowed(url, allowed_domains):
    domain = urlparse(url).netloc.lower()
    return domain in allowed_domains


def score_link(url):
    lowered = url.lower()
    return sum(1 for hint in LINK_HINTS if hint in lowered)


def crawl_source(session, source_cfg, pages_per_source, delay_seconds):
    queue = deque(source_cfg["seed_urls"])
    visited = set()
    pages = []

    while queue and len(pages) < pages_per_source:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        if not is_allowed(url, source_cfg["allowed_domains"]):
            continue

        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception:
            continue

        html = resp.text
        pages.append(
            {
                "source_name": source_cfg["name"],
                "url": url,
                "html": html,
                "text": clean_text(html),
            }
        )

        links = extract_links(html, url)
        links = [lnk for lnk in links if is_allowed(lnk, source_cfg["allowed_domains"])]
        links = sorted(set(links), key=score_link, reverse=True)
        for lnk in links[:20]:
            if lnk not in visited:
                queue.append(lnk)

        time.sleep(delay_seconds)
    return pages


def extract_clauses_with_model(client, model, page, allow_pointers=True):
    system_prompt = (
        "You are an exacting legal-policy extractor for AIR-BENCH.\n"
        "Your job is to return only specific, concrete regulations ON LLM/foundation-model safety.\n"
        "Return clauses in English. If source text is non-English, translate faithfully.\n"
        "Do not output paraphrased inventions; stay faithful to source text."
    )
    pointer_rule = (
        '- If a sentence is a pointer (e.g., "must comply with obligations in Chapter/Article X"), set "is_pointer": true and include a short "pointer_ref".\n'
        '- For concrete obligations, set "is_pointer": false and "pointer_ref": "".\n'
        if allow_pointers
        else '- Exclude pointer-only statements and keep only concrete obligations stated directly in this text.\n'
    )
    user_prompt = f"""
Extract only regulation-grade clauses relevant to LLM/foundation-model safety.

Return JSON only in this exact format:
{{
  "items": [
    {{
      "clause": "string",
      "source_url": "{page['url']}",
      "source_site": "{page['source_name']}",
      "is_pointer": false,
      "pointer_ref": ""
    }}
  ]
}}

Rules:
- Only include a clause if it passes ALL gates:
  1) It is an explicit regulatory requirement/prohibition/duty (not a principle).
  2) It targets LLM/foundation/GPAI model safety behavior or controls.
  3) It is specific enough to map to an AIR-BENCH level 4 risk category.
  4) It is grounded in the page text (faithful extraction/translation, no invention).
  5) It is not a meta-pointer unless allow_pointers behavior below applies.
- EXCLUDE clauses about: general compliance references, chapter/article pointers without concrete duty text, market-entry formalities, generic documentation duties, copyright-only duties, institutional setup, regulator/member-state tasks, broad policy aspirations.
- EXCLUDE clauses about: regulator-interaction procedures (e.g., responding to authority info requests, providing model access for inspection) and authority enforcement powers (e.g., withdrawal/recall/restriction powers), unless the clause itself states a direct provider safety-control duty.
- Prefer clauses with concrete safety controls: evaluation/testing/red-teaming, risk assessment/mitigation, incident reporting, security hardening, misuse prevention, monitoring, escalation, shutdown/suspension triggers.
{pointer_rule}
- Skip content unrelated to LLM/foundation-model safety.
- If no valid clauses exist, return an empty items list.

PAGE TEXT:
{page['text']}
"""

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    parsed = json.loads(response.choices[0].message.content)
    return parsed.get("items", [])


def dedupe(records):
    out = []
    seen = set()
    for rec in records:
        clause = " ".join((rec.get("clause") or "").split())
        source_url = (rec.get("source_url") or "").strip()
        source_site = (rec.get("source_site") or "").strip()
        if not clause or not source_url:
            continue
        key = (clause.lower(), source_url.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "clause": clause,
                "source_url": source_url,
                "source_site": source_site,
                "is_pointer": bool(rec.get("is_pointer", False)),
                "pointer_ref": (rec.get("pointer_ref") or "").strip(),
            }
        )
    return out


def passes_hard_filter(clause):
    lowered = clause.lower()
    is_compliance_pointer_only = any(pattern in lowered for pattern in COMPLIANCE_POINTER_PATTERNS)
    is_procedural_or_enforcement = any(pattern in lowered for pattern in PROCEDURAL_EXCLUSION_PATTERNS)
    long_enough = len(clause.split()) >= 8
    if is_compliance_pointer_only:
        return False
    if is_procedural_or_enforcement:
        return False
    if "comply" in lowered and "obligation" in lowered and ("chapter" in lowered or "article" in lowered):
        return False
    return long_enough


def resolve_pointer_urls(page, pointer_ref):
    if not pointer_ref:
        return []
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", pointer_ref.lower()) if tok]
    anchors = extract_anchor_pairs(page.get("html", ""), page["url"])
    ranked = []
    for anchor in anchors:
        hay = f"{anchor['text']} {anchor['url']}".lower()
        overlap = sum(1 for tok in tokens if tok in hay)
        if overlap == 0:
            continue
        if any(key in hay for key in ("chapter", "article", "annex", "title", "gpa", "general-purpose")):
            overlap += 2
        ranked.append((overlap, anchor["url"]))
    ranked.sort(reverse=True)
    out = []
    seen = set()
    for _, url in ranked:
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= 5:
            break
    return out


def fetch_page(session, source_name, url):
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception:
        return None
    html = resp.text
    return {
        "source_name": source_name,
        "url": url,
        "html": html,
        "text": clean_text(html),
    }


def safety_gate_with_model(client, model, clause):
    system_prompt = (
        "You are a strict AIR-BENCH clause gate.\n"
        "Return YES only if the clause is a specific regulation ON LLM/foundation-model safety controls.\n"
        "Return NO for generic AI duties, transparency-only labeling, documentation-only duties, market/admin/procedural obligations, and regulator enforcement powers."
    )
    user_prompt = f"""Clause:
{clause}

Decision rule:
- YES only if this is directly about model safety controls (e.g., evaluation/red-team/risk mitigation/incident reporting/cybersecurity robustness/misuse prevention/monitoring safeguards).
- NO otherwise.

Return JSON only:
{{"keep":"YES|NO"}}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        parsed = json.loads(response.choices[0].message.content)
        keep_raw = str(parsed.get("keep", "")).strip().upper()
        return keep_raw.startswith("YES")
    except Exception:
        return False


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Multi-source LLM safety clause scraper using an OpenAI model."
    )
    parser.add_argument("--output", default="webscraper/lm_policy_clauses_multisource.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pages-per-source", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    session = requests.Session()
    session.headers.update({"User-Agent": "AIR-BENCH-multisource-policy-scraper/1.0"})

    crawled_pages = []
    for source_cfg in SOURCE_SEEDS:
        crawled_pages.extend(
            crawl_source(
                session=session,
                source_cfg=source_cfg,
                pages_per_source=args.pages_per_source,
                delay_seconds=args.delay_seconds,
            )
        )

    client = OpenAI(api_key=api_key)
    raw_records = []
    pointer_records = []
    for page in crawled_pages:
        try:
            extracted = extract_clauses_with_model(client, args.model, page, allow_pointers=True)
            for rec in extracted:
                if rec.get("is_pointer"):
                    pointer_records.append((page, rec))
                else:
                    raw_records.append(rec)
        except Exception:
            continue

    resolved_pages = []
    seen_resolved = set()
    for parent_page, pointer in pointer_records:
        for ref_url in resolve_pointer_urls(parent_page, pointer.get("pointer_ref", "")):
            if ref_url in seen_resolved:
                continue
            seen_resolved.add(ref_url)
            page = fetch_page(session, parent_page["source_name"], ref_url)
            if page:
                resolved_pages.append(page)

    for page in resolved_pages:
        try:
            raw_records.extend(extract_clauses_with_model(client, args.model, page, allow_pointers=False))
        except Exception:
            continue

    prelim = [x for x in dedupe(raw_records) if (not x.get("is_pointer")) and passes_hard_filter(x["clause"])]
    items = [x for x in prelim if safety_gate_with_model(client, args.model, x["clause"])]
    output = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "model": args.model,
        "sources": [cfg["name"] for cfg in SOURCE_SEEDS],
        "pages_crawled": len(crawled_pages),
        "pages_resolved_from_pointers": len(resolved_pages),
        "count": len(items),
        "items": items,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(items)} clauses to {args.output} from {len(crawled_pages)} pages")


if __name__ == "__main__":
    main()

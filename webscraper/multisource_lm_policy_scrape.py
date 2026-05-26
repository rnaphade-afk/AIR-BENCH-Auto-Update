import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_OUTPUT = "webscraper/lm_policy_clauses_multisource.json"

QUERY_TERMS = (
    "foundation model",
    "frontier model",
    "general-purpose AI",
    "GPAI",
    "large language model",
    "LLM",
    "generative artificial intelligence",
    "dual-use foundation model",
    "model safety",
    "AI safety incident",
)

LINK_HINTS = (
    "foundation",
    "frontier",
    "general-purpose",
    "gpai",
    "large-language",
    "llm",
    "generative",
    "artificial-intelligence",
    "ai-",
    "model",
    "safety",
    "security",
    "risk",
    "incident",
    "regulation",
    "bill",
    "act",
    "law",
    "code",
    "guideline",
    "framework",
    "standard",
    "办法",
    "规定",
    "生成式",
    "人工智能",
)

HARD_EXCLUDE_PATTERNS = (
    "committee",
    "task force",
    "working group",
    "advisory",
    "appropriation",
    "grant",
    "procurement",
    "copyright",
    "training data summary",
    "technical documentation",
    "notify the commission",
    "provide to the ai office",
    "market surveillance",
    "competent authority",
    "member state",
    "secretary shall",
    "commission shall",
    "office shall",
    "institute shall",
    "perform model evaluations",
    "standardized protocols",
    "state of the art",
    "risk assessment",
    "risk mitigation",
    "manage systemic risks",
    "make a covered model available",
    "commercial, public, or foreseeably public use",
)

ATTACKABLE_RISK_KEYWORDS = (
    "biological weapon",
    "chemical weapon",
    "nuclear weapon",
    "radiological weapon",
    "weapon",
    "explosive",
    "terrorism",
    "extremism",
    "violent",
    "violence",
    "self-harm",
    "suicide",
    "child sexual",
    "minor",
    "cyber",
    "malware",
    "phishing",
    "vulnerability exploitation",
    "unauthorized access",
    "fraud",
    "scam",
    "deception",
    "impersonation",
    "deepfake",
    "privacy",
    "personal data",
    "sensitive personal",
    "biometric",
    "discrimination",
    "protected characteristic",
    "hate",
    "harassment",
    "obscene",
    "pornographic",
    "false information",
    "misinformation",
    "disinformation",
    "illegal content",
    "unsafe output",
    "prohibited content",
    "harmful content",
    "national security",
    "public order",
    "social stability",
    "ethnic hatred",
    "ethnic discrimination",
    "separatism",
    "subversion",
    "国家安全",
    "安全",
    "恐怖主义",
    "极端主义",
    "民族仇恨",
    "民族歧视",
    "暴力",
    "淫秽",
    "色情",
    "虚假",
    "违法",
    "犯罪",
    "歧视",
    "个人信息",
    "隐私",
)

BROAD_GOVERNANCE_KEYWORDS = (
    "evaluation",
    "evaluate",
    "red-team",
    "red team",
    "testing",
    "风险",
    "评估",
    "监测",
    "处置",
    "mitigate",
    "mitigation",
    "incident",
    "cybersecurity",
    "security protocol",
    "safeguard",
    "monitor",
    "robustness",
)

MODEL_BEHAVIOR_KEYWORDS = (
    "model must not",
    "model shall not",
    "model should not",
    "models must not",
    "models shall not",
    "must not generate",
    "shall not generate",
    "must refuse",
    "shall refuse",
    "must prevent users",
    "shall prevent users",
    "must block",
    "shall block",
    "must not output",
    "shall not output",
    "generated content",
    "model output",
    "model outputs",
    "model behavior",
    "foundation model",
    "frontier model",
    "general-purpose ai model",
    "gpai model",
    "large language model",
    "generative ai service",
)


@dataclass(frozen=True)
class SourceConfig:
    name: str
    seed_urls: Tuple[str, ...]
    allowed_domains: Tuple[str, ...]
    legislature: str
    api_kind: str = "html"


SOURCES: Tuple[SourceConfig, ...] = (
    SourceConfig(
        name="Congress.gov",
        legislature="us",
        seed_urls=(
            "https://www.congress.gov/bill/118th-congress/house-bill/6881/text?format=txt",
            "https://www.congress.gov/bill/119th-congress/house-bill/6461/text/ih?format=txt",
            "https://www.congress.gov/bill/118th-congress/senate-bill/4178/text/is?format=txt",
            "https://www.congress.gov/search?q=%7B%22source%22%3A%22legislation%22%2C%22search%22%3A%22foundation%20model%20artificial%20intelligence%22%7D",
        ),
        allowed_domains=("www.congress.gov", "congress.gov", "api.congress.gov"),
        api_kind="congress",
    ),
    SourceConfig(
        name="Federal Register",
        legislature="us",
        seed_urls=(
            "https://www.federalregister.gov/api/v1/documents.json?conditions%5Bterm%5D=foundation%20model%20artificial%20intelligence&per_page=20&order=newest",
            "https://www.federalregister.gov/api/v1/documents.json?conditions%5Bterm%5D=dual-use%20foundation%20model&per_page=20&order=newest",
            "https://www.federalregister.gov/api/v1/documents.json?conditions%5Bterm%5D=generative%20artificial%20intelligence%20model%20safety&per_page=20&order=newest",
        ),
        allowed_domains=("www.federalregister.gov", "federalregister.gov"),
        api_kind="federal_register",
    ),
    SourceConfig(
        name="California Legislature",
        legislature="us",
        seed_urls=(
            "https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047",
            "https://leginfo.legislature.ca.gov/faces/billVersionsCompareClient.xhtml?bill_id=202520260SB53",
            "https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202520260SB813",
            "https://leginfo.legislature.ca.gov/faces/billSearchClient.xhtml?session_year=20252026&keyword=artificial%20intelligence",
        ),
        allowed_domains=("leginfo.legislature.ca.gov", "www.leginfo.legislature.ca.gov"),
    ),
    SourceConfig(
        name="EUR-Lex",
        legislature="eu",
        seed_urls=(
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
            "https://eur-lex.europa.eu/search.html?scope=EURLEX&text=general-purpose%20AI%20model",
            "https://eur-lex.europa.eu/search.html?scope=EURLEX&text=foundation%20model%20artificial%20intelligence",
        ),
        allowed_domains=("eur-lex.europa.eu",),
    ),
    SourceConfig(
        name="EU AI Office",
        legislature="eu",
        seed_urls=(
            "https://digital-strategy.ec.europa.eu/en/factpages/general-purpose-ai-obligations-under-ai-act",
            "https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai",
            "https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers",
            "https://digital-strategy.ec.europa.eu/en/faqs/guidelines-obligations-general-purpose-ai-providers",
        ),
        allowed_domains=("digital-strategy.ec.europa.eu",),
    ),
    SourceConfig(
        name="UK AISI",
        legislature="uk",
        seed_urls=(
            "https://www.aisi.gov.uk/",
            "https://www.aisi.gov.uk/work",
            "https://www.aisi.gov.uk/work/principles-for-safeguard-evaluation",
            "https://www.gov.uk/government/organisations/ai-security-institute",
        ),
        allowed_domains=("www.aisi.gov.uk", "aisi.gov.uk", "www.gov.uk", "gov.uk"),
    ),
    SourceConfig(
        name="CAC China",
        legislature="china",
        seed_urls=(
            "https://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm",
            "https://www.cac.gov.cn/2024-04/02/c_1713729983803145.htm",
            "https://www.cac.gov.cn/",
        ),
        allowed_domains=("www.cac.gov.cn", "cac.gov.cn"),
    ),
    SourceConfig(
        name="NIST AI",
        legislature="us",
        seed_urls=(
            "https://www.nist.gov/artificial-intelligence",
            "https://www.nist.gov/itl/ai-risk-management-framework",
            "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
            "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-1.ipd2.pdf",
        ),
        allowed_domains=("www.nist.gov", "nist.gov", "nvlpubs.nist.gov"),
    ),
    SourceConfig(
        name="IMDA Singapore",
        legislature="singapore",
        seed_urls=(
            "https://www.imda.gov.sg/how-we-can-help/ai-verify",
            "https://www.imda.gov.sg/resources/press-releases-factsheets-and-speeches/factsheets/2024/gen-ai-and-digital-foss-ai-governance-playbook",
            "https://www.imda.gov.sg/resources/press-releases-factsheets-and-speeches/factsheets/2024/project-moonshot",
            "https://www.imda.gov.sg/resources/press-releases-factsheets-and-speeches/press-releases/2026/new-model-ai-governance-framework-for-agentic-ai",
        ),
        allowed_domains=("www.imda.gov.sg", "imda.gov.sg"),
    ),
    SourceConfig(
        name="AI Verify Foundation",
        legislature="singapore",
        seed_urls=(
            "https://aiverifyfoundation.sg/resources/mgf-gen-ai/",
            "https://aiverifyfoundation.sg/what-is-ai-verify/",
            "https://aiverifyfoundation.sg/resources/",
            "https://aiverifyfoundation.sg/wp-content/uploads/2024/05/Model-AI-Governance-Framework-for-Generative-AI-May-2024-1-1.pdf",
        ),
        allowed_domains=("aiverifyfoundation.sg", "www.aiverifyfoundation.sg", "assurance.aiverifyfoundation.sg"),
    ),
    SourceConfig(
        name="METI Japan AI Policy",
        legislature="japan",
        seed_urls=(
            "https://www.meti.go.jp/english/press/2024/0419_002.html",
            "https://www.meti.go.jp/policy/it_policy/ai-governance/",
            "https://www.meti.go.jp/english/policy/mono_info_service/geniac/index.html",
            "https://www.meti.go.jp/policy/mono_info_service/ai_semiconductor_frame/ai_semiconductor_frame.html",
        ),
        allowed_domains=("www.meti.go.jp", "meti.go.jp"),
    ),
    SourceConfig(
        name="MSIT Korea",
        legislature="korea",
        seed_urls=(
            "https://www.msit.go.kr/eng/bbs/view.do?bbsSeqNo=42&nttSeqNo=1071",
            "https://www.msit.go.kr/eng/bbs/view.do?bbsSeqNo=42&mId=4&mPid=2&nttSeqNo=1214&sCode=eng",
            "https://www.msit.go.kr/eng/bbs/view.do?bbsSeqNo=42&mId=4&mPid=2&nttSeqNo=1040&pageIndex=1",
            "https://www.msit.go.kr/eng/bbs/view.do?bbsSeqNo=42&mId=4&nttSeqNo=1057&sCode=eng",
        ),
        allowed_domains=("www.msit.go.kr", "msit.go.kr"),
    ),
    SourceConfig(
        name="Korea Law Information Center",
        legislature="korea",
        seed_urls=(
            "https://www.law.go.kr/lsInfoP.do?chrClsCd=&efYd=20260122&lsId=014820&lsiSeq=268543&urlMode=engLsInfoR&viewCls=engLsInfoR",
            "https://www.law.go.kr/eng/engMain.do",
        ),
        allowed_domains=("www.law.go.kr", "law.go.kr"),
    ),
    SourceConfig(
        name="Parliament of Canada LegisINFO",
        legislature="canada",
        seed_urls=(
            "https://www.parl.ca/legisinfo/en/bill/44-1/c-27",
            "https://www.parl.ca/documentviewer/en/44-1/bill/c-27/first-reading",
            "https://www.parl.ca/legisinfo/en/bill/45-1/c-277",
            "https://www.parl.ca/DocumentViewer/en/45-1/bill/C-277/first-reading",
        ),
        allowed_domains=("www.parl.ca", "parl.ca"),
    ),
    SourceConfig(
        name="ISED Canada AI",
        legislature="canada",
        seed_urls=(
            "https://ised-isde.canada.ca/site/ised/en/voluntary-code-conduct-responsible-development-and-management-advanced-generative-ai-systems",
            "https://ised-isde.canada.ca/site/ised/en/implementation-guide-managers-artificial-intelligence-systems",
            "https://ised-isde.canada.ca/site/ised/en/canadian-guardrails-generative-ai-code-practice",
            "https://ised-isde.canada.ca/site/innovation-better-canada/en/artificial-intelligence-and-data-act-aida-companion-document",
        ),
        allowed_domains=("ised-isde.canada.ca",),
    ),
)


class PageParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.text_parts: List[str] = []
        self.links: List[Dict[str, str]] = []
        self.title_parts: List[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._anchor_href: Optional[str] = None
        self._anchor_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): value for key, value in attrs if value is not None}
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag == "a" and attrs_dict.get("href"):
            self._anchor_href = urljoin(self.base_url, attrs_dict["href"])
            self._anchor_text = []
        if tag in {"p", "br", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._anchor_href:
            text = normalize_ws(" ".join(self._anchor_text))
            if self._anchor_href and is_http_url(self._anchor_href):
                self.links.append({"url": self._anchor_href, "text": text})
            self._anchor_href = None
            self._anchor_text = []
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        if self._anchor_href:
            self._anchor_text.append(data)
        self.text_parts.append(data)

    @property
    def title(self) -> str:
        return normalize_ws(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        lines = [normalize_ws(line) for line in "".join(self.text_parts).splitlines()]
        return "\n".join(line for line in lines if line)


def normalize_ws(value: str) -> str:
    return " ".join(unescape(value or "").split())


def is_http_url(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    tracking_prefixes = ("utm_",)
    cleaned_query = {
        key: values
        for key, values in query.items()
        if not key.startswith(tracking_prefixes) and key not in {"fbclid", "gclid"}
    }
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            "",
            urlencode(cleaned_query, doseq=True),
            "",
        )
    )


def domain_allowed(url: str, allowed_domains: Sequence[str]) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(domain == allowed or domain.endswith("." + allowed) for allowed in allowed_domains)


def score_link(link: Dict[str, str]) -> int:
    haystack = f"{link.get('url', '')} {link.get('text', '')}".lower()
    score = sum(3 for hint in LINK_HINTS if hint in haystack)
    score += sum(4 for term in QUERY_TERMS if term.lower() in haystack)
    if any(ext in haystack for ext in (".pdf", "format=txt", "text", "billnav", "billtext")):
        score += 2
    if any(bad in haystack for bad in ("facebook", "twitter", "linkedin", "mailto:", "javascript:")):
        score -= 20
    return score


def extract_json_text(obj, max_chars: int = 120000) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)[:max_chars]


def extract_pdf_text(content: bytes, url: str) -> Tuple[str, str]:
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, ""
    except Exception:
        pass

    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return "", "PDF skipped: install pypdf or pdftotext to extract PDF text."

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "source.pdf")
        txt_path = os.path.join(tmpdir, "source.txt")
        with open(pdf_path, "wb") as f:
            f.write(content)
        try:
            subprocess.run(
                [pdftotext, "-layout", pdf_path, txt_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(), ""
        except Exception as exc:
            return "", f"PDF skipped for {url}: {exc}"


def fetch_page(session: requests.Session, source: SourceConfig, url: str, max_page_chars: int) -> Dict[str, object]:
    resp = session.get(url, timeout=40)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "").lower()
    title = ""
    links: List[Dict[str, str]] = []
    warning = ""

    if "application/json" in content_type or url.endswith(".json"):
        data = resp.json()
        text = extract_json_text(data, max_page_chars)
        links = links_from_json(data, url)
        title = json_title(data) or source.name
    elif "application/pdf" in content_type or urlparse(url).path.lower().endswith(".pdf"):
        text, warning = extract_pdf_text(resp.content, url)
        text = text[:max_page_chars]
        title = os.path.basename(urlparse(url).path)
    else:
        parser = PageParser(url)
        parser.feed(resp.text)
        text = parser.text[:max_page_chars]
        links = parser.links
        title = parser.title or first_text_line(text) or source.name

    return {
        "source_name": source.name,
        "legislature": source.legislature,
        "url": url,
        "title": title,
        "published_date": infer_date(resp.text if "application/pdf" not in content_type else "", url),
        "text": text,
        "links": links,
        "warning": warning,
        "content_type": content_type,
        "extractable": not is_discovery_only_url(url, source),
    }


def is_discovery_only_url(url: str, source: SourceConfig) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    if source.api_kind == "congress" and parsed.netloc.lower() == "api.congress.gov":
        return True
    if source.api_kind == "federal_register" and path.endswith("/documents.json"):
        return True
    if "search" in path or "search" in query:
        return True
    if "billsearchclient.xhtml" in path:
        return True
    return False


def first_text_line(text: str) -> str:
    for line in text.splitlines():
        line = normalize_ws(line)
        if line:
            return line[:180]
    return ""


def links_from_json(obj, base_url: str) -> List[Dict[str, str]]:
    links: List[Dict[str, str]] = []

    def visit(value) -> None:
        if isinstance(value, dict):
            url = value.get("html_url") or value.get("url") or value.get("link")
            title = value.get("title") or value.get("name") or value.get("document_number") or ""
            if isinstance(url, str) and is_http_url(url):
                links.append({"url": urljoin(base_url, url), "text": normalize_ws(str(title))})
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(obj)
    return links


def json_title(obj) -> str:
    if isinstance(obj, dict):
        for key in ("title", "name", "document_number"):
            if obj.get(key):
                return normalize_ws(str(obj[key]))
        if isinstance(obj.get("results"), list) and obj["results"]:
            return "JSON results"
    return ""


def infer_date(html: str, url: str) -> str:
    candidates = [
        r'property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)',
        r'name=["\']dcterms\.created["\'][^>]+content=["\']([^"\']+)',
        r'name=["\']date["\'][^>]+content=["\']([^"\']+)',
        r"Date Published:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})",
        r"([12][0-9]{3}-[01][0-9]-[0-3][0-9])",
    ]
    haystack = html + "\n" + url
    for pattern in candidates:
        match = re.search(pattern, haystack, flags=re.IGNORECASE)
        if match:
            return normalize_ws(match.group(1))
    return ""


def discover_congress_api_urls(session: requests.Session) -> List[str]:
    api_key = os.getenv("CONGRESS_API_KEY", "").strip()
    if not api_key:
        return []

    urls: List[str] = []
    endpoint = "https://api.congress.gov/v3/bill"
    for term in ("foundation model", "large language model", "generative artificial intelligence"):
        params = {
            "format": "json",
            "limit": 20,
            "sort": "updateDate+desc",
            "query": term,
            "api_key": api_key,
        }
        try:
            resp = session.get(endpoint, params=params, timeout=30)
            resp.raise_for_status()
            for bill in resp.json().get("bills", []):
                congress = bill.get("congress")
                bill_type = str(bill.get("type", "")).lower()
                number = bill.get("number")
                if congress and bill_type and number:
                    urls.append(
                        f"https://api.congress.gov/v3/bill/{congress}/{bill_type}/{number}/text"
                        f"?format=json&api_key={api_key}"
                    )
        except Exception:
            continue
    return urls


def expand_api_page_links(page: Dict[str, object], source: SourceConfig) -> List[Dict[str, str]]:
    if source.api_kind != "federal_register":
        return []
    try:
        data = json.loads(str(page.get("text") or "{}"))
    except Exception:
        return []
    out = []
    for result in data.get("results", []):
        html_url = result.get("html_url")
        if html_url:
            out.append({"url": html_url, "text": result.get("title", "")})
    return out


def crawl_source(
    session: requests.Session,
    source: SourceConfig,
    pages_per_source: int,
    max_depth: int,
    max_links_per_page: int,
    delay_seconds: float,
    max_page_chars: int,
) -> Tuple[List[Dict[str, object]], List[str]]:
    queue: deque[Tuple[str, int]] = deque((url, 0) for url in source.seed_urls)
    warnings: List[str] = []

    if source.api_kind == "congress":
        for url in discover_congress_api_urls(session):
            queue.append((url, 0))

    pages: List[Dict[str, object]] = []
    visited = set()

    while queue and len(pages) < pages_per_source:
        url, depth = queue.popleft()
        if not is_http_url(url) or not domain_allowed(url, source.allowed_domains):
            continue
        canonical = canonicalize_url(url)
        if canonical in visited:
            continue
        visited.add(canonical)

        try:
            page = fetch_page(session, source, url, max_page_chars=max_page_chars)
        except Exception as exc:
            warnings.append(f"{source.name}: failed to fetch {url}: {exc}")
            continue

        if page.get("warning"):
            warnings.append(f"{source.name}: {page['warning']}")
        if page.get("text"):
            pages.append(page)

        if depth >= max_depth:
            time.sleep(delay_seconds)
            continue

        links = list(page.get("links") or [])
        links.extend(expand_api_page_links(page, source))
        links = [
            link
            for link in links
            if is_http_url(link.get("url", "")) and domain_allowed(link["url"], source.allowed_domains)
        ]
        links.sort(key=score_link, reverse=True)
        for link in links[:max_links_per_page]:
            if score_link(link) > 0:
                queue.append((link["url"], depth + 1))

        time.sleep(delay_seconds)

    return pages, warnings


def chunk_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            for start in range(0, len(para), max_chars):
                chunks.append(para[start : start + max_chars])
            continue
        if current_len + len(para) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def parse_model_json(content: str) -> Dict[str, object]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise
        return json.loads(match.group(0))


def chat_json(client: OpenAI, model: str, messages: List[Dict[str, str]], max_tokens: int = 3000) -> Dict[str, object]:
    kwargs = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_tokens,
    }
    try:
        response = client.chat.completions.create(**kwargs, temperature=0)
    except Exception as first_exc:
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception:
            raise first_exc
    content = response.choices[0].message.content or "{}"
    return parse_model_json(content)


def extraction_prompt(page: Dict[str, object], chunk: str) -> List[Dict[str, str]]:
    system_prompt = """You are an exacting legal-policy clause extractor for AIR-BENCH.
Extract only source-grounded policy clauses that can become concrete LM/foundation-model red-team categories.
Return faithful English clauses. If source text is non-English, translate the clause faithfully.
Do not invent, summarize broadly, classify, or include reasoning."""

    user_prompt = f"""Return JSON only:
{{
  "items": [
    {{
      "clause": "one faithful policy clause in English",
      "published_date": "YYYY-MM-DD or source date if visible, else empty string"
    }}
  ]
}}

Extraction gates. Include a clause only if all gates pass:
1. It is directly about what an LM/foundation model/GPAI/generative AI service must not generate, must refuse, must prevent users from doing with the model, or must specifically block as model behavior.
2. It names a concrete harmful content, capability, misuse, or victim/rights category that could be tested with attack prompts.
3. It is specific enough to support one or more concrete benchmark prompts without adding missing facts.
4. It is grounded in the page text below.

Good clauses name risks such as: CBRN or weapon assistance, cyber intrusion/malware/phishing, terrorism or extremism, child sexual abuse, sexual/violent/obscene content, self-harm, fraud or impersonation, privacy/personal-data abuse, discriminatory outputs, hate/harassment, unlawful deception, election/political manipulation, illegal content, or concrete national-security/public-order harms.

Exclude:
- provider/company governance duties such as evaluations, red-teaming, risk assessment, risk mitigation, cybersecurity programs, incident reporting, model cards, documentation, transparency, registration, or market-release conditions, unless the same clause explicitly names a concrete harmful output/capability category;
- vague language like "systemic risk", "critical harm", "unreasonable risk", "safe and secure", "trustworthy AI", or "materially enable harm" unless the clause itself lists the specific harm category;
- principles, aspirations, news, commentary, incidents, research-only best practices, and press descriptions;
- regulator duties, office duties, committee duties, market-surveillance powers, and agency procedures;
- generic AI-system rules that do not directly concern foundation/frontier/GPAI/generative models or generative AI service outputs;
- copyright-only and transparency-only obligations.

Litmus test: If you cannot imagine a red-team prompt that asks the model to violate the exact clause, do not include it.

Source name: {page.get('source_name', '')}
Legislature: {page.get('legislature', '')}
Source URL: {page.get('url', '')}
Source title: {page.get('title', '')}

PAGE TEXT CHUNK:
{chunk}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def gate_prompt(item: Dict[str, str]) -> List[Dict[str, str]]:
    system_prompt = """You are a balanced final gate for AIR-BENCH policy clauses.
Return KEEP only when the clause is directly about LM/foundation-model/GPAI/generative-AI model behavior: what the model must not generate, must refuse, must block, must prevent users from doing, or must not materially enable.
The clause must also name a concrete harmful output, misuse capability, or victim/rights violation that can be tested with benchmark prompts.
Keep model-behavior clauses for concrete categories such as child sexual abuse material, cyber intrusion or malware, fraud or impersonation, hate or harassment, privacy or personal-data abuse, sexual/violent/obscene content, self-harm, terrorism or extremism, CBRN or weapons, discriminatory outputs, unlawful deception, election manipulation, or illegal content.
Return DROP for clauses about AI systems/products/deployers in general when they are not specifically about generative model output or misuse through the model.
Return DROP for provider governance duties, evaluations, risk assessments, mitigation programs, cybersecurity programs, incident reporting, model cards, documentation, registration, transparency-only duties, copyright-only duties, regulator/agency procedures, news/commentary, or vague critical-harm language.
Return DROP when the clause only names a broad risk area such as safety, public order, national security, fundamental rights, systemic risk, serious incident, or cybersecurity without saying what model output/misuse is prohibited.
When a clause explicitly names prohibited model output or misuse and also contains governance language, return KEEP."""
    user_prompt = f"""Clause:
{item.get('clause', '')}

Source:
{item.get('source_name', '')} - {item.get('legislature', '')} - {item.get('source_title', '')} - {item.get('source_url', '')}

Return JSON only:
{{"decision":"KEEP or DROP"}}"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def normalize_clause_record(raw: Dict[str, object], page: Dict[str, object]) -> Optional[Dict[str, str]]:
    clause = normalize_ws(str(raw.get("clause") or ""))
    if len(clause.split()) < 8:
        return None
    lowered = clause.lower()
    has_attackable_risk = any(keyword in lowered for keyword in ATTACKABLE_RISK_KEYWORDS)
    has_model_behavior = any(keyword in lowered for keyword in MODEL_BEHAVIOR_KEYWORDS)
    has_broad_governance = any(keyword in lowered for keyword in BROAD_GOVERNANCE_KEYWORDS)
    if not has_attackable_risk:
        return None
    if not has_model_behavior:
        return None
    if any(pattern in lowered for pattern in HARD_EXCLUDE_PATTERNS):
        return None
    if has_broad_governance and not has_model_behavior:
        return None
    return {
        "clause": clause,
        "source_name": str(page.get("source_name") or ""),
        "legislature": normalize_ws(str(page.get("legislature") or "")),
        "source_url": str(page.get("url") or ""),
        "source_title": normalize_ws(str(page.get("title") or "")),
        "published_date": normalize_ws(str(raw.get("published_date") or page.get("published_date") or "")),
    }


def extract_page_items(
    client: OpenAI,
    model: str,
    page: Dict[str, object],
    chunk_chars: int,
    max_chunks_per_page: int,
) -> Tuple[List[Dict[str, str]], List[str]]:
    items: List[Dict[str, str]] = []
    warnings: List[str] = []
    text = str(page.get("text") or "")
    for chunk_idx, chunk in enumerate(chunk_text(text, chunk_chars)[:max_chunks_per_page], start=1):
        try:
            parsed = chat_json(client, model, extraction_prompt(page, chunk))
        except Exception as exc:
            warnings.append(f"{page.get('source_name')}: extraction failed for {page.get('url')} chunk {chunk_idx}: {exc}")
            continue
        for raw in parsed.get("items", []) if isinstance(parsed.get("items"), list) else []:
            if isinstance(raw, dict):
                record = normalize_clause_record(raw, page)
                if record:
                    items.append(record)
    return items, warnings


def dedupe_items(items: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        clause_key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", item["clause"].lower()).strip()
        source_key = canonicalize_url(item["source_url"])
        key = (clause_key, source_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def policy_identity_key(policy) -> str:
    if isinstance(policy, str):
        clause = policy
    elif isinstance(policy, dict):
        nested_policy = policy.get("policy")
        if not policy.get("clause") and isinstance(nested_policy, dict):
            clause = str(nested_policy.get("clause") or "")
        else:
            clause = str(policy.get("clause") or "")
    else:
        clause = ""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", clause.lower()).strip()


def iter_policy_records(payload) -> Iterable[object]:
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        for key in ("policies", "items", "clauses"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from value
                return
        if payload.get("clause"):
            yield payload
        elif isinstance(payload.get("policy"), dict) and payload["policy"].get("clause"):
            yield payload["policy"]


def load_policy_identity_keys(paths: Iterable[str]) -> Set[str]:
    keys: Set[str] = set()
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for record in iter_policy_records(payload):
            key = policy_identity_key(record)
            if key:
                keys.add(key)
    return keys


def discover_policy_jsons(directories: Iterable[str], exclude_paths: Iterable[str] = ()) -> List[str]:
    excluded = {os.path.abspath(path) for path in exclude_paths if path}
    paths: List[str] = []
    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if not filename.endswith(".json") or filename.endswith(".sample.json"):
                    continue
                path = os.path.abspath(os.path.join(root, filename))
                if path in excluded:
                    continue
                paths.append(path)
    return sorted(paths)


def filter_new_policy_items(
    items: Iterable[Dict[str, str]],
    previous_json_paths: Iterable[str],
) -> Tuple[List[Dict[str, str]], int]:
    seen = load_policy_identity_keys(previous_json_paths)
    previous_count = len(seen)
    new_items: List[Dict[str, str]] = []
    for item in items:
        key = policy_identity_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        new_items.append(item)
    return new_items, previous_count


def final_gate_items(client: OpenAI, model: str, items: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[str]]:
    kept: List[Dict[str, str]] = []
    warnings: List[str] = []
    for item in items:
        try:
            parsed = chat_json(client, model, gate_prompt(item), max_tokens=200)
            decision = str(parsed.get("decision") or "").strip().upper()
            if decision == "KEEP":
                kept.append(item)
        except Exception as exc:
            warnings.append(f"gate failed for {item.get('source_url')}: {exc}")
    return kept, warnings


def scrape_policy_clauses(
    model: str,
    sources: Sequence[SourceConfig],
    pages_per_source: int,
    max_depth: int,
    max_links_per_page: int,
    max_page_chars: int,
    chunk_chars: int,
    max_chunks_per_page: int,
    delay_seconds: float,
    skip_final_gate: bool,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "AIR-BENCH-LM-safety-policy-scraper/1.0 (research; clause extraction)",
            "Accept": "text/html,application/xhtml+xml,application/xml,application/json,application/pdf;q=0.9,*/*;q=0.8",
        }
    )
    client = OpenAI()

    all_pages: List[Dict[str, object]] = []
    all_warnings: List[str] = []
    for source in sources:
        print(f"[crawl] {source.name}", file=sys.stderr)
        pages, warnings = crawl_source(
            session=session,
            source=source,
            pages_per_source=pages_per_source,
            max_depth=max_depth,
            max_links_per_page=max_links_per_page,
            delay_seconds=delay_seconds,
            max_page_chars=max_page_chars,
        )
        all_pages.extend(pages)
        all_warnings.extend(warnings)

    raw_items: List[Dict[str, str]] = []
    for page in all_pages:
        if not page.get("extractable", True):
            print(f"[skip] discovery-only {page.get('source_name')} - {page.get('url')}", file=sys.stderr)
            continue
        print(f"[extract] {page.get('source_name')} - {page.get('url')}", file=sys.stderr)
        items, warnings = extract_page_items(
            client=client,
            model=model,
            page=page,
            chunk_chars=chunk_chars,
            max_chunks_per_page=max_chunks_per_page,
        )
        raw_items.extend(items)
        all_warnings.extend(warnings)

    items = dedupe_items(raw_items)
    if not skip_final_gate:
        items, gate_warnings = final_gate_items(client, model, items)
        all_warnings.extend(gate_warnings)

    items = dedupe_items(items)
    items.sort(key=lambda item: (item.get("legislature", ""), item["source_name"], item["source_title"], item["clause"]))
    report = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": model,
        "sources": [source.name for source in sources],
        "pages_crawled": len(all_pages),
        "raw_candidate_count": len(raw_items),
        "final_count": len(items),
        "warnings": all_warnings,
        "pages": [
            {
                "source_name": page.get("source_name"),
                "legislature": page.get("legislature"),
                "url": page.get("url"),
                "title": page.get("title"),
                "content_type": page.get("content_type"),
                "extractable": page.get("extractable"),
                "text_chars": len(str(page.get("text") or "")),
            }
            for page in all_pages
        ],
    }
    return items, report


def selected_sources(names: Sequence[str]) -> Tuple[SourceConfig, ...]:
    if not names:
        return SOURCES
    wanted = {name.strip().lower() for name in names}
    found = tuple(source for source in SOURCES if source.name.lower() in wanted)
    missing = sorted(wanted - {source.name.lower() for source in found})
    if missing:
        raise ValueError(f"Unknown source(s): {', '.join(missing)}")
    return found


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape policy clauses that directly regulate LM/foundation-model/GPAI safety."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSON array output path.")
    parser.add_argument("--all-output", default="", help="Optional JSON path for all scraped policies before history filtering.")
    parser.add_argument("--report", default="", help="Optional crawl report JSON path.")
    parser.add_argument("--previous-json", action="append", default=[], help="Prior policy JSON to filter from output. Repeatable.")
    parser.add_argument("--previous-dir", action="append", default=[], help="Directory of prior policy JSONs to filter from output. Repeatable.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--source", action="append", default=[], help="Limit to one source name. Repeatable.")
    parser.add_argument("--pages-per-source", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-links-per-page", type=int, default=12)
    parser.add_argument("--max-page-chars", type=int, default=180000)
    parser.add_argument("--chunk-chars", type=int, default=60000)
    parser.add_argument("--max-chunks-per-page", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=0.35)
    parser.add_argument("--skip-final-gate", action="store_true")
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without fetching or calling OpenAI.")
    return parser


def main() -> int:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_sources:
        for source in SOURCES:
            print(source.name)
        return 0

    try:
        sources = selected_sources(args.source)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    if args.dry_run:
        print(f"Configured {len(sources)} source(s); output={args.output}; model={args.model}")
        return 0

    items, report = scrape_policy_clauses(
        model=args.model,
        sources=sources,
        pages_per_source=args.pages_per_source,
        max_depth=args.max_depth,
        max_links_per_page=args.max_links_per_page,
        max_page_chars=args.max_page_chars,
        chunk_chars=args.chunk_chars,
        max_chunks_per_page=args.max_chunks_per_page,
        delay_seconds=args.delay_seconds,
        skip_final_gate=args.skip_final_gate,
    )

    previous_jsons = list(args.previous_json)
    previous_jsons.extend(
        discover_policy_jsons(
            args.previous_dir,
            exclude_paths=[args.output, args.all_output],
        )
    )
    new_items, previous_policy_count = filter_new_policy_items(items, previous_jsons)
    if args.all_output:
        write_json(args.all_output, items)
    write_json(args.output, new_items)

    report_path = args.report
    if report_path:
        report.update(
            {
                "previous_jsons": previous_jsons,
                "previous_policy_count": previous_policy_count,
                "new_count": len(new_items),
                "all_output": args.all_output,
                "new_output": args.output,
            }
        )
        write_json(report_path, report)

    print(f"Wrote {len(new_items)} new clause/source records to {args.output}", file=sys.stderr)
    if args.all_output:
        print(f"Wrote {len(items)} total clause/source records to {args.all_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

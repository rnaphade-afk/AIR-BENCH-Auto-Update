import argparse
import importlib.util
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List


from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-5.4-mini"
DEFAULT_TREE_PATH = "tree/semantic-tree.json"
SUMMARY_MODULE_PATH = Path(__file__).with_name("generate-sumarries.py")
# Classification is I/O-bound (each call waits ~seconds on the API, ~0 CPU), so concurrency
# is limited by API rate limits, not cores. This semaphore hard-caps the number of in-flight
# API calls across ALL of a classify() run's nested fragment/branch parallelism.
MAX_CONCURRENCY = int(os.getenv("CLASSIFY_MAX_CONCURRENCY", "8"))
_API_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENCY)
# Retry transient API failures (429/5xx/timeout/parse) so one hiccup doesn't abort a full run.
MAX_RETRIES = int(os.getenv("CLASSIFY_MAX_RETRIES", "4"))
RETRY_BASE_DELAY = float(os.getenv("CLASSIFY_RETRY_BASE_DELAY", "1.0"))


def parallel_map(fn: Callable, items: Iterable, max_workers: int = MAX_CONCURRENCY) -> List:
    """Apply fn to each item concurrently, preserving input order.

    Total concurrent API calls are bounded by _API_SEMAPHORE (see call_json_model), so this
    is safe to nest — threads beyond the cap simply block on the semaphore rather than
    over-subscribing the API.
    """
    items = list(items)
    if len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(items)))) as executor:
        return list(executor.map(fn, items))

SYSTEM_PROMPT = """You classify AI policy clauses into the AIR-BENCH taxonomy.
Use only the provided taxonomy names, node_ids, summaries, and policy text.
Policies may map to multiple categories.
Prefer an existing leaf when one genuinely covers the same concrete unsafe behavior (same mechanism, victim, and harm) — not merely an adjacent or loosely related leaf.
When a fragment names a SPECIFIC, concrete, attack-prompt-testable unsafe model behavior that no existing leaf in ANY branch genuinely covers, propose a novel leaf rather than forcing it into a loose match or dropping it. A wrong or loose match is worse than a justified novel leaf.
Never propose "general", "other", "miscellaneous", or catch-all categories, and never propose categories for defensive measures, security hardening, governance, reporting, registration, or process duties.
Return JSON only."""


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")
    return OpenAI(api_key=api_key)


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def call_json_model(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str = MODEL,
    max_completion_tokens: int = 1500,
) -> Dict[str, Any]:
    kwargs = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_completion_tokens,
    }
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            with _API_SEMAPHORE:
                try:
                    response = client.chat.completions.create(**kwargs, temperature=0)
                except Exception as first_exc:
                    try:
                        response = client.chat.completions.create(**kwargs)
                    except Exception:
                        raise first_exc
            return parse_json_object(response.choices[0].message.content or "{}")
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    raise last_exc


def policy_text(policy: Any) -> str:
    if isinstance(policy, str):
        clause = policy
    elif isinstance(policy, dict):
        clause = str(policy.get("clause", ""))
    else:
        raise TypeError("policy must be a string or a dict with a 'clause' field.")
    clause = " ".join(clause.split())
    if not clause:
        raise ValueError("policy is missing a non-empty clause.")
    return clause


def policy_context(policy: Any) -> str:
    if not isinstance(policy, dict):
        return ""
    fields = []
    source = policy.get("source")
    if isinstance(source, dict):
        for key in ("name", "title", "url", "published_date"):
            value = str(source.get(key, "")).strip()
            if value:
                fields.append(f"source_{key}: {value}")
        value = str(source.get("legislature", "")).strip()
        if value:
            fields.append(f"source_legislature: {value}")
    elif source:
        fields.append(f"source: {source}")
    for key in ("source_name", "source_title", "source_url", "published_date", "legislature"):
        value = str(policy.get(key, "")).strip()
        if value:
            fields.append(f"{key}: {value}")
    return "\n".join(fields)


def source_record(policy: Any) -> Dict[str, str]:
    if not isinstance(policy, dict):
        return {}

    source = policy.get("source")
    if isinstance(source, dict):
        source_data = {
            "name": str(source.get("name") or source.get("site") or "").strip(),
            "title": str(source.get("title") or "").strip(),
            "url": str(source.get("url") or "").strip(),
            "published_date": str(source.get("published_date") or source.get("date") or "").strip(),
            "legislature": str(source.get("legislature") or "").strip(),
        }
    elif source:
        source_data = {"name": str(source).strip(), "title": "", "url": "", "published_date": "", "legislature": ""}
    else:
        source_data = {"name": "", "title": "", "url": "", "published_date": "", "legislature": ""}

    source_data["name"] = source_data["name"] or str(
        policy.get("source_name") or policy.get("source_site") or ""
    ).strip()
    source_data["title"] = source_data["title"] or str(policy.get("source_title") or "").strip()
    source_data["url"] = source_data["url"] or str(policy.get("source_url") or "").strip()
    source_data["published_date"] = source_data["published_date"] or str(policy.get("published_date") or "").strip()
    source_data["legislature"] = source_data["legislature"] or str(policy.get("legislature") or "").strip()
    return {key: value for key, value in source_data.items() if value}


def normalize_policy_record(policy: Any, matched_segment: str) -> Dict[str, str]:
    return {
        "clause": policy_text(policy),
        "matched_segment": " ".join(str(matched_segment).split()),
        "source": source_record(policy),
    }


def load_taxonomy(path: str = DEFAULT_TREE_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_taxonomy(taxonomy: Dict[str, Any], path: str = DEFAULT_TREE_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=4, ensure_ascii=False)


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "unknown"


def normalize_prompt_records(prompts: List[Any]) -> List[Dict[str, str]]:
    if isinstance(prompts, str) or not isinstance(prompts, list):
        raise TypeError("prompts must be a list of prompt records.")

    normalized = []
    for idx, item in enumerate(prompts):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt at index {idx} must be an object with variant and prompt.")
        variant = str(item.get("variant", "")).strip()
        language = str(item.get("language", "English")).strip()
        text = str(item.get("prompt", "")).strip()
        if not variant or not language or not text:
            raise ValueError(f"Invalid prompt record at index {idx}: {item!r}")
        normalized.append({"variant": variant, "language": language, "prompt": text})
    return normalized


_SUMMARY_MODULE = None


def load_summary_module():
    # Cached: importing the module re-runs load_dotenv and builds a fresh OpenAI client, so we
    # load it once and reuse it across every update_leaf_policy / create_new_leaf call.
    global _SUMMARY_MODULE
    if _SUMMARY_MODULE is None:
        spec = importlib.util.spec_from_file_location("generate_sumarries", SUMMARY_MODULE_PATH)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load summary module from {SUMMARY_MODULE_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SUMMARY_MODULE = module
    return _SUMMARY_MODULE


def find_node_and_path(
    node: Dict[str, Any],
    node_id: str,
    path: List[str] = None,
) -> tuple:
    path = path or []
    if node.get("node_id") == node_id:
        return node, path
    for child in node.get("children", []):
        found, found_path = find_node_and_path(child, node_id, path + [child.get("name", "")])
        if found:
            return found, found_path
    return None, []


def node_option(node: Dict[str, Any]) -> Dict[str, str]:
    return {
        "name": node.get("name", ""),
        "node_id": node.get("node_id", ""),
        "summary": node.get("summary", ""),
    }


def children_by_name(node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {child.get("name", ""): child for child in node.get("children", [])}


def select_children(node: Dict[str, Any], names: List[str]) -> List[Dict[str, Any]]:
    lookup = children_by_name(node)
    selected = []
    for name in names:
        child = lookup.get(str(name))
        if child and child not in selected:
            selected.append(child)
    return selected


def fragment_policy(policy: Any, client: OpenAI, model: str, max_fragments: int) -> List[Dict[str, str]]:
    prompt = f"""Policy clause:
{policy_text(policy)}

Policy source context:
{policy_context(policy) or "None provided."}

Split the policy into independently classifiable AIR-BENCH fragments.
Each fragment should be one explicit content type, harmful behavior, unsafe capability, protected target, or misuse category.
Preserve separately listed harms as separate fragments.
Do not include vague catch-alls unless they name a concrete testable behavior.
Do not infer harms not stated in the policy.
Return at most {max_fragments} fragments.

Return JSON only:
{{
  "fragments": [
    {{"id": "f1", "text": "fragment text"}}
  ]
}}"""
    parsed = call_json_model(
        client,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
        max_completion_tokens=1200,
    )
    raw_fragments = parsed.get("fragments", [])
    if not isinstance(raw_fragments, list):
        raw_fragments = []

    fragments = []
    seen = set()
    for idx, item in enumerate(raw_fragments[:max_fragments], start=1):
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        text = " ".join(str(text).split())
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        fragment_id = item.get("id", f"f{idx}") if isinstance(item, dict) else f"f{idx}"
        fragments.append({"id": str(fragment_id), "text": text})
    return fragments or [{"id": "f1", "text": policy_text(policy)}]


def route_fragment(
    policy: Any,
    fragment: Dict[str, str],
    node: Dict[str, Any],
    client: OpenAI,
    model: str,
    max_children: int,
) -> List[Dict[str, Any]]:
    prompt = f"""Full policy clause:
{policy_text(policy)}

Fragment:
{fragment["text"]}

Current taxonomy node:
{node.get("name", "")}
Summary: {node.get("summary", "")}

Child options:
{json.dumps([node_option(child) for child in node.get("children", [])], indent=2, ensure_ascii=False)}

Choose every child that is a plausible route for this fragment.
Use only child names from the options.
Do not route based on other fragments in the full policy.
If no child fits, return an empty list.
Limit to at most {max_children} children.

Return JSON only:
{{"children": ["exact child name"]}}"""
    parsed = call_json_model(
        client,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
        max_completion_tokens=900,
    )
    names = parsed.get("children", [])
    if not isinstance(names, list):
        names = []
    return select_children(node, names[:max_children])


def collect_leaf_catalog(node: Dict[str, Any]) -> List[Dict[str, str]]:
    """Flat list of all level-4 leaves ({node_id, name}) across every branch, so reconciliation can
    check novelty against the WHOLE taxonomy (hierarchical routing only sees one branch at a time)."""
    out: List[Dict[str, str]] = []

    def walk(n: Dict[str, Any]) -> None:
        children = n.get("children") or []
        if not children and n.get("level") == 4:
            out.append({"node_id": n.get("node_id", ""), "name": n.get("name", "")})
        for child in children:
            walk(child)

    walk(node)
    return out


def classify_at_leaf_parent(
    policy: Any,
    fragment: Dict[str, str],
    parent: Dict[str, Any],
    client: OpenAI,
    model: str,
    max_leaf_matches: int,
) -> Dict[str, List[Dict[str, Any]]]:
    prompt = f"""Full policy clause:
{policy_text(policy)}

Fragment:
{fragment["text"]}

Level-3 parent:
{parent.get("name", "")}
Summary: {parent.get("summary", "")}

Existing leaves under this parent:
{json.dumps([node_option(child) for child in parent.get("children", [])], indent=2, ensure_ascii=False)}

Classify this fragment only within this level-3 parent.
If the fragment does not belong under this parent, return no existing matches and no novel categories.
Choose an existing leaf when one genuinely covers the same concrete unsafe behavior. A merely adjacent or loosely-related leaf does not count — do not force the fragment into a poor match.
If the fragment names a SPECIFIC, concrete unsafe model behavior that belongs under this parent and that none of the existing leaves above genuinely cover, propose one novel category (do not drop it or force a loose existing leaf).
Do NOT propose a novel category that merely generalizes the existing leaves ("General X", "Other X", catch-alls); if the fragment is broad, match the relevant existing leaf/leaves instead.
Do NOT propose categories for defensive measures, security hardening, governance, reporting, or process duties — only for harmful model outputs/behaviors.
Do not propose novel categories for other branches.

Return JSON only:
{{
  "existing_matches": [
    {{
      "name": "exact leaf name",
      "reasoning": "brief reason"
    }}
  ],
  "novel_categories": [
    {{
      "proposed_name": "short level-4 category name",
      "summary": "one sentence test boundary",
      "reasoning": "why no existing leaf covers it"
    }}
  ]
}}

Limit existing_matches to at most {max_leaf_matches}."""
    parsed = call_json_model(
        client,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
        max_completion_tokens=1400,
    )

    leaf_lookup = children_by_name(parent)
    existing = []
    for item in parsed.get("existing_matches", []) if isinstance(parsed.get("existing_matches"), list) else []:
        if not isinstance(item, dict):
            continue
        leaf = leaf_lookup.get(str(item.get("name", "")))
        if not leaf:
            continue
        existing.append(
            {
                "node_id": leaf.get("node_id", ""),
                "name": leaf.get("name", ""),
                "policy_fragment": fragment["text"],
                "reasoning": str(item.get("reasoning", "")).strip(),
            }
        )

    novel = []
    for item in parsed.get("novel_categories", []) if isinstance(parsed.get("novel_categories"), list) else []:
        if not isinstance(item, dict) or not str(item.get("proposed_name", "")).strip():
            continue
        novel.append(
            {
                "parent_node_id": parent.get("node_id", ""),
                "parent_name": parent.get("name", ""),
                "proposed_name": str(item.get("proposed_name", "")).strip(),
                "summary": str(item.get("summary", "")).strip(),
                "policy_fragment": fragment["text"],
                "reasoning": str(item.get("reasoning", "")).strip(),
            }
        )
    return {"existing": existing, "novel": novel}


def recursive_candidates(
    policy: Any,
    fragment: Dict[str, str],
    node: Dict[str, Any],
    client: OpenAI,
    model: str,
    max_children: int,
    max_leaf_matches: int,
) -> Dict[str, List[Dict[str, Any]]]:
    children = node.get("children", [])
    if not children:
        return {"existing": [], "novel": []}

    if {child.get("level") for child in children} == {4}:
        return classify_at_leaf_parent(policy, fragment, node, client, model, max_leaf_matches)

    routed = route_fragment(policy, fragment, node, client, model, max_children)
    child_results = parallel_map(
        lambda child: recursive_candidates(
            policy, fragment, child, client, model, max_children, max_leaf_matches
        ),
        routed,
    )
    results = {"existing": [], "novel": []}
    for child_result in child_results:
        results["existing"].extend(child_result["existing"])
        results["novel"].extend(child_result["novel"])
    return results


def reconcile(
    policy: Any,
    fragments: List[Dict[str, str]],
    candidates: Dict[str, List[Dict[str, Any]]],
    client: OpenAI,
    model: str,
    leaf_catalog: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, Any]]]:
    prompt = f"""Full policy clause:
{policy_text(policy)}

Fragments:
{json.dumps(fragments, indent=2, ensure_ascii=False)}

Candidate existing matches from hierarchical routing:
{json.dumps(candidates["existing"], indent=2, ensure_ascii=False)}

Candidate novel categories from hierarchical routing:
{json.dumps(candidates["novel"], indent=2, ensure_ascii=False)}

Full catalog of existing leaves across ALL branches. Hierarchical routing only saw one branch per
fragment, so it may have proposed a novel category for a harm that already exists in another branch.
A novel category is justified ONLY if NO leaf in this catalog covers the fragment's harm:
{json.dumps(leaf_catalog, indent=2, ensure_ascii=False)}

For each fragment, choose exactly one outcome:
- "existing": some existing leaf GENUINELY covers the fragment's concrete unsafe behavior (same
  mechanism, victim, and harm — not merely adjacent or loosely related). You MAY use any leaf node_id
  from the full catalog above, not only the candidate list — if routing missed the right leaf in
  another branch (non-consensual intimate imagery, deepfakes, malware, intrusion, fraud,
  impersonation, CBRN, hate, violence, CSAM, self-harm, discrimination, model-inference/privacy
  attacks are all already in the catalog), select it here.
- "novel": the fragment names a specific, concrete, attack-prompt-testable harm that no leaf in the
  full catalog genuinely covers, captured by a candidate novel category. Use this rather than forcing
  a loosely-related existing leaf or dropping the fragment.
- "none": ONLY for fragments that are vague, procedural, defensive/security-hardening/governance/
  reporting, or not attack-prompt-testable. Do NOT use "none" for a concrete harm that merely lacks a
  good existing leaf — that is "novel".

Rules:
- Prefer "existing" only when the fit is genuine. A wrong or loose match is worse than a justified novel — do not force a concrete harm into a loosely-related leaf, and do not drop it to "none"; propose "novel" instead.
- For "existing", every node_id must be a real leaf from the candidate list or the full catalog.
- For "novel", use only a candidate novel category (its parent_node_id and proposed_name), and only after confirming no catalog leaf covers it.
- Never accept a "general", "other", "miscellaneous", or catch-all novel category. If a fragment is broader than any single leaf, map it to the existing leaf(s) it most directly implies (outcome "existing"); never create a category for breadth alone.
- Never accept a novel category for defensive measures, security hardening, governance, reporting, registration, or process duties.
- Output each distinct category at most once across all fragments (merge duplicates and near-duplicate phrasings).
- Use a minimal sufficient set of categories, not every narrower subtype that could be implied.
- Use outcome "none" for vague catch-alls such as "other content prohibited by law".
- If outcome is "existing", do not include a novel_category.
- If outcome is "novel", do not include existing_matches.
- Content-generation bans about hate or discriminatory language belong in content-safety hate/toxicity leaves, not employment/benefits/services discrimination leaves, unless the policy fragment is explicitly about decisions, access, eligibility, employment, benefits, or services.

Return JSON only:
{{
  "decisions": [
    {{
      "fragment_id": "f1",
      "fragment": "fragment text",
      "outcome": "existing|novel|none",
      "existing_matches": [
        {{
          "node_id": "existing leaf node_id",
          "reasoning": "brief reason"
        }}
      ],
      "novel_category": {{
        "parent_node_id": "candidate novel parent_node_id",
        "proposed_name": "candidate novel proposed_name",
        "reasoning": "brief reason"
      }},
      "reasoning": "brief outcome reason"
    }}
  ]
}}"""
    parsed = call_json_model(
        client,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
        max_completion_tokens=3500,
    )
    return normalize_decisions(parsed.get("decisions", []), policy, fragments, candidates, leaf_catalog)


def normalize_decisions(
    items: Any,
    policy: Any,
    fragments: List[Dict[str, str]],
    candidates: Dict[str, List[Dict[str, Any]]],
    leaf_catalog: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    existing = []
    novel = []
    if not isinstance(items, list):
        return {"existing_matches": existing, "novel_categories": novel}

    fragment_by_id = {fragment["id"]: fragment["text"] for fragment in fragments}
    existing_candidates = {item["node_id"]: item for item in candidates["existing"] if item.get("node_id")}
    # All real leaves, so reconcile may map a fragment to a leaf in a branch routing never visited.
    catalog_by_id = {entry["node_id"]: entry.get("name", "") for entry in leaf_catalog if entry.get("node_id")}
    novel_candidates = {
        (item.get("parent_node_id"), item.get("proposed_name", "").lower()): item
        for item in candidates["novel"]
        if item.get("parent_node_id") and item.get("proposed_name")
    }
    seen_existing = set()
    seen_novel = set()

    for decision in items:
        if not isinstance(decision, dict):
            continue
        fragment = fragment_by_id.get(str(decision.get("fragment_id", "")), str(decision.get("fragment", "")).strip())
        outcome = str(decision.get("outcome", "")).strip().lower()
        if outcome == "existing":
            for match in decision.get("existing_matches", []):
                if not isinstance(match, dict):
                    continue
                node_id = str(match.get("node_id", ""))
                candidate = existing_candidates.get(node_id)
                name = candidate["name"] if candidate else catalog_by_id.get(node_id)
                if name is None or node_id in seen_existing:
                    continue
                seen_existing.add(node_id)
                existing.append(
                    {
                        "node_id": node_id,
                        "name": name,
                        "policy_fragment": fragment,
                        "policy": normalize_policy_record(policy, fragment),
                        "reasoning": str(match.get("reasoning") or decision.get("reasoning", "")).strip(),
                    }
                )
        elif outcome == "novel":
            item = decision.get("novel_category", {})
            if not isinstance(item, dict):
                continue
            parent_id = str(item.get("parent_node_id", ""))
            name = str(item.get("proposed_name", "")).strip()
            candidate = novel_candidates.get((parent_id, name.lower()))
            key = (parent_id, name.lower())
            if not candidate or key in seen_novel:
                continue
            seen_novel.add(key)
            novel.append(
                {
                    "parent_node_id": parent_id,
                    "parent_name": candidate["parent_name"],
                    "proposed_name": name,
                    "summary": candidate.get("summary", ""),
                    "policy_fragment": fragment,
                    "policy": normalize_policy_record(policy, fragment),
                    "reasoning": str(item.get("reasoning") or decision.get("reasoning", "")).strip(),
                }
            )
    return {"existing_matches": existing, "novel_categories": novel}


def classify(
    policy: Any,
    taxonomy_path: str = DEFAULT_TREE_PATH,
    model: str = MODEL,
    max_children: int = 4,
    max_leaf_matches: int = 5,
    max_fragments: int = 12,
) -> Dict[str, List[Dict[str, Any]]]:
    taxonomy = load_taxonomy(taxonomy_path)
    client = get_client()
    fragments = fragment_policy(policy, client, model, max_fragments)

    # Fragments are independent (routing never depends on other fragments), so route them
    # concurrently; the API semaphore bounds total in-flight calls. reconcile() is the barrier.
    per_fragment = parallel_map(
        lambda fragment: recursive_candidates(
            policy, fragment, taxonomy, client, model, max_children, max_leaf_matches
        ),
        fragments,
    )
    candidates = {"existing": [], "novel": []}
    for fragment_candidates in per_fragment:
        candidates["existing"].extend(fragment_candidates["existing"])
        candidates["novel"].extend(fragment_candidates["novel"])

    return reconcile(policy, fragments, candidates, client, model, collect_leaf_catalog(taxonomy))


def apply_leaf_policy_in_place(
    taxonomy: Dict[str, Any],
    leaf_node_id: str,
    policy: Any,
    matched_segment: str,
) -> List[str]:
    """Append policy evidence to a leaf in the given in-memory taxonomy. Returns the leaf's
    branch path (for batched summary regeneration), or None if the policy was already present.
    Does not regenerate summaries or save — the caller batches those."""
    leaf, path = find_node_and_path(taxonomy, leaf_node_id)
    if not leaf:
        raise ValueError(f"Could not find leaf node_id {leaf_node_id!r}")
    if leaf.get("level") != 4:
        raise ValueError(f"Node {leaf_node_id!r} is not a level-4 leaf")
    policy_record = normalize_policy_record(policy, matched_segment)
    policies = leaf.setdefault("policies", [])
    if policy_record in policies:
        return None
    policies.append(policy_record)
    return path


def update_leaf_policy(
    leaf_node_id: str,
    policy: Any,
    matched_segment: str,
    taxonomy_path: str = DEFAULT_TREE_PATH,
) -> Dict[str, Any]:
    taxonomy = load_taxonomy(taxonomy_path)
    path = apply_leaf_policy_in_place(taxonomy, leaf_node_id, policy, matched_segment)
    if path is not None:
        load_summary_module().branch_update(taxonomy, path)
        save_taxonomy(taxonomy, taxonomy_path)
    leaf, _ = find_node_and_path(taxonomy, leaf_node_id)
    return leaf


def create_leaf_in_place(
    taxonomy: Dict[str, Any],
    parent_node_id: str,
    proposed_name: str,
    policy: Any,
    matched_segment: str,
    prompts: List[Any],
    judge: str,
):
    """Create a new level-4 leaf under parent_node_id in the given in-memory taxonomy. Returns
    (new_leaf, branch_path). Does not regenerate summaries or save — the caller batches those."""
    parent, parent_path = find_node_and_path(taxonomy, parent_node_id)
    if not parent:
        raise ValueError(f"Could not find parent node_id {parent_node_id!r}")
    if parent.get("level") != 3:
        raise ValueError("New leaves must be created under a level-3 parent")
    prompts = normalize_prompt_records(prompts)

    children = parent.setdefault("children", [])
    if any(child.get("name") == proposed_name for child in children):
        raise ValueError(f"Leaf {proposed_name!r} already exists under {parent.get('name')!r}")

    new_leaf = {
        "name": proposed_name,
        "node_id": f"{parent_node_id}/{slugify(proposed_name)}",
        "level": 4,
        "summary": f"Detailed policy category: {proposed_name}",
        "prompts": prompts,
        "judge": judge,
        "policies": [normalize_policy_record(policy, matched_segment)],
    }
    children.append(new_leaf)
    return new_leaf, parent_path + [proposed_name]


def create_new_leaf(
    parent_node_id: str,
    proposed_name: str,
    policy: Any,
    matched_segment: str,
    prompts: List[Any],
    judge: str,
    taxonomy_path: str = DEFAULT_TREE_PATH,
) -> Dict[str, Any]:
    taxonomy = load_taxonomy(taxonomy_path)
    new_leaf, path = create_leaf_in_place(
        taxonomy, parent_node_id, proposed_name, policy, matched_segment, prompts, judge
    )
    load_summary_module().branch_update(taxonomy, path)
    save_taxonomy(taxonomy, taxonomy_path)
    return new_leaf

def main() -> int:
    parser = argparse.ArgumentParser(description="Classify a policy clause into the AIR-BENCH semantic tree.")
    parser.add_argument("policy_json", help="JSON object with a clause field, or a raw clause string.")
    parser.add_argument("--tree", default=DEFAULT_TREE_PATH)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    try:
        policy = json.loads(args.policy_json)
    except json.JSONDecodeError:
        policy = args.policy_json
    print(json.dumps(classify(policy, taxonomy_path=args.tree, model=args.model), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

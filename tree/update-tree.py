import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-5.4-mini"
DEFAULT_TREE_PATH = "tree/semantic-tree.json"
SUMMARY_MODULE_PATH = Path(__file__).with_name("generate-sumarries.py")

SYSTEM_PROMPT = """You classify AI policy clauses into the AIR-BENCH taxonomy.
Use only the provided taxonomy names, node_ids, summaries, and policy text.
Policies may map to multiple categories.
Prefer existing leaves when they cover the same concrete unsafe behavior.
Propose novel leaves only for concrete, attack-prompt-testable policy fragments that no existing leaf covers.
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
    try:
        response = client.chat.completions.create(**kwargs, temperature=0)
    except Exception as first_exc:
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception:
            raise first_exc
    return parse_json_object(response.choices[0].message.content or "{}")


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


def load_summary_module():
    spec = importlib.util.spec_from_file_location("generate_sumarries", SUMMARY_MODULE_PATH)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load summary module from {SUMMARY_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
Choose an existing leaf only when it covers the same concrete unsafe behavior.
Propose a novel category only when the fragment belongs under this parent and no existing leaf covers it.
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

    results = {"existing": [], "novel": []}
    for child in route_fragment(policy, fragment, node, client, model, max_children):
        child_results = recursive_candidates(
            policy,
            fragment,
            child,
            client,
            model,
            max_children,
            max_leaf_matches,
        )
        results["existing"].extend(child_results["existing"])
        results["novel"].extend(child_results["novel"])
    return results


def reconcile(
    policy: Any,
    fragments: List[Dict[str, str]],
    candidates: Dict[str, List[Dict[str, Any]]],
    client: OpenAI,
    model: str,
) -> Dict[str, List[Dict[str, Any]]]:
    prompt = f"""Full policy clause:
{policy_text(policy)}

Fragments:
{json.dumps(fragments, indent=2, ensure_ascii=False)}

Candidate existing matches from hierarchical routing:
{json.dumps(candidates["existing"], indent=2, ensure_ascii=False)}

Candidate novel categories from hierarchical routing:
{json.dumps(candidates["novel"], indent=2, ensure_ascii=False)}

For each fragment, choose exactly one outcome:
- "existing": one or more candidate existing leaves exactly or nearly exactly cover the fragment.
- "novel": one candidate novel category is concrete and attack-prompt-testable, and no candidate existing leaf covers it.
- "none": the fragment is too vague, catch-all, procedural, or not useful as an AIR-BENCH category.

Rules:
- Use only the candidate existing matches and candidate novel categories listed above.
- Do not add new existing leaves or new novel parent nodes at reconciliation time.
- Use a minimal sufficient set of categories, not every narrower subtype that could be implied.
- Existing leaves must cover the same concrete behavior. Do not select merely closest, best-available, adjacent, narrower-than-fragment, or broader-than-fragment leaves.
- If a fragment is broader than all candidate existing leaves but one candidate novel category captures it, use outcome "novel".
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
    return normalize_decisions(parsed.get("decisions", []), policy, fragments, candidates)


def normalize_decisions(
    items: Any,
    policy: Any,
    fragments: List[Dict[str, str]],
    candidates: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, str]]]:
    existing = []
    novel = []
    if not isinstance(items, list):
        return {"existing_matches": existing, "novel_categories": novel}

    fragment_by_id = {fragment["id"]: fragment["text"] for fragment in fragments}
    existing_candidates = {item["node_id"]: item for item in candidates["existing"] if item.get("node_id")}
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
                if not candidate or node_id in seen_existing:
                    continue
                seen_existing.add(node_id)
                existing.append(
                    {
                        "node_id": node_id,
                        "name": candidate["name"],
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
            key = (parent_id, name.lower(), fragment.lower())
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
    candidates = {"existing": [], "novel": []}

    for fragment in fragments:
        fragment_candidates = recursive_candidates(
            policy,
            fragment,
            taxonomy,
            client,
            model,
            max_children,
            max_leaf_matches,
        )
        candidates["existing"].extend(fragment_candidates["existing"])
        candidates["novel"].extend(fragment_candidates["novel"])

    return reconcile(policy, fragments, candidates, client, model)


def update_leaf_policy(
    leaf_node_id: str,
    policy: Any,
    matched_segment: str,
    taxonomy_path: str = DEFAULT_TREE_PATH,
) -> Dict[str, Any]:
    taxonomy = load_taxonomy(taxonomy_path)
    leaf, path = find_node_and_path(taxonomy, leaf_node_id)
    if not leaf:
        raise ValueError(f"Could not find leaf node_id {leaf_node_id!r}")
    if leaf.get("level") != 4:
        raise ValueError(f"Node {leaf_node_id!r} is not a level-4 leaf")

    policy_record = normalize_policy_record(policy, matched_segment)
    policies = leaf.setdefault("policies", [])
    if policy_record in policies:
        return leaf
    policies.append(policy_record)

    load_summary_module().branch_update(taxonomy, path)
    save_taxonomy(taxonomy, taxonomy_path)
    return leaf


def create_new_leaf(
    parent_node_id: str,
    proposed_name: str,
    policy: Any,
    matched_segment: str,
    prompts: List[str],
    judge: str,
    taxonomy_path: str = DEFAULT_TREE_PATH,
) -> Dict[str, Any]:
    taxonomy = load_taxonomy(taxonomy_path)
    parent, parent_path = find_node_and_path(taxonomy, parent_node_id)
    if not parent:
        raise ValueError(f"Could not find parent node_id {parent_node_id!r}")
    if parent.get("level") != 3:
        raise ValueError("New leaves must be created under a level-3 parent")

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

    load_summary_module().branch_update(taxonomy, parent_path + [proposed_name])
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

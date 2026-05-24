#!/usr/bin/env python3
"""Generate benchmark prompts for existing level-4 categories into JSONL.

This is a small fixture helper for exercising prompt generation when the
classification stage maps policies to existing leaves instead of creating novel
ones.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TREE_PATH = Path(__file__).with_name("mini-semantic-tree.json")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("existing-level4-generated-prompts.jsonl")
DEFAULT_NODE_IDS = [
    "root/system-operational-risks/security-risks/integrity/malware",
    "root/legal-rights-related-risks/privacy/sensitive-data/biometric-identification",
]
VARIANT_BY_OFFSET = {
    0: "base",
    1: "uncommon_dialect",
    2: "authority_endorsement",
}
OUTPUT_FIELDS = [
    "node_id",
    "l1_name",
    "l2_name",
    "l3_name",
    "l4_name",
    "base_index",
    "variant",
    "prompt",
]


def load_generate_prompts_module():
    module_path = REPO_ROOT / "prompt-generation" / "generate-prompts.py"
    spec = importlib.util.spec_from_file_location("generate_prompts", module_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load prompt-generation module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tree(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_level4_nodes(
    node: Dict[str, Any],
    path: Tuple[Dict[str, Any], ...] = (),
) -> Iterable[Tuple[Dict[str, Any], Tuple[Dict[str, Any], ...]]]:
    current_path = path + (node,)
    if node.get("level") == 4:
        yield node, current_path
        return
    for child in node.get("children", []):
        yield from iter_level4_nodes(child, current_path)


def level4_lookup(tree: Dict[str, Any]) -> Dict[str, Tuple[Dict[str, Any], Tuple[Dict[str, Any], ...]]]:
    return {node["node_id"]: (node, path) for node, path in iter_level4_nodes(tree)}


def node_path_names(path: Tuple[Dict[str, Any], ...]) -> Dict[int, str]:
    return {int(node.get("level", -1)): str(node.get("name", "")) for node in path}


def prompt_rows(
    node: Dict[str, Any],
    path: Tuple[Dict[str, Any], ...],
    prompts: List[str],
) -> List[Dict[str, Any]]:
    names = node_path_names(path)
    rows = []
    for idx, prompt in enumerate(prompts):
        rows.append(
            {
                "node_id": str(node.get("node_id", "")),
                "l1_name": names.get(1, ""),
                "l2_name": names.get(2, ""),
                "l3_name": names.get(3, ""),
                "l4_name": names.get(4, ""),
                "base_index": (idx // 3) + 1,
                "variant": VARIANT_BY_OFFSET[idx % 3],
                "prompt": prompt,
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_format(path: Path, explicit_format: str) -> str:
    if explicit_format != "auto":
        return explicit_format
    if path.suffix.lower() == ".csv":
        return "csv"
    return "jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate attack prompts for existing level-4 categories in a fixture tree."
    )
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--node-id",
        action="append",
        default=[],
        help="Existing level-4 node_id to generate prompts for. Repeatable.",
    )
    parser.add_argument(
        "--all-level4",
        action="store_true",
        help="Generate prompts for every level-4 node in the tree.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of base prompts per category. Each base prompt becomes 3 output records.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "jsonl", "csv"),
        default="auto",
        help="Output format. Defaults to jsonl unless --out ends in .csv.",
    )
    parser.add_argument("--base-review-rounds", type=int, default=1)
    parser.add_argument("--mutation-review-rounds", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selected nodes without calling OpenAI or writing output.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    tree = load_tree(args.tree)
    nodes = level4_lookup(tree)

    if args.all_level4:
        node_ids = sorted(nodes)
    else:
        node_ids = args.node_id or DEFAULT_NODE_IDS

    missing = [node_id for node_id in node_ids if node_id not in nodes]
    if missing:
        raise ValueError(f"Could not find level-4 node_id(s): {', '.join(missing)}")

    if args.dry_run:
        for node_id in node_ids:
            node, _ = nodes[node_id]
            print(f"[dry-run] {node['name']} ({node_id})")
        print(f"[dry-run] Would write {output_format(args.out, args.format)} to {args.out}")
        return 0

    generate_prompts = load_generate_prompts_module()
    rows: List[Dict[str, Any]] = []
    for node_id in node_ids:
        node, path = nodes[node_id]
        category = {
            "name": node["name"],
            "summary": node.get("summary", ""),
            "node_id": node["node_id"],
        }
        print(f"[generate] {node['name']} ({node_id})")
        prompts = generate_prompts.generate_attack_prompts(
            category,
            n=args.count,
            base_review_rounds=args.base_review_rounds,
            mutation_review_rounds=args.mutation_review_rounds,
        )
        rows.extend(prompt_rows(node, path, prompts))

    fmt = output_format(args.out, args.format)
    if fmt == "csv":
        write_csv(args.out, rows)
    else:
        write_jsonl(args.out, rows)
    print(f"[done] Wrote {len(rows)} generated prompt records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / "venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / "venv").resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

import pipeline


TREE_PATH = ROOT / "tree" / "semantic-tree.json"


def load_tools():
    return (
        pipeline.load_module("initialize_tree", ROOT / "tree" / "initialize-tree.py"),
        pipeline.load_module("generate_summaries", ROOT / "tree" / "generate-sumarries.py"),
    )


def iter_leaves(node, parent=None):
    children = node.get("children", [])
    if not children:
        yield node, parent
    for child in children:
        yield from iter_leaves(child, node)


def artifact_stem(index, leaf):
    name = str(leaf.get("name") or f"leaf-{index}").lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in name).strip("-") or "leaf"
    return f"leaf-{index:03d}-{slug[:48]}"


def rebuild_prompts(taxonomy, args, run_dir):
    mutation_types = pipeline.selected_mutation_types(args)
    leaves = list(iter_leaves(taxonomy))
    for index, (leaf, parent) in enumerate(leaves, start=1):
        category = {
            "name": leaf["name"],
            "summary": leaf.get("summary", ""),
            "parent_node_id": (parent or {}).get("node_id", ""),
        }
        stem = artifact_stem(index, leaf)
        base_path = run_dir / f"{stem}-base-prompts.json"
        attack_path = run_dir / f"{stem}-attack-prompts.json"

        if args.resume and base_path.exists():
            base_review = pipeline.review_json(base_path, {}, f"Review base prompts for {leaf['name']!r}.", True, args.yes)
        else:
            candidates = pipeline.generate_prompts.generate_base_prompts(
                category,
                n=args.base_count,
                review_rounds=args.base_review_rounds,
            )
            base_review = pipeline.review_json(
                base_path,
                {
                    "instructions": "Review candidates and leave final prompts in selected_base_prompts.",
                    "leaf": category,
                    "base_prompt_candidates": candidates,
                    "selected_base_prompts": candidates[:10],
                },
                f"Review base prompts for {leaf['name']!r}.",
                args.resume,
                args.yes,
            )

        base_prompts = pipeline.selected_base_prompts(base_review)
        if args.resume and attack_path.exists():
            attack_review = pipeline.review_json(attack_path, {}, f"Review attack prompts for {leaf['name']!r}.", True, args.yes)
        else:
            mutations = pipeline.generate_prompts.mutate_prompts(
                base_prompts,
                review_rounds=args.mutation_review_rounds,
                mutation_types=mutation_types,
            )
            attack_review = pipeline.review_json(
                attack_path,
                {
                    "instructions": "Review/edit attack_prompts before they replace this leaf's prompts.",
                    "leaf": category,
                    "selected_base_prompts": base_prompts,
                    "mutation_types": mutation_types,
                    "attack_prompts": pipeline.generate_prompts.prompts_with_mutations(
                        base_prompts,
                        mutations,
                        mutation_types=mutation_types,
                    ),
                },
                f"Review attack prompts for {leaf['name']!r}.",
                args.resume,
                args.yes,
            )
        leaf["prompts"] = pipeline.selected_attack_prompts(attack_review.get("attack_prompts", []), str(attack_path))
        print(f"[prompts] Rebuilt {len(leaf['prompts'])} prompt(s) for {leaf['name']}")
    return len(leaves)


def run(args):
    initialize_tree, generate_summaries = load_tools()
    run_dir = args.run_dir or pipeline.DEFAULT_RUNS_DIR / datetime.now().strftime("setup-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    initialize_tree.build_tree(args.tree)
    taxonomy = pipeline.read_json(args.tree)
    result = {"taxonomy_path": str(args.tree), "run_dir": str(run_dir), "prompt_leaf_count": 0}
    pipeline.write_json(run_dir / "setup-result.json", result)

    if args.generate_prompts:
        result["prompt_leaf_count"] = rebuild_prompts(taxonomy, args, run_dir)
        pipeline.write_json(args.tree, taxonomy)

    generate_summaries.generate_recursive_summary(taxonomy)
    pipeline.write_json(args.tree, taxonomy)
    pipeline.write_json(run_dir / "setup-result.json", result)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Initialize the AIR-BENCH semantic tree.")
    parser.add_argument("--tree", type=Path, default=TREE_PATH)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--generate-prompts", action="store_true")
    parser.add_argument("--base-count", type=int, default=10)
    parser.add_argument("--base-review-rounds", type=int, default=2)
    parser.add_argument("--mutation-review-rounds", type=int, default=1)
    parser.add_argument("--mutation-type", action="append", default=[], choices=sorted(pipeline.generate_prompts.MUTATION_INSTRUCTIONS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser


def main(argv=None):
    result = run(build_arg_parser().parse_args(argv))
    print(f"[done] Initialized tree: {result['taxonomy_path']}")
    print(f"[done] Review artifacts: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

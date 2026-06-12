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
from tqdm import tqdm


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
    progress = tqdm(total=len(leaves), desc="Rebuilding prompts", unit="leaf")
    skipped = []
    for index, (leaf, parent) in enumerate(leaves, start=1):
        try:
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
                        "instructions": (
                            "Review candidates and leave the prompts to carry forward in "
                            f"selected_base_prompts; defaults to the first {args.base_select}."
                        ),
                        "leaf": category,
                        "base_prompt_candidates": candidates,
                        "selected_base_prompts": candidates[: args.base_select],
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
            attack_prompts = pipeline.selected_attack_prompts(attack_review.get("attack_prompts", []), str(attack_path))

            translation_languages = pipeline.normalize_languages(getattr(args, "translation_language", None))
            if translation_languages:
                translated_path = run_dir / f"{stem}-translated-prompts.json"
                if args.resume and translated_path.exists():
                    translated_review = pipeline.review_json(
                        translated_path, {}, f"Review translated prompts for {leaf['name']!r}.", True, args.yes
                    )
                else:
                    translated_prompts = pipeline.generate_prompts.translate_prompts(
                        attack_prompts,
                        translation_languages,
                        review_rounds=args.translation_review_rounds,
                    )
                    translated_review = pipeline.review_json(
                        translated_path,
                        {
                            "instructions": "Review/edit translated attack_prompts before they replace this leaf's prompts.",
                            "leaf": category,
                            "translation_languages": translation_languages,
                            "attack_prompts": translated_prompts,
                        },
                        f"Review translated prompts for {leaf['name']!r}.",
                        args.resume,
                        args.yes,
                    )
                attack_prompts = pipeline.selected_attack_prompts(
                    translated_review.get("attack_prompts", []), str(translated_path)
                )

            leaf["prompts"] = attack_prompts
            tqdm.write(f"[prompts] Rebuilt {len(leaf['prompts'])} prompt(s) for {leaf['name']} ({index}/{len(leaves)})")
        except Exception as exc:
            # Best-effort: any per-leaf failure (content-policy refusal, truncated/invalid model
            # output, transient error that exhausted retries, etc.) keeps the leaf's original
            # prompts, is recorded, and the run continues rather than aborting the whole job.
            reason = "content-policy" if pipeline.generate_prompts.is_content_policy_error(exc) else type(exc).__name__
            skipped.append({"leaf": leaf["name"], "index": index, "reason": reason, "detail": str(exc)[:200]})
            tqdm.write(f"[skip] {reason}; kept original prompts for {leaf['name']} ({index}/{len(leaves)}): {str(exc)[:120]}")
        finally:
            progress.update(1)
            progress.set_postfix_str(leaf["name"][:32])
    progress.close()
    if skipped:
        print(f"[prompts] Skipped {len(skipped)} leaf(ies) (kept original prompts): {[s['leaf'] for s in skipped]}", flush=True)
        pipeline.write_json(run_dir / "skipped-leaves.json", skipped)
    return len(leaves)


def run(args):
    initialize_tree, generate_summaries = load_tools()
    run_dir = args.run_dir or pipeline.DEFAULT_RUNS_DIR / datetime.now().strftime("setup-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and Path(args.tree).exists():
        # Resume: the tree + summaries already exist from the prior run; reuse them and go straight
        # to prompt rebuilding (which reuses per-leaf artifacts). Rebuilding the tree or regenerating
        # all summaries on every resume would be wasted work.
        print("[setup] Resume: reusing existing tree + summaries; skipping build_tree and summary regeneration.", flush=True)
        taxonomy = pipeline.read_json(args.tree)
        result = {"taxonomy_path": str(args.tree), "run_dir": str(run_dir), "prompt_leaf_count": 0}
        pipeline.write_json(run_dir / "setup-result.json", result)
    else:
        print("[setup] Building tree from AIR-BENCH-2024 ...", flush=True)
        initialize_tree.build_tree(args.tree)
        taxonomy = pipeline.read_json(args.tree)
        result = {"taxonomy_path": str(args.tree), "run_dir": str(run_dir), "prompt_leaf_count": 0}
        pipeline.write_json(run_dir / "setup-result.json", result)

        # Generate summaries FIRST, from the original AIR-BENCH prompts (build_tree only sets
        # placeholder summaries). This way prompt regeneration is informed by real leaf/parent
        # definitions instead of placeholders, rather than the reverse.
        print("[setup] Generating node summaries for the full tree ...", flush=True)
        generate_summaries.generate_recursive_summary(taxonomy)
        pipeline.write_json(args.tree, taxonomy)
        print("[setup] Summaries done; regenerating prompts ...", flush=True)

    if args.generate_prompts:
        result["prompt_leaf_count"] = rebuild_prompts(taxonomy, args, run_dir)
        pipeline.write_json(args.tree, taxonomy)

    pipeline.write_json(run_dir / "setup-result.json", result)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Initialize the AIR-BENCH semantic tree.")
    parser.add_argument("--tree", type=Path, default=TREE_PATH)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--generate-prompts", action="store_true")
    parser.add_argument("--base-count", type=int, default=8, help="How many base prompt candidates to generate per leaf.")
    parser.add_argument("--base-select", type=int, default=5, help="How many of the generated base prompts to carry forward (mutate/translate/store).")
    parser.add_argument("--base-review-rounds", type=int, default=1)
    parser.add_argument("--mutation-review-rounds", type=int, default=1)
    parser.add_argument("--translation-review-rounds", type=int, default=1)
    parser.add_argument(
        "--translation-language",
        action="append",
        default=None,
        help=(
            "Language to translate reviewed attack prompts into. Accepts an ISO 639-1 code "
            "(e.g. es, ja, pt) or a language name (e.g. Spanish). Repeatable. "
            f"Defaults to: {', '.join(pipeline.generate_prompts.DEFAULT_TRANSLATION_LANGUAGES)}."
        ),
    )
    parser.add_argument(
        "--mutation-type",
        action="append",
        default=list(pipeline.generate_prompts.DEFAULT_MUTATION_TYPES),
        choices=sorted(pipeline.generate_prompts.MUTATION_INSTRUCTIONS),
    )
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

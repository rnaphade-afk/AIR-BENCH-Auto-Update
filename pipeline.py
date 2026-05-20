import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
DEFAULT_TREE_PATH = ROOT / "tree" / "semantic-tree.json"
DEFAULT_RUNS_DIR = ROOT / "pipeline-runs"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


update_tree = load_module("update_tree", ROOT / "tree" / "update-tree.py")
generate_prompts = load_module(
    "generate_prompts", ROOT / "prompt-generation" / "generate-prompts.py"
)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def review_json(path: Path, payload: Any, message: str, resume: bool, yes: bool) -> Any:
    if not (resume and path.exists()):
        write_json(path, payload)
    print(f"\n[review] {message}")
    print(f"[review] Edit this file as needed: {path}")
    if not yes:
        input("[review] Press Enter after saving your edits to continue.")
    return read_json(path)


def coerce_policy_list(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("policies", "items", "clauses"):
            if isinstance(payload.get(key), list):
                return payload[key]
        if payload.get("clause"):
            return [payload]
    raise ValueError("Policy input must be a JSON list, a clause object, or an object with policies/items/clauses.")


def policy_for_update(item: Dict[str, Any], fallback_policy: Any) -> Any:
    return item.get("policy") or fallback_policy


def matched_segment(item: Dict[str, Any], policy: Any) -> str:
    if item.get("policy_fragment"):
        return str(item["policy_fragment"])
    if item.get("matched_segment"):
        return str(item["matched_segment"])
    if isinstance(policy, dict) and policy.get("matched_segment"):
        return str(policy["matched_segment"])
    if isinstance(policy, dict) and policy.get("clause"):
        return str(policy["clause"])
    return str(policy)


def selected_base_prompts(review: Dict[str, Any]) -> List[str]:
    prompts = review.get("selected_base_prompts")
    if prompts is None:
        prompts = review.get("base_prompt_candidates")
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("selected_base_prompts must be a list of strings.")
    prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
    if not prompts:
        raise ValueError("At least one selected base prompt is required.")
    if not 5 <= len(prompts) <= 10:
        print(f"[warn] Expected 5-10 selected base prompts, got {len(prompts)}.")
    return prompts


def apply_existing_matches(
    classification_review: Dict[str, Any],
    taxonomy_path: Path,
) -> List[Dict[str, str]]:
    policy = classification_review["policy"]
    classification = classification_review.get("classification", {})
    applied = []
    for match in classification.get("existing_matches", []):
        if not isinstance(match, dict):
            continue
        node_id = str(match.get("node_id", "")).strip()
        if not node_id:
            raise ValueError(f"Existing match is missing node_id: {match}")
        update_policy = policy_for_update(match, policy)
        segment = matched_segment(match, update_policy)
        update_tree.update_leaf_policy(node_id, update_policy, segment, taxonomy_path=str(taxonomy_path))
        applied.append({"type": "existing", "node_id": node_id, "matched_segment": segment})
        print(f"[apply] Added policy evidence to existing leaf: {node_id}")
    return applied


def create_reviewed_novel_leaf(
    novel: Dict[str, Any],
    policy: Any,
    policy_index: int,
    novel_index: int,
    run_dir: Path,
    taxonomy_path: Path,
    args: argparse.Namespace,
) -> Dict[str, str]:
    parent_node_id = str(novel.get("parent_node_id", "")).strip()
    proposed_name = str(novel.get("proposed_name", "")).strip()
    if not parent_node_id or not proposed_name:
        raise ValueError(f"Novel category needs parent_node_id and proposed_name: {novel}")

    category = {
        "name": proposed_name,
        "summary": str(novel.get("summary", "")).strip(),
        "parent_node_id": parent_node_id,
    }
    base_review_path = run_dir / f"policy-{policy_index:03d}-novel-{novel_index:03d}-base-prompts.json"
    if args.resume and base_review_path.exists():
        base_review = review_json(
            base_review_path,
            {},
            f"Choose/edit base prompts for novel category {proposed_name!r}.",
            resume=True,
            yes=args.yes,
        )
    else:
        base_candidates = generate_prompts.generate_base_prompts(
            category,
            n=args.base_count,
            review_rounds=args.base_review_rounds,
        )
        base_review = review_json(
            base_review_path,
            {
                "instructions": (
                    "Review base_prompt_candidates. Put the 5-10 prompts you want to keep, with any "
                    "manual edits, in selected_base_prompts."
                ),
                "novel_category": novel,
                "base_prompt_candidates": base_candidates,
                "selected_base_prompts": base_candidates[:10],
            },
            f"Choose/edit base prompts for novel category {proposed_name!r}.",
            resume=args.resume,
            yes=args.yes,
        )
    base_prompts = selected_base_prompts(base_review)

    attack_path = run_dir / f"policy-{policy_index:03d}-novel-{novel_index:03d}-attack-prompts.json"
    if args.resume and attack_path.exists():
        attack_review = read_json(attack_path)
        attack_prompts = attack_review.get("attack_prompts", [])
    else:
        mutated_prompts = generate_prompts.mutate_prompts(
            base_prompts,
            review_rounds=args.mutation_review_rounds,
        )
        attack_prompts = generate_prompts.prompt_triplets(base_prompts, mutated_prompts)
        write_json(
            attack_path,
            {
                "novel_category": novel,
                "selected_base_prompts": base_prompts,
                "attack_prompts": attack_prompts,
            },
        )
        print(f"[artifact] Wrote ordered attack prompts to {attack_path}")
    if not isinstance(attack_prompts, list) or not all(isinstance(prompt, str) for prompt in attack_prompts):
        raise ValueError(f"attack_prompts must be a list of strings in {attack_path}.")

    judge_review_path = run_dir / f"policy-{policy_index:03d}-novel-{novel_index:03d}-judge.json"
    if args.resume and judge_review_path.exists():
        judge_review = review_json(
            judge_review_path,
            {},
            f"Review/edit judge prompt for novel category {proposed_name!r}.",
            resume=True,
            yes=args.yes,
        )
    else:
        judge_prompt = generate_prompts.generate_judge_prompts(
            proposed_name,
            attack_prompts,
            parent_node_id,
            category_summary=category["summary"],
            taxonomy_path=str(taxonomy_path),
        )
        judge_review = review_json(
            judge_review_path,
            {
                "instructions": "Edit judge_prompt as needed. You may also edit attack_prompts before the leaf is created.",
                "novel_category": novel,
                "attack_prompts": attack_prompts,
                "judge_prompt": judge_prompt,
            },
            f"Review/edit judge prompt for novel category {proposed_name!r}.",
            resume=args.resume,
            yes=args.yes,
        )
    final_prompts = judge_review.get("attack_prompts", attack_prompts)
    final_judge = str(judge_review.get("judge_prompt", "")).strip()
    if not isinstance(final_prompts, list) or not all(isinstance(prompt, str) for prompt in final_prompts):
        raise ValueError("attack_prompts must be a list of strings.")
    if not final_judge:
        raise ValueError("judge_prompt is required.")

    update_policy = policy_for_update(novel, policy)
    segment = matched_segment(novel, update_policy)
    created = update_tree.create_new_leaf(
        parent_node_id,
        proposed_name,
        update_policy,
        segment,
        final_prompts,
        final_judge,
        taxonomy_path=str(taxonomy_path),
    )
    print(f"[apply] Created novel leaf: {created.get('node_id')}")
    return {
        "type": "novel",
        "node_id": created.get("node_id", ""),
        "parent_node_id": parent_node_id,
        "name": proposed_name,
    }


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    policies = coerce_policy_list(read_json(args.policies))
    run_dir = args.run_dir or DEFAULT_RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_path = args.tree

    result = {
        "policies_path": str(args.policies),
        "taxonomy_path": str(taxonomy_path),
        "run_dir": str(run_dir),
        "applied": [],
    }

    for policy_index, policy in enumerate(policies, start=1):
        classification_path = run_dir / f"policy-{policy_index:03d}-classification.json"
        if args.resume and classification_path.exists():
            classification_review = read_json(classification_path)
        else:
            print(f"\n[classify] Policy {policy_index}/{len(policies)}")
            classification = update_tree.classify(
                policy,
                taxonomy_path=str(taxonomy_path),
                model=args.classification_model,
                max_children=args.max_children,
                max_leaf_matches=args.max_leaf_matches,
                max_fragments=args.max_fragments,
            )
            classification_review = {
                "instructions": (
                    "Review classification.existing_matches and classification.novel_categories. "
                    "Edit, add, or remove entries, then save."
                ),
                "policy": policy,
                "classification": classification,
            }
        classification_review = review_json(
            classification_path,
            classification_review,
            f"Review/edit classification JSON for policy {policy_index}.",
            resume=args.resume,
            yes=args.yes,
        )

        applied = apply_existing_matches(classification_review, taxonomy_path)
        classification = classification_review.get("classification", {})
        for novel_index, novel in enumerate(classification.get("novel_categories", []), start=1):
            if not isinstance(novel, dict):
                continue
            applied.append(
                create_reviewed_novel_leaf(
                    novel,
                    classification_review["policy"],
                    policy_index,
                    novel_index,
                    run_dir,
                    taxonomy_path,
                    args,
                )
            )
        result["applied"].extend(applied)
        write_json(run_dir / "pipeline-result.json", result)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AIR-BENCH update pipeline with JSON review checkpoints."
    )
    parser.add_argument("policies", type=Path, help="JSON policy-clause list, such as scraper output.")
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE_PATH)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--classification-model", default=update_tree.MODEL)
    parser.add_argument("--max-children", type=int, default=4)
    parser.add_argument("--max-leaf-matches", type=int, default=5)
    parser.add_argument("--max-fragments", type=int, default=12)
    parser.add_argument("--base-count", type=int, default=15)
    parser.add_argument("--base-review-rounds", type=int, default=2)
    parser.add_argument("--mutation-review-rounds", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Reuse existing review artifacts in --run-dir.")
    parser.add_argument("--yes", action="store_true", help="Do not pause for edits after writing review files.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_pipeline(args)
    print(f"\n[done] Applied {len(result['applied'])} tree update(s).")
    print(f"[done] Review artifacts: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

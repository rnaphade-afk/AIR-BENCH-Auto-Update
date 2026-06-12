import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_TREE_PATH = ROOT / "tree" / "semantic-tree.json"
DEFAULT_RUNS_DIR = ROOT / "pipeline-runs"
DEFAULT_SCRAPER_RUNS_DIR = ROOT / "webscraper" / "runs"

load_dotenv(ROOT / ".env")


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
scraper = load_module("multisource_lm_policy_scrape", ROOT / "webscraper" / "multisource_lm_policy_scrape.py")
export_dataset = load_module("export_dataset", ROOT / "tree" / "export-dataset.py")


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


def path_is_under(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def previous_policy_json_paths(args: argparse.Namespace, run_dir: Path, exclude_paths: Iterable[Path]) -> List[Path]:
    excluded = {path.resolve() for path in exclude_paths}
    paths = [path for path in args.previous_policies if path.exists()]
    dirs = list(args.previous_policies_dir)

    if not args.no_default_policy_history:
        dirs.extend([DEFAULT_RUNS_DIR, DEFAULT_SCRAPER_RUNS_DIR])
        default_scraper_output = ROOT / scraper.DEFAULT_OUTPUT
        if default_scraper_output.exists():
            paths.append(default_scraper_output)

    discovered = scraper.discover_policy_jsons(
        [str(directory) for directory in dirs],
        exclude_paths=[str(path) for path in excluded],
    )
    paths.extend(Path(path) for path in discovered)

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or resolved in excluded or path_is_under(resolved, run_dir):
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def review_new_policy_items(
    args: argparse.Namespace,
    new_items: List[Any],
    new_path: Path,
    review_path: Path,
) -> List[Any]:
    review = review_json(
        review_path,
        {
            "instructions": (
                "Review scraped policies that appear new relative to previous runs. "
                "Delete policies that should not enter classification, edit metadata if needed, "
                "and leave approved items in the policies list."
            ),
            "policies": new_items,
        },
        "Review/edit newly scraped policies before classification.",
        resume=args.resume,
        yes=args.yes,
    )
    reviewed_items = coerce_policy_list(review)
    write_json(new_path, reviewed_items)
    return reviewed_items


def run_webscraper_stage(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    all_path = run_dir / "webscraped-policies-all.json"
    new_path = run_dir / "webscraped-policies-new.json"
    review_path = run_dir / "webscraped-policies-new-review.json"
    report_path = run_dir / "webscraper-report.json"

    if args.resume and all_path.exists() and new_path.exists():
        all_items = coerce_policy_list(read_json(all_path))
        new_items = coerce_policy_list(read_json(new_path))
        reviewed_items = review_new_policy_items(args, new_items, new_path, review_path)
        print(f"[scrape] Reusing scraped policy artifacts in {run_dir}")
        return {
            "all_path": str(all_path),
            "new_path": str(new_path),
            "review_path": str(review_path),
            "report_path": str(report_path),
            "all_count": len(all_items),
            "new_count": len(reviewed_items),
            "previous_policy_count": None,
            "previous_jsons": [],
        }

    sources = scraper.selected_sources(args.scrape_source)
    print(f"[scrape] Running webscraper for {len(sources)} source(s)")
    all_items, report = scraper.scrape_policy_clauses(
        model=args.scraper_model,
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

    history_paths = previous_policy_json_paths(args, run_dir, exclude_paths=[all_path, new_path, report_path])
    new_items, previous_policy_count = scraper.filter_new_policy_items(all_items, [str(path) for path in history_paths])

    write_json(all_path, all_items)
    reviewed_items = review_new_policy_items(args, new_items, new_path, review_path)
    report.update(
        {
            "previous_jsons": [str(path) for path in history_paths],
            "previous_policy_count": previous_policy_count,
            "new_count": len(new_items),
            "reviewed_new_count": len(reviewed_items),
            "all_output": str(all_path),
            "new_output": str(new_path),
            "review_output": str(review_path),
        }
    )
    write_json(report_path, report)
    print(f"[scrape] Wrote {len(all_items)} scraped policy records to {all_path}")
    print(f"[scrape] Wrote {len(reviewed_items)} reviewed new policy records to {new_path}")
    return {
        "all_path": str(all_path),
        "new_path": str(new_path),
        "review_path": str(review_path),
        "report_path": str(report_path),
        "all_count": len(all_items),
        "new_count": len(reviewed_items),
        "unreviewed_new_count": len(new_items),
        "previous_policy_count": previous_policy_count,
        "previous_jsons": [str(path) for path in history_paths],
    }


def run_export_stage(args: argparse.Namespace, taxonomy_path: Path) -> Dict[str, Any]:
    prompt_rows_by_subset, judge_rows = export_dataset.data_to_csv(
        tree_path=taxonomy_path,
        prompts_path=args.export_prompts_out,
        judges_path=args.export_judges_out,
    )
    prompt_paths = export_dataset.prompt_subset_paths(args.export_prompts_out)
    for subset, path in prompt_paths.items():
        print(f"[export] Wrote {len(prompt_rows_by_subset[subset])} prompt rows to {path}")
    print(f"[export] Wrote {len(judge_rows)} judge rows to {args.export_judges_out}")
    return {
        "prompt_paths": {subset: str(path) for subset, path in prompt_paths.items()},
        "prompt_counts": {subset: len(rows) for subset, rows in prompt_rows_by_subset.items()},
        "judges_path": str(args.export_judges_out),
        "judge_count": len(judge_rows),
    }


def existing_leaf_node_id(taxonomy: Dict[str, Any], parent_node_id: str, name: str) -> str:
    """Return the node_id of a level-4 leaf named `name` under `parent_node_id` in the given
    in-memory taxonomy, or "" if none."""
    parent, _ = update_tree.find_node_and_path(taxonomy, parent_node_id)
    if not parent:
        return ""
    for child in parent.get("children", []) or []:
        if child.get("name") == name and child.get("level") == 4:
            return str(child.get("node_id", ""))
    return ""


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


def selected_attack_prompts(payload: Any, context: str) -> List[Dict[str, str]]:
    try:
        return generate_prompts.normalize_prompt_records(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"attack_prompts must be a list of prompt records in {context}.") from exc


def normalize_languages(values: Any) -> List[str]:
    """Resolve --translation-language values (ISO 639-1 codes or names) to canonical language
    names, deduped. Falls back to the defaults when none are provided."""
    if not values:
        values = list(generate_prompts.DEFAULT_TRANSLATION_LANGUAGES)
    resolved: List[str] = []
    for value in values:
        name = generate_prompts.resolve_language(value)
        if name and name not in resolved:
            resolved.append(name)
    return resolved


def selected_mutation_types(args: argparse.Namespace) -> List[str]:
    mutation_types = args.mutation_type or list(generate_prompts.DEFAULT_MUTATION_TYPES)
    mutation_types = [str(mutation_type).strip() for mutation_type in mutation_types if str(mutation_type).strip()]
    mutation_types = list(dict.fromkeys(mutation_types))
    if not mutation_types:
        raise ValueError("At least one mutation type is required.")

    known_mutation_types = set(generate_prompts.MUTATION_INSTRUCTIONS)
    unknown = [mutation_type for mutation_type in mutation_types if mutation_type not in known_mutation_types]
    if unknown:
        raise ValueError(
            "Unknown mutation type(s): "
            f"{', '.join(unknown)}. Known types: {', '.join(sorted(known_mutation_types))}."
        )
    return mutation_types


def apply_existing_matches(
    classification_review: Dict[str, Any],
    taxonomy: Dict[str, Any],
):
    """Apply each existing match to the in-memory taxonomy. Returns (applied, affected_paths);
    summary regeneration and saving are batched by the caller."""
    policy = classification_review["policy"]
    classification = classification_review.get("classification", {})
    applied = []
    affected_paths = []
    for match in classification.get("existing_matches", []):
        if not isinstance(match, dict):
            continue
        node_id = str(match.get("node_id", "")).strip()
        if not node_id:
            raise ValueError(f"Existing match is missing node_id: {match}")
        update_policy = policy_for_update(match, policy)
        segment = matched_segment(match, update_policy)
        path = update_tree.apply_leaf_policy_in_place(taxonomy, node_id, update_policy, segment)
        if path is not None:
            affected_paths.append(path)
        applied.append({"type": "existing", "node_id": node_id, "matched_segment": segment})
        print(f"[apply] Added policy evidence to existing leaf: {node_id}")
    return applied, affected_paths


def create_reviewed_novel_leaf(
    novel: Dict[str, Any],
    policy: Any,
    policy_index: int,
    novel_index: int,
    run_dir: Path,
    taxonomy: Dict[str, Any],
    taxonomy_path: Path,
    args: argparse.Namespace,
):
    """Generate/review prompts for a novel category and apply it to the in-memory taxonomy.
    Returns (applied_record, affected_path); summary regeneration and saving are batched by the
    caller. The new leaf's prompts/judge are generated as before."""
    parent_node_id = str(novel.get("parent_node_id", "")).strip()
    proposed_name = str(novel.get("proposed_name", "")).strip()
    if not parent_node_id or not proposed_name:
        raise ValueError(f"Novel category needs parent_node_id and proposed_name: {novel}")

    # Guard: if a leaf with this name already exists under the parent (e.g. another policy in a
    # --parallel-policies run already created it, since classification ran against a pre-run tree
    # snapshot, or an earlier novel category in this same policy), attach the policy to that leaf
    # instead of creating a duplicate (which would raise).
    duplicate_node_id = existing_leaf_node_id(taxonomy, parent_node_id, proposed_name)
    if duplicate_node_id:
        update_policy = policy_for_update(novel, policy)
        segment = matched_segment(novel, update_policy)
        path = update_tree.apply_leaf_policy_in_place(taxonomy, duplicate_node_id, update_policy, segment)
        print(f"[apply] Novel category {proposed_name!r} already exists as {duplicate_node_id}; attached policy evidence.")
        return {
            "type": "existing",
            "node_id": duplicate_node_id,
            "parent_node_id": parent_node_id,
            "name": proposed_name,
        }, path

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
                    "Review persona-styled base_prompt_candidates. Keep the ones you want to carry "
                    "forward (with any manual edits) in selected_base_prompts; defaults to the first "
                    f"{args.base_select}."
                ),
                "novel_category": novel,
                "base_prompt_candidates": base_candidates,
                "selected_base_prompts": base_candidates[: args.base_select],
            },
            f"Choose/edit base prompts for novel category {proposed_name!r}.",
            resume=args.resume,
            yes=args.yes,
        )
    base_prompts = selected_base_prompts(base_review)
    mutation_types = selected_mutation_types(args)

    attack_path = run_dir / f"policy-{policy_index:03d}-novel-{novel_index:03d}-attack-prompts.json"
    if args.resume and attack_path.exists():
        attack_review = read_json(attack_path)
        attack_prompts = selected_attack_prompts(
            attack_review.get("attack_prompts", []),
            str(attack_path),
        )
    else:
        mutated_prompts = generate_prompts.mutate_prompts(
            base_prompts,
            review_rounds=args.mutation_review_rounds,
            mutation_types=mutation_types,
        )
        attack_prompts = generate_prompts.prompts_with_mutations(
            base_prompts,
            mutated_prompts,
            mutation_types=mutation_types,
        )
        write_json(
            attack_path,
            {
                "novel_category": novel,
                "selected_base_prompts": base_prompts,
                "mutation_types": mutation_types,
                "attack_prompts": attack_prompts,
            },
        )
        print(f"[artifact] Wrote ordered attack prompts to {attack_path}")
    attack_prompts = selected_attack_prompts(attack_prompts, str(attack_path))

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
    final_prompts = selected_attack_prompts(final_prompts, str(judge_review_path))
    if not final_judge:
        raise ValueError("judge_prompt is required.")

    translation_languages = normalize_languages(getattr(args, "translation_language", None))
    if translation_languages:
        translated_path = run_dir / f"policy-{policy_index:03d}-novel-{novel_index:03d}-translated-prompts.json"
        if args.resume and translated_path.exists():
            translated_review = review_json(
                translated_path,
                {},
                f"Review/edit translated prompts for novel category {proposed_name!r}.",
                resume=True,
                yes=args.yes,
            )
        else:
            translated_prompts = generate_prompts.translate_prompts(
                final_prompts,
                translation_languages,
                review_rounds=args.translation_review_rounds,
            )
            translated_review = review_json(
                translated_path,
                {
                    "instructions": "Review/edit translated attack_prompts before the leaf is created.",
                    "novel_category": novel,
                    "translation_languages": translation_languages,
                    "judge_prompt": final_judge,
                    "attack_prompts": translated_prompts,
                },
                f"Review/edit translated prompts for novel category {proposed_name!r}.",
                resume=args.resume,
                yes=args.yes,
            )
        final_prompts = selected_attack_prompts(
            translated_review.get("attack_prompts", []),
            str(translated_path),
        )

    update_policy = policy_for_update(novel, policy)
    segment = matched_segment(novel, update_policy)
    created, path = update_tree.create_leaf_in_place(
        taxonomy,
        parent_node_id,
        proposed_name,
        update_policy,
        segment,
        final_prompts,
        final_judge,
    )
    print(f"[apply] Created novel leaf: {created.get('node_id')}")
    return {
        "type": "novel",
        "node_id": created.get("node_id", ""),
        "parent_node_id": parent_node_id,
        "name": proposed_name,
    }, path


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = args.run_dir or DEFAULT_RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_path = args.tree
    webscraping = None

    if args.scrape:
        webscraping = run_webscraper_stage(args, run_dir)
        policies_path = Path(webscraping["new_path"])
        policies = coerce_policy_list(read_json(policies_path))
    elif args.policies:
        policies_path = args.policies
        policies = coerce_policy_list(read_json(policies_path))
    elif args.export:
        policies_path = None
        policies = []
    else:
        raise ValueError("Provide a policy JSON path, pass --scrape, or pass --export for export-only.")

    result = {
        "policies_path": str(policies_path) if policies_path else None,
        "taxonomy_path": str(taxonomy_path),
        "run_dir": str(run_dir),
        "webscraping": webscraping,
        "export": None,
        "applied": [],
    }
    write_json(run_dir / "pipeline-result.json", result)

    def classify_policy(policy: Any) -> Dict[str, Any]:
        return update_tree.classify(
            policy,
            taxonomy_path=str(taxonomy_path),
            model=args.classification_model,
            max_children=args.max_children,
            max_leaf_matches=args.max_leaf_matches,
            max_fragments=args.max_fragments,
        )

    # Optional: classify every policy concurrently up front. classify() only reads the tree, so
    # this is parallel-safe; all tree mutations remain in the sequential loop below. Policies whose
    # classification artifact already exists (--resume) are skipped here and read in the loop.
    precomputed = [None] * len(policies)
    if args.parallel_policies and policies:
        pending = [
            (idx, policy)
            for idx, policy in enumerate(policies, start=1)
            if not (args.resume and (run_dir / f"policy-{idx:03d}-classification.json").exists())
        ]
        if pending:
            print(f"[classify] Pre-classifying {len(pending)} policies concurrently...")
            workers = max(1, min(update_tree.MAX_CONCURRENCY, len(pending)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for (idx, _), classification in zip(
                    pending, executor.map(lambda it: classify_policy(it[1]), pending)
                ):
                    precomputed[idx - 1] = classification

    policy_bar = tqdm(total=len(policies), desc="Policies", unit="policy")
    for policy_index, policy in enumerate(policies, start=1):
        classification_path = run_dir / f"policy-{policy_index:03d}-classification.json"
        if args.resume and classification_path.exists():
            classification_review = read_json(classification_path)
        else:
            classification = precomputed[policy_index - 1]
            if classification is None:
                tqdm.write(f"[classify] Policy {policy_index}/{len(policies)}")
                classification = classify_policy(policy)
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

        # Load the tree once, apply all of this policy's matches/novels in memory, regenerate the
        # affected branches' summaries a single time, and save once — instead of reloading,
        # regenerating, and rewriting the whole tree per individual attachment.
        taxonomy = update_tree.load_taxonomy(str(taxonomy_path))
        applied, affected_paths = apply_existing_matches(classification_review, taxonomy)
        classification = classification_review.get("classification", {})
        for novel_index, novel in enumerate(classification.get("novel_categories", []), start=1):
            if not isinstance(novel, dict):
                continue
            result_item, path = create_reviewed_novel_leaf(
                novel,
                classification_review["policy"],
                policy_index,
                novel_index,
                run_dir,
                taxonomy,
                taxonomy_path,
                args,
            )
            applied.append(result_item)
            if path is not None:
                affected_paths.append(path)
        if affected_paths:
            update_tree.load_summary_module().branch_update_many(taxonomy, affected_paths)
            update_tree.save_taxonomy(taxonomy, str(taxonomy_path))
        result["applied"].extend(applied)
        write_json(run_dir / "pipeline-result.json", result)
        tqdm.write(f"[policy {policy_index}/{len(policies)}] applied {len(applied)} update(s); {len(result['applied'])} total")
        policy_bar.update(1)
        policy_bar.set_postfix_str(f"{len(result['applied'])} updates")
    policy_bar.close()

    if args.export:
        result["export"] = run_export_stage(args, taxonomy_path)
        write_json(run_dir / "pipeline-result.json", result)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AIR-BENCH update pipeline with JSON review checkpoints."
    )
    parser.add_argument("policies", nargs="?", type=Path, help="JSON policy-clause list, such as scraper output.")
    parser.add_argument("--scrape", action="store_true", help="Run webscraping first and feed only new policies downstream.")
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE_PATH)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--scraper-model", default=scraper.DEFAULT_MODEL)
    parser.add_argument("--scrape-source", action="append", default=[], help="Limit scraping to one source name. Repeatable.")
    parser.add_argument("--pages-per-source", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-links-per-page", type=int, default=12)
    parser.add_argument("--max-page-chars", type=int, default=180000)
    parser.add_argument("--chunk-chars", type=int, default=60000)
    parser.add_argument("--max-chunks-per-page", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=0.35)
    parser.add_argument("--skip-final-gate", action="store_true")
    parser.add_argument("--previous-policies", action="append", type=Path, default=[])
    parser.add_argument("--previous-policies-dir", action="append", type=Path, default=[])
    parser.add_argument("--no-default-policy-history", action="store_true")
    parser.add_argument("--classification-model", default=update_tree.MODEL)
    parser.add_argument("--max-children", type=int, default=4)
    parser.add_argument("--max-leaf-matches", type=int, default=5)
    parser.add_argument("--max-fragments", type=int, default=12)
    parser.add_argument("--base-count", type=int, default=8, help="How many base prompt candidates to generate per category.")
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
            f"Defaults to: {', '.join(generate_prompts.DEFAULT_TRANSLATION_LANGUAGES)}."
        ),
    )
    parser.add_argument(
        "--mutation-type",
        action="append",
        default=list(generate_prompts.DEFAULT_MUTATION_TYPES),
        choices=sorted(generate_prompts.MUTATION_INSTRUCTIONS),
        help=(
            "Mutation type to generate after each base prompt. Repeatable. "
            f"Defaults to: {', '.join(generate_prompts.DEFAULT_MUTATION_TYPES)}."
        ),
    )
    parser.add_argument("--export", action="store_true", help="Export AIR-BENCH prompt and judge CSVs after updates.")
    parser.add_argument(
        "--export-prompts-out",
        type=Path,
        default=export_dataset.DEFAULT_PROMPTS_PATH,
        help="Default prompt CSV path. Jurisdiction and language prompt CSVs are written as sibling files.",
    )
    parser.add_argument("--export-judges-out", type=Path, default=export_dataset.DEFAULT_JUDGES_PATH)
    parser.add_argument("--resume", action="store_true", help="Reuse existing review artifacts in --run-dir.")
    parser.add_argument("--yes", action="store_true", help="Do not pause for edits after writing review files.")
    parser.add_argument(
        "--parallel-policies",
        action="store_true",
        help=(
            "Classify all policies concurrently up front (read-only against the current tree "
            "snapshot) before the sequential review/apply loop. Faster for large automated runs; "
            "tree mutations stay sequential. Note: cross-policy novel-category de-duplication then "
            "relies on the review step / the apply-time guard rather than on later policies seeing "
            "earlier policies' new leaves."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_pipeline(args)
    print(f"\n[done] Applied {len(result['applied'])} tree update(s).")
    if result.get("export"):
        print(f"[done] Exported CSVs to {result['export']['prompt_paths']['default']}")
    print(f"[done] Review artifacts: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

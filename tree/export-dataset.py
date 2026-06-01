import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TREE_PATH = Path(__file__).with_name("semantic-tree.json")
DEFAULT_PROMPTS_PATH = Path(__file__).with_name("air_bench_prompts_default.csv")
DEFAULT_JUDGES_PATH = Path(__file__).with_name("air_bench_judge_prompts.csv")
ENGLISH_LANGUAGE = "English"
LEGISLATURE_PROMPT_SUBSETS = ("default", "china", "eu", "us")
LANGUAGE_PROMPT_SUBSETS = (ENGLISH_LANGUAGE, "Spanish", "Japanese", "Portuguese")

PROMPT_COLUMNS = ["cate-idx", "l2-name", "l3-name", "l4-name", "prompt"]
JUDGE_COLUMNS = ["cate-idx", "l2-name", "l3-name", "l4-name", "judge_prompt"]


def node_to_cate_id(l2_index: int, l3_index: int, l4_index: int) -> str:
    return f"{l2_index}.{l3_index}.{l4_index}"


def load_tree(path: Path = DEFAULT_TREE_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_node(node: Dict[str, Any], expected_level: int) -> None:
    if node.get("level") != expected_level:
        raise ValueError(
            f"Expected level {expected_level} node, got {node.get('level')}: "
            f"{node.get('node_id') or node.get('name')}"
        )


def validate_leaf(leaf: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    validate_node(leaf, 4)
    prompts = normalize_prompt_records(leaf.get("prompts", []))
    judge = leaf.get("judge", "")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Leaf {leaf.get('node_id') or leaf.get('name')} has no prompts.")
    if not isinstance(judge, str) or not judge.strip():
        raise ValueError(f"Leaf {leaf.get('node_id') or leaf.get('name')} has no judge prompt.")
    return prompts, judge


def normalize_prompt_records(prompts: Any) -> List[Dict[str, str]]:
    if not isinstance(prompts, list):
        raise ValueError("prompts must be a list.")

    normalized = []
    for idx, item in enumerate(prompts):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt at index {idx} must be an object with variant and prompt.")
        variant = str(item.get("variant", "")).strip()
        language = str(item.get("language", ENGLISH_LANGUAGE)).strip()
        text = str(item.get("prompt", "")).strip()
        if not variant or not language or not text:
            raise ValueError(f"Invalid prompt record at index {idx}: {item!r}")
        normalized.append({"variant": variant, "language": language, "prompt": text})
    return normalized


def source_legislatures(leaf: Dict[str, Any]) -> set:
    legislatures = set()
    for policy in leaf.get("policies", []):
        if not isinstance(policy, dict):
            continue
        source = policy.get("source", {})
        legislature = ""
        if isinstance(source, dict):
            legislature = str(source.get("legislature") or "").strip().lower()
        legislature = legislature or str(policy.get("legislature") or "").strip().lower()
        if legislature:
            legislatures.add(legislature)
    return legislatures


def include_leaf(leaf: Dict[str, Any], legislature: Optional[str]) -> bool:
    if legislature is None:
        return True
    return legislature in source_legislatures(leaf)


def include_prompt(prompt: Dict[str, str], language: Optional[str]) -> bool:
    if language is None:
        return True
    return prompt["language"].strip().lower() == language.strip().lower()


def tree_to_data(
    root: Dict[str, Any],
    legislature: Optional[str] = None,
    language: Optional[str] = None,
    include_judges: bool = True,
) -> Tuple[List[List[str]], List[List[str]]]:
    prompt_rows = []
    judge_rows = []
    l2_index = 0
    l3_index = 0

    for l1_node in root.get("children", []):
        validate_node(l1_node, 1)
        for l2_node in l1_node.get("children", []):
            validate_node(l2_node, 2)
            l2_index += 1
            l4_index = 0
            l2_name = l2_node.get("name", "")

            for l3_node in l2_node.get("children", []):
                validate_node(l3_node, 3)
                l3_index += 1
                l3_name = l3_node.get("name", "")

                for l4_node in l3_node.get("children", []):
                    prompts, judge = validate_leaf(l4_node)
                    l4_index += 1
                    if not include_leaf(l4_node, legislature):
                        continue
                    l4_name = l4_node.get("name", "")
                    cate_id = node_to_cate_id(l2_index, l3_index, l4_index)

                    for prompt in prompts:
                        if not include_prompt(prompt, language):
                            continue
                        prompt_rows.append([cate_id, l2_name, l3_name, l4_name, prompt["prompt"]])
                    if include_judges:
                        judge_rows.append([cate_id, l2_name, l3_name, l4_name, judge])

    return prompt_rows, judge_rows


def write_csv(path: Path, columns: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def prompt_subset_paths(default_path: Path = DEFAULT_PROMPTS_PATH) -> Dict[str, Path]:
    paths = {"default": default_path}
    stem = default_path.stem
    for subset in LEGISLATURE_PROMPT_SUBSETS:
        if subset == "default":
            continue
        paths[subset] = prompt_subset_path(default_path, stem, subset)
    for language in LANGUAGE_PROMPT_SUBSETS:
        paths[f"language_{language.lower()}"] = prompt_subset_path(
            default_path,
            stem,
            language.lower(),
        )
    return paths


def prompt_subset_path(default_path: Path, stem: str, subset: str) -> Path:
    if stem.endswith("_default"):
        subset_stem = f"{stem[: -len('_default')]}_{subset}"
    else:
        subset_stem = f"{stem}_{subset}"
    return default_path.with_name(f"{subset_stem}{default_path.suffix}")


def data_to_csv(
    tree_path: Path = DEFAULT_TREE_PATH,
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
    judges_path: Path = DEFAULT_JUDGES_PATH,
) -> Tuple[Dict[str, List[List[str]]], List[List[str]]]:
    root = load_tree(tree_path)
    prompt_rows_by_subset: Dict[str, List[List[str]]] = {}
    for subset, path in prompt_subset_paths(prompts_path).items():
        legislature = None if subset == "default" else subset
        language = None
        if subset.startswith("language_"):
            legislature = None
            language = subset.removeprefix("language_")
        prompt_rows, _ = tree_to_data(
            root,
            legislature=legislature,
            language=language,
            include_judges=False,
        )
        prompt_rows_by_subset[subset] = prompt_rows
        write_csv(path, PROMPT_COLUMNS, prompt_rows)

    _, judge_rows = tree_to_data(root)
    write_csv(judges_path, JUDGE_COLUMNS, judge_rows)
    return prompt_rows_by_subset, judge_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the semantic tree as AIR-BENCH CSV files.")
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE_PATH)
    parser.add_argument(
        "--prompts-out",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help=(
            "Default prompt CSV path. Legislature and language-specific prompt CSVs are written "
            "as sibling files."
        ),
    )
    parser.add_argument("--judges-out", type=Path, default=DEFAULT_JUDGES_PATH)
    args = parser.parse_args()

    prompt_rows_by_subset, judge_rows = data_to_csv(args.tree, args.prompts_out, args.judges_out)
    for subset, path in prompt_subset_paths(args.prompts_out).items():
        print(f"Wrote {len(prompt_rows_by_subset[subset])} prompt rows to {path}")
    print(f"Wrote {len(judge_rows)} judge rows to {args.judges_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

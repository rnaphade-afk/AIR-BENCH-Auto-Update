import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_TREE_PATH = Path(__file__).with_name("semantic-tree.json")
DEFAULT_PROMPTS_PATH = Path(__file__).with_name("air_bench_prompts.csv")
DEFAULT_JUDGES_PATH = Path(__file__).with_name("air_bench_judge_prompts.csv")

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


def validate_leaf(leaf: Dict[str, Any]) -> Tuple[List[str], str]:
    validate_node(leaf, 4)
    prompts = leaf.get("prompts", [])
    judge = leaf.get("judge", "")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Leaf {leaf.get('node_id') or leaf.get('name')} has no prompts.")
    if not isinstance(judge, str) or not judge.strip():
        raise ValueError(f"Leaf {leaf.get('node_id') or leaf.get('name')} has no judge prompt.")
    return prompts, judge


def tree_to_data(root: Dict[str, Any]) -> Tuple[List[List[str]], List[List[str]]]:
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
                    l4_name = l4_node.get("name", "")
                    cate_id = node_to_cate_id(l2_index, l3_index, l4_index)

                    for prompt in prompts:
                        prompt_rows.append([cate_id, l2_name, l3_name, l4_name, prompt])
                    judge_rows.append([cate_id, l2_name, l3_name, l4_name, judge])

    return prompt_rows, judge_rows


def write_csv(path: Path, columns: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def data_to_csv(
    tree_path: Path = DEFAULT_TREE_PATH,
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
    judges_path: Path = DEFAULT_JUDGES_PATH,
) -> Tuple[List[List[str]], List[List[str]]]:
    prompt_rows, judge_rows = tree_to_data(load_tree(tree_path))
    write_csv(prompts_path, PROMPT_COLUMNS, prompt_rows)
    write_csv(judges_path, JUDGE_COLUMNS, judge_rows)
    return prompt_rows, judge_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the semantic tree as AIR-BENCH CSV files.")
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE_PATH)
    parser.add_argument("--prompts-out", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--judges-out", type=Path, default=DEFAULT_JUDGES_PATH)
    args = parser.parse_args()

    prompt_rows, judge_rows = data_to_csv(args.tree, args.prompts_out, args.judges_out)
    print(f"Wrote {len(prompt_rows)} prompt rows to {args.prompts_out}")
    print(f"Wrote {len(judge_rows)} judge rows to {args.judges_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import re
import json
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

L2_TO_L1_MAP = {
    "1": "System & Operational Risks",
    "2": "System & Operational Risks",
    "3": "Content Safety Risks",
    "4": "Content Safety Risks",
    "5": "Content Safety Risks",
    "6": "Content Safety Risks",
    "7": "Content Safety Risks",
    "8": "Societal Risks",
    "9": "Societal Risks",
    "10": "Societal Risks",
    "11": "Societal Risks",
    "12": "Societal Risks",
    "13": "Legal & Rights-Related Risks",
    "14": "Legal & Rights-Related Risks",
    "15": "Legal & Rights-Related Risks",
    "16": "Legal & Rights-Related Risks",
}

BASE_VARIANT = "base"
MUTATION_VARIANT = "mutation"

def slugify(value):
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "unknown"

def cate_idx_sort_key(value):
    return tuple(int(part) for part in str(value).split("."))

def subtree_sort_key(data):
    if isinstance(data, dict) and 'node_id' in data:
        return cate_idx_sort_key(data['node_id'])

    child_keys = [
        subtree_sort_key(child_data)
        for child_data in data.values()
        if isinstance(child_data, dict)
    ]
    return min(child_keys) if child_keys else (float("inf"),)

def prompt_record(prompt, idx):
    variant = BASE_VARIANT if idx % 3 == 0 else MUTATION_VARIANT
    return {
        "variant": variant,
        "prompt": str(prompt).strip(),
    }

def build_tree(path='tree/semantic-tree.json'):
    #load data
    ds_prompts = load_dataset("stanford-crfm/air-bench-2024", "default", split="test")
    ds_judges = load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")
    judge_lookup = {row['cate-idx']: row['judge_prompt'] for row in ds_judges}

    #initializes a recursive dictionary to represent a tree
    def tree(): return defaultdict(tree)
    root = tree()

    #iterate through prompts
    for row in ds_prompts:        
        #extract classification at each level
        idx = row['cate-idx']
        l1_cat = L2_TO_L1_MAP.get((idx.split('.'))[0], "Unknown Risk")
        l2_cat = row['l2-name']
        l3_cat = row['l3-name']
        l4_cat = row['l4-name']

        #create/access corresponding leaf
        node = root[l1_cat][l2_cat][l3_cat][l4_cat]

        #if unitialized leaf, initialize
        if 'prompts' not in node:
            node['node_id'] = f"{idx}"
            node['name'] = l4_cat
            node['level'] = 4
            node['prompts'] = []
            node['judge'] = judge_lookup.get(idx)
            node['policies'] = []

        #add prompt
        node['prompts'].append(prompt_record(row['prompt'], len(node['prompts'])))
    
   #convert to JSON structure
    def format(name, data, level, parent_id):
        # leaf logic
        if isinstance(data, dict) and 'prompts' in data:
            leaf_node_id = f"{parent_id}/{slugify(name)}"
            return {
                "name": name,
                "node_id": leaf_node_id,
                "level": 4,
                "summary": f"Detailed policy category: {name}",
                "prompts": data['prompts'],
                "judge": data['judge'],
                "policies": data['policies']
            }
        
        # inner-node logic
        node_id = f"{parent_id}/{slugify(name)}" if level > 0 else "root"
        child_items = sorted(data.items(), key=lambda item: subtree_sort_key(item[1]))
        return {
            "name": name,
            "node_id": node_id,
            "level": level,
            "summary": f"Placeholder summary for {name}",
            "children": [format(child_name, child_data, level + 1, node_id)
                         for child_name, child_data in child_items]
        }

    final_json = format("AIR-BENCH Root", root, 0, "")

    # 5. Save it
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    build_tree()

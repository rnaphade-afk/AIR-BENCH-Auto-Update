import json
import pandas as pd
from datasets import load_dataset
from collections import defaultdict

L1_MAP = {
    "1": "System & Operational Risks",
    "2": "Content Safety Risks",
    "3": "Societal Risks",
    "4": "Legal & Rights Risks"
}

def build_tree():
    #load data
    ds = load_dataset("stanford-crfm/air-bench-2024", split="test")

    #initializes a recursive dictionary to represent a tree
    def tree(): return defaultdict(tree)
    root = tree()

    #iterate through prompts
    for row in ds:        
        #extract classification at each level
        l1_cat = L1_MAP.get((row['cate-idx'].split('.'))[0], "Unknown Risk")
        l2_cat = row['l2-name']
        l3_cat = row['l3-name']
        l4_cat = row['l4-name']

        #create/access corresponding leaf
        node = root[l1_cat][l2_cat][l3_cat][l4_cat]

        #if unitialized leaf, initialize
        if 'prompts' not in node:
            node['node_id'] = f"L4-{row['cate-idx']}"
            node['name'] = l4_cat
            node['level'] = 4
            node['prompts'] = []
            node['representative_clause'] = None

        #add prompt
        node['prompts'].append(row['prompt'])
    
   #convert to JSON structure
    def format(name, data, level):
        # leaf logic
        if isinstance(data, dict) and 'prompts' in data:
            return {
                "name": name,
                "node_id": data['node_id'],
                "level": 4,
                "summary": f"Detailed policy category: {name}",
                "prompts": data['prompts'],
                "representative_clause": data['representative_clause']
            }
        
        # inner-node logic
        return {
            "name": name,
            "level": level,
            "summary": f"Placeholder summary for {name}",
            "children": [format(child_name, child_data, level + 1) 
                         for child_name, child_data in data.items()]
        }

    final_json = format("AIR-BENCH Root", root, 0)

    # 5. Save it
    with open('semantic-tree.json', 'w') as f:
        json.dump(final_json, f, indent=4)

if __name__ == "__main__":
    build_tree()

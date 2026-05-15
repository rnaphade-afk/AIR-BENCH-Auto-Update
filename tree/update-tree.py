import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-5.4-mini"

SUMMARY_SYSTEM_PROMPT = """You write compact taxonomy definitions for a hierarchical LLM-safety classifier.
Return only the definition text. Do not include labels, markdown, examples, citations, caveats, or policy advice.
Prefer concrete regulatory concepts, protected targets, actors, and prohibited actions over broad safety language."""

def clean_summary(text):
    summary = " ".join(text.strip().split())
    for prefix in ("Definition:", "Summary:", "Category:"):
        if summary.lower().startswith(prefix.lower()):
            summary = summary[len(prefix):].strip()
    return summary

def get_policy_clauses(node):
    policies = node.get('policies') or []
    clauses = [policy.get('clause', '').strip() for policy in policies]
    clauses = [clause for clause in clauses if clause]
    if clauses:
        return clauses
    return node.get('policy_clauses') or []

def select_evidence_prompts(prompts, limit=8):
    base_prompts = prompts[::3]
    selected = base_prompts[:limit]
    if len(selected) < limit:
        selected.extend(prompt for idx, prompt in enumerate(prompts) if idx % 3 != 0)
    return selected[:limit]

def generate_summary(prompt, temperature=0.1):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_completion_tokens=120,
    )
    return clean_summary(response.choices[0].message.content)

def generate_leaf_summary(node):
    #case 1: leaf with policy clauses
    policy_clauses = get_policy_clauses(node)
    if policy_clauses:
        clauses = "\n".join([f"- {p}" for p in policy_clauses[:8]])

        prompt = f"""Category name: {node['name']}

Policy clauses:
{clauses}

Write exactly one sentence, no more than 35 words, defining the classification boundary for this leaf category.
The sentence must identify the regulated action, the protected target or domain, and the distinguishing boundary from adjacent risk categories.
Do not restate the category name unless needed for clarity."""

        node['summary'] = generate_summary(prompt, temperature=0.1)
        return node['summary']
    #case 2: leaf without policy clauses
    else:
        #generate summary based on category name and attack prompts
        evidence_prompts = "\n".join(
            [f"- {p}" for p in select_evidence_prompts(node['prompts'])]
        )
        prompt = f"""Category name: {node['name']}

Attack-prompt evidence:
{evidence_prompts}

Infer the tested safety boundary from the evidence while ignoring prompt wrappers, dialect changes, authority endorsements, named entities, and jailbreak tactics.
Write exactly one sentence, no more than 35 words, defining what model assistance should be classified into this leaf.
The sentence must name the concrete unsafe capability or decision, target/domain, and distinguishing boundary from nearby categories."""
        
        summary = generate_summary(prompt, temperature=0.1)
        node['summary'] = summary
        node['representative_clause'] = None
        return summary

def generate_recursive_summary(node):
    #base case
    if 'children' not in node or not node['children']:
        return generate_leaf_summary(node)

    #recursive step
    child_context = []
    for child in node['children']:
        child_summary = generate_recursive_summary(child)
        child_context.append(f"- {child['name']}: {child_summary}")

    # 3. Summarization Logic: Use GPT to summarize the children
    context_str = "\n".join(child_context)
    prompt = f"""Parent category: {node['name']}

Child category definitions:
{context_str}

Write exactly one sentence, no more than 45 words, defining the shared classification boundary of this parent category.
Use the child definitions as coverage constraints, but do not list every child.
Include the main kinds of actions, targets, or rights implicated, and exclude unrelated safety domains."""
    
    summary = generate_summary(prompt, temperature=0.1)
    node['summary'] = summary
    return summary

def branch_update(node, path):
    #base case
    if not path:
        if 'children' not in node or not node['children']:
            return generate_leaf_summary(node)
        return generate_recursive_summary(node)
    
    cur = path[0]
    remaining_path = path[1:]

    children = node.get('children', [])
    target_child = next((child for child in children if child['name'] == cur), None)
    if not target_child:
        raise ValueError(f"Could not find child {cur!r} under {node.get('name')!r}")

    #recurse
    branch_update(target_child, remaining_path)

    child_summaries = [f"- {c['name']}: {c['summary']}" for c in children]
    context_str = "\n".join(child_summaries)
    old_summary = node.get('summary', "No existing summary.")
    
    prompt = f"""Parent category: {node['name']}

Previous parent definition:
{old_summary}

Current child category definitions:
{context_str}

Write exactly one sentence, no more than 45 words, updating the shared classification boundary of this parent category.
Use the child definitions as coverage constraints, but do not list every child.
Include the main kinds of actions, targets, or rights implicated, and exclude unrelated safety domains."""

    summary = generate_summary(prompt, temperature=0.1)
    node['summary'] = summary
    return summary

def rebuild_tree_summaries(path='tree/semantic-tree.json'):
    with open(path, 'r') as f:
        taxonomy = json.load(f)

    generate_recursive_summary(taxonomy)

    with open(path, 'w') as f:
        json.dump(taxonomy, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    rebuild_tree_summaries()

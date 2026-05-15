import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_medoid(policies):
    #edge cases that shouldn't be accessed
    if not policies:
        return ""
    if len(policies) == 1:
        return policies[0]

    embeddings = embedding_model.encode(policies, convert_to_tensor=True)

    cos_sim = util.cos_sim(embeddings, embeddings)
    avg_similarities = torch.mean(cos_sim, dim=1)
    medoid_idx = torch.argmax(avg_similarities).item()
    
    return policies[medoid_idx]

def generate_leaf_summary(node):
    #case 1: leaf with policy clauses
    if node.get('policy_clauses'):
        medoid = calculate_medoid(node['policy_clauses'])
        node['representative_clause'] = medoid

        prompt = f"""Category: {node['name']}
        Full Policy Clause: {medoid}
        
        Create a discriminative definition for classification. In 1 sentence, concisely define the specific regulatory boundary of this clause. 
        Use precise terminology that distinguishes it from related safety risks (e.g., distinguish 'Copyright Infringement' from 'Fair Use Bypassing')."""

        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1 # Low temperature for consistency
        )

        node['summary'] = response.choices[0].message.content.strip()
        return node['summary']
    #case 2: leaf without policy clauses
    else:
        #generate summary based on category name and attack prompts
        evidence_prompts = "\n".join([f"- {p}" for p in node['prompts'][:8]])
        prompt = f"""For LLM safety category: {node['name']}, here are some specific attack prompts used for safety testing:
        {evidence_prompts}
        
        Analyze these attack prompts. Identify, but do not output the core 'vulnerability' or 'harm' they are attempting to exploit. 
        Based on this, write a concise 1-sentence definition of the tested category's scope for an automated classifier."""

        response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
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
    prompt = f"""You are synthesizing a parent risk category from its sub-categories: '{node['name']}':
    {context_str}
    
    Review the sub-category definitions above and provide a concise, high-level, two-sentence 'umbrella' definition.
    It should encompasses all sub-categories them without losing the technical focus of the group, effective for an automated classifier"""

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    summary = response.choices[0].message.content.strip()
    node['summary'] = summary
    return summary

with open('tree/semantic-tree.json', 'r') as f:
    taxonomy = json.load(f)

generate_recursive_summary(taxonomy)

with open('tree/semantic-tree.json', 'w') as f:
    json.dump(taxonomy, f, indent=4, ensure_ascii=False)
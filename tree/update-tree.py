import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_medoid(prompts):
    #edge cases that shouldn't be accessed
    if not prompts:
        return ""
    if len(prompts) == 1:
        return prompts[0]

    embeddings = embedding_model.encode(prompts, convert_to_tensor=True)

    cos_sim = util.cos_sim(embeddings, embeddings)
    avg_similarities = torch.mean(cos_sim, dim=1)
    medoid_idx = torch.argmax(avg_similarities).item()
    
    return prompts[medoid_idx]

def generate_recursive_summary(node):
    #base case
    if 'children' not in node or not node['children']:
        # use medoid as summary
        if not node.get('representative_clause'):
            node['representative_clause'] = calculate_medoid(node['prompts'])
        return node['name']

    #recursive step
    child_context = []
    for child in node['children']:
        child_summary = generate_recursive_summary(child)
        child_context.append(f"- {child['name']}: {child_summary}")

    # 3. Summarization Logic: Use GPT-4o to summarize the children
    context_str = "\n".join(child_context)
    prompt = f"""You are an AI Safety Policy Expert. 
    The following are sub-categories of AI risks under the category '{node['name']}':
    {context_str}
    
    Write a concise, high-level summary (3 sentences max) explaining what these risks share in common."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    summary = response.choices[0].message.content.strip()
    node['summary'] = summary
    return summary

with open('semantic-tree.json', 'r') as f:
    taxonomy = json.load(f)

generate_recursive_summary(taxonomy)

with open('semantic-tree.json', 'w') as f:
    json.dump(taxonomy, f, indent=4)
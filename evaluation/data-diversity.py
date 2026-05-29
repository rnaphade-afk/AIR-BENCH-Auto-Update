import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

TREE_PATH = "tree/semantic-tree.json"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLOTS_DIR = ROOT / "evaluation" / "plots"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/bert-base-nli-mean-tokens"
DEFAULT_DEVICE = "cpu"


def slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "unknown"


def embedding_diversity_metrics(embeddings):
    import numpy as np

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    cosine_distance = 1 - normalized @ normalized.T
    pairwise = cosine_distance[np.triu_indices(len(embeddings), k=1)]

    nearest = cosine_distance.copy()
    np.fill_diagonal(nearest, np.inf)
    nearest = nearest.min(axis=1)

    return {
        "mean_pairwise": float(pairwise.mean()),
        "median_pairwise": float(np.median(pairwise)),
        "mean_nearest": float(nearest.mean()),
        "near_duplicate_pairs": int((pairwise < 0.05).sum()),
    }


def plot_prompt_diversity(
    prompts,
    category,
    output_path=None,
    show=False,
    model_name=DEFAULT_EMBEDDING_MODEL,
    device=DEFAULT_DEVICE,
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if len(prompts) < 2:
        raise ValueError("At least two prompts are required to plot diversity.")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(sentences=prompts, convert_to_numpy=True, show_progress_bar=False)

    pca_2d = PCA(n_components=2, random_state=42)
    embeddings_pca = pca_2d.fit_transform(embeddings)
    metrics = embedding_diversity_metrics(embeddings)

    perplexity = min(30, max(1, len(prompts) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.scatterplot(
        x=embeddings_pca[:, 0], y=embeddings_pca[:, 1],
        color="#2f6f9f", alpha=0.75, ax=ax1, edgecolor=None, s=36
    )
    ax1.set_title("PCA Map", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.set_axisbelow(True)
    ax1.grid(True, which="major", color="#d7dde5", linewidth=0.8, alpha=0.9)
    if ax1.get_legend():
        ax1.get_legend().remove()

    sns.scatterplot(
        x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1],
        color="#2f6f9f", alpha=0.75, ax=ax2, edgecolor=None, s=36
    )
    ax2.set_title("t-SNE Map", fontsize=14, fontweight='bold')
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")
    ax2.set_axisbelow(True)
    ax2.grid(True, which="major", color="#d7dde5", linewidth=0.8, alpha=0.9)

    plt.suptitle(f"Prompt Diversity: {category}", fontsize=18, fontweight='bold', y=1.02)
    fig.text(
        0.5,
        0.01,
        (
            f"model={model_name} | n={len(prompts)} | mean pairwise cosine distance={metrics['mean_pairwise']:.3f} | "
            f"median={metrics['median_pairwise']:.3f} | mean nearest-neighbor distance={metrics['mean_nearest']:.3f} | "
            f"near-duplicate pairs (<0.05)={metrics['near_duplicate_pairs']} | "
            f"PCA variance shown={pca_2d.explained_variance_ratio_.sum():.1%}"
        ),
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def prompt_text(prompt):
    if isinstance(prompt, dict):
        return str(prompt.get("prompt", "")).strip()
    return str(prompt).strip()


def plot_leaf_diversity(
    node,
    output_dir=DEFAULT_PLOTS_DIR,
    show=False,
    model_name=DEFAULT_EMBEDDING_MODEL,
    device=DEFAULT_DEVICE,
):
    prompt_records = node.get("prompts", [])
    prompts = [prompt_text(prompt) for prompt in prompt_records]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise ValueError(f"Node {node.get('node_id') or node.get('name')} has no usable prompts.")

    leaf_name = node.get("name", "Unknown leaf")
    output_path = Path(output_dir) / f"{slugify(leaf_name)}.png"
    return plot_prompt_diversity(
        prompts,
        leaf_name,
        output_path=output_path,
        show=show,
        model_name=model_name,
        device=device,
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path, base_dir):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    repo_path = ROOT / path
    if repo_path.exists():
        return repo_path
    return base_dir / path


def novel_node_ids(pipeline_result):
    ids = []
    seen = set()
    for item in pipeline_result.get("applied", []):
        if not isinstance(item, dict) or item.get("type") != "novel":
            continue
        node_id = str(item.get("node_id", "")).strip()
        if node_id and node_id not in seen:
            ids.append(node_id)
            seen.add(node_id)
    return ids


def find_node_by_id(node, node_id):
    if isinstance(node, dict) and node.get("node_id") == node_id:
        return node
    for child in node.get("children", []):
        found = find_node_by_id(child, node_id)
        if found:
            return found
    return None


def novel_nodes_from_pipeline_result(pipeline_result_path, tree_path=None):
    pipeline_result_path = Path(pipeline_result_path)
    pipeline_result = load_json(pipeline_result_path)
    tree_path = tree_path or pipeline_result.get("taxonomy_path") or TREE_PATH
    tree_path = resolve_path(tree_path, pipeline_result_path.parent)
    taxonomy = load_json(tree_path)

    node_ids = novel_node_ids(pipeline_result)
    if not node_ids:
        raise ValueError(f"No novel node ids found in {pipeline_result_path}.")

    nodes = []
    missing = []
    for node_id in node_ids:
        node = find_node_by_id(taxonomy, node_id)
        if node is None:
            missing.append(node_id)
        else:
            nodes.append(node)
    if missing:
        raise ValueError(
            f"Could not find {len(missing)} novel node(s) in {tree_path}: "
            f"{', '.join(missing)}"
        )
    return nodes


def main():
    parser = argparse.ArgumentParser(
        description="Plot prompt diversity for novel leaves created by a pipeline run."
    )
    parser.add_argument("pipeline_result", type=Path, help="Path to a pipeline-result.json file.")
    parser.add_argument(
        "--tree",
        type=Path,
        default=None,
        help="Optional taxonomy JSON path. Defaults to taxonomy_path from pipeline-result.json.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory where plot PNGs are written. Defaults to evaluation/plots.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model name. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Torch device for SentenceTransformer. Defaults to cpu.",
    )
    parser.add_argument("--show", action="store_true", help="Also display plots interactively.")
    args = parser.parse_args()

    nodes = novel_nodes_from_pipeline_result(args.pipeline_result, args.tree)
    for node in nodes:
        output_path = plot_leaf_diversity(
            node,
            output_dir=args.plots_dir,
            show=args.show,
            model_name=args.model,
            device=args.device,
        )
        print(f"Wrote {output_path} for {node['node_id']} ({node['name']})")
    return


if __name__ == "__main__":
    main()

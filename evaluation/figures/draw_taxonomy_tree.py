#!/usr/bin/env python3
"""Draw the AIR-BENCH semantic-tree taxonomy as a node-link tree for a blog post.

Root -> 4 top-level risk categories -> 21 subcategories, with each subcategory's level-4 leaf
count shown as a bar. Reads the real structure/counts from tree/semantic-tree.json so the figure
stays accurate. Outputs PNG (raster) + PDF/SVG (vector) into this directory.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
TREE_PATH = ROOT / "tree" / "semantic-tree.json"
OUT_STEM = Path(__file__).with_name("air_bench_taxonomy_tree")

# Muted, print-friendly palette (one hue per top-level category) — deliberately restrained.
L1_COLORS = ["#3E6D9C", "#4F8A6B", "#C08A3E", "#8B5E83"]
INK = "#2b2b2b"        # near-black text
LINE = "#9aa0a6"       # connector grey
SUBTLE = "#6b7177"     # secondary text


def children(node):
    return node.get("children") or []


def count_leaves(node):
    kids = children(node)
    return 1 if not kids else sum(count_leaves(k) for k in kids)


def main():
    tree = json.loads(TREE_PATH.read_text())
    l1_nodes = children(tree)

    # Flatten to (l1_index, l1_name, l2_name, leaf_count) rows, preserving tree order.
    rows = []
    for i, l1 in enumerate(l1_nodes):
        for l2 in children(l1):
            rows.append((i, l1["name"], l2["name"], count_leaves(l2)))

    # Vertical layout: one row per subcategory, top-to-bottom, with a gap between categories.
    row_h, group_gap = 1.0, 0.7
    ys, y, prev = [], 0.0, None
    for (i, *_rest) in rows:
        if prev is not None and i != prev:
            y -= group_gap
        ys.append(y)
        y -= row_h
        prev = i

    # Parent y = midpoint of its children's rows.
    l1_y = {i: sum(ys[k] for k, r in enumerate(rows) if r[0] == i)
               / sum(1 for r in rows if r[0] == i) for i in range(len(l1_nodes))}
    root_y = sum(l1_y.values()) / len(l1_y)

    # x coordinates (data units). The L1->L2 gap is wide enough that even the longest category
    # label clears the subcategory column.
    x_root, x_l1, x_l2 = 0.0, 1.9, 5.4
    x_bar0, bar_max = 9.7, 4.4
    max_leaves = max(r[3] for r in rows)
    label_bbox = dict(boxstyle="round,pad=0.12", fc="white", ec="none")

    fig_h = max(6.5, (max(ys) - min(ys)) * 0.46 + 1.6)
    fig, ax = plt.subplots(figsize=(13.0, fig_h))

    def elbow(x0, y0, x1, y1, color=LINE, lw=0.9):
        """Orthogonal parent->child connector (horizontal, vertical, horizontal)."""
        xm = (x0 + x1) / 2
        ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color=color, lw=lw,
                solid_capstyle="round", zorder=1)

    # root -> L1
    for i in range(len(l1_nodes)):
        elbow(x_root, root_y, x_l1, l1_y[i], lw=1.1)
    # L1 -> L2
    for k, (i, _l1name, _l2name, _n) in enumerate(rows):
        elbow(x_l1, l1_y[i], x_l2, ys[k], color=L1_COLORS[i], lw=0.9)

    # Root node
    ax.plot([x_root], [root_y], "o", ms=9, color=INK, zorder=3)
    ax.text(x_root - 0.12, root_y, "AIR-BENCH\ntaxonomy", ha="right", va="center",
            fontsize=12.5, color=INK, weight="bold", linespacing=1.05, zorder=4)

    total = count_leaves(tree)
    ax.text(x_root - 0.12, root_y - 1.15, f"{len(l1_nodes)} categories\n{total} leaf risks",
            ha="right", va="center", fontsize=8.5, color=SUBTLE, linespacing=1.2, zorder=4)

    # L1 nodes (white-backed labels so connector lines don't strike through the text)
    for i, l1 in enumerate(l1_nodes):
        c = L1_COLORS[i]
        ax.plot([x_l1], [l1_y[i]], "o", ms=8, color=c, zorder=3)
        ax.text(x_l1 + 0.15, l1_y[i] + 0.17, l1["name"], ha="left", va="bottom",
                fontsize=10.0, color=INK, weight="bold", zorder=4, bbox=label_bbox)
        ax.text(x_l1 + 0.15, l1_y[i] - 0.19, f"{count_leaves(l1)} leaves", ha="left", va="top",
                fontsize=7.8, color=SUBTLE, zorder=4, bbox=label_bbox)

    # L2 rows: marker, label, leaf-count bar
    for k, (i, _l1name, l2name, n) in enumerate(rows):
        c = L1_COLORS[i]
        y0 = ys[k]
        ax.plot([x_l2], [y0], "o", ms=5, color=c, zorder=3)
        ax.text(x_l2 + 0.12, y0, l2name, ha="left", va="center", fontsize=9.6, color=INK)
        w = n / max_leaves * bar_max
        ax.add_patch(plt.Rectangle((x_bar0, y0 - 0.26), w, 0.52, color=c, alpha=0.85,
                                   lw=0, zorder=2))
        ax.text(x_bar0 + w + 0.12, y0, str(n), ha="left", va="center", fontsize=8.6,
                color=SUBTLE)

    # Header
    ax.text(x_root - 0.12, max(ys) + 1.7,
            "The AIR-BENCH risk taxonomy",
            ha="left", va="bottom", fontsize=15, color=INK, weight="bold")
    ax.text(x_root - 0.12, max(ys) + 1.05,
            f"Four policy-derived risk categories, {len(rows)} subcategories, "
            f"{total} leaf-level risks.",
            ha="left", va="bottom", fontsize=9.5, color=SUBTLE)
    ax.text(x_bar0, max(ys) + 0.55, "level-4 leaves per subcategory", ha="left", va="bottom",
            fontsize=8.2, color=SUBTLE, style="italic")

    ax.set_xlim(-2.0, x_bar0 + bar_max + 1.2)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 2.4)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.01)

    for ext, dpi in (("png", 220), ("pdf", None), ("svg", None)):
        out = OUT_STEM.with_suffix("." + ext)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()

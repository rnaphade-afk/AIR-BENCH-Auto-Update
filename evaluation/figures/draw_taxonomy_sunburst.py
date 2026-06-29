#!/usr/bin/env python3
"""Sunburst of the AIR-BENCH Live taxonomy (nested rings, AIR-Bench-2024 Figure-1 style).

Three rings of filled wedges sized by leaf count: top-level categories (inner), subcategories
(middle), level-3 groups (outer); colored by top category and lightened outward. Level-1 labels
sit in the inner ring; the 16 subcategories are listed in a grouped legend with leaf counts.
Reads the live tree so it stays accurate.
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge

ROOT = Path(__file__).resolve().parents[2]
TREE_PATH = ROOT / "tree" / "semantic-tree.json"
OUT_STEM = Path(__file__).with_name("air_bench_taxonomy_sunburst")
L1_COLORS = ["#3E6D9C", "#4F8A6B", "#C08A3E", "#8B5E83"]
INK, SUBTLE = "#2b2b2b", "#5b6066"
RINGS = {1: (0.30, 0.55), 2: (0.55, 0.78), 3: (0.78, 1.0)}
LIGHTEN = {1: 0.0, 2: 0.34, 3: 0.58}


def children(n):
    return n.get("children") or []


def count_leaves(n):
    k = children(n)
    return 1 if not k else sum(count_leaves(c) for c in k)


def lighten(hexc, frac):
    r, g, b = mcolors.to_rgb(hexc)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def main():
    tree = json.loads(TREE_PATH.read_text())
    total = count_leaves(tree)
    nodes = []  # (depth, l1, idx, theta1, theta2, name)

    def assign(node, start, depth, l1, idx):
        span = count_leaves(node) / total * 360.0
        if depth >= 1:
            nodes.append((depth, l1, idx, start, start + span, node.get("name", "")))
        s = start
        for j, c in enumerate(children(node)):
            assign(c, s, depth + 1, (j if depth == 0 else l1), j)
            s += count_leaves(c) / total * 360.0
        return span

    assign(tree, 90.0, 0, -1, 0)

    fig, ax = plt.subplots(figsize=(11, 11))

    for depth, l1, idx, t1, t2, name in nodes:
        if depth > 3:
            continue
        r0, r1 = RINGS[depth]
        base = lighten(L1_COLORS[l1], LIGHTEN[depth])
        # subtle alternating shade to separate same-parent neighbours
        shade = lighten(base, 0.06) if idx % 2 else base
        ax.add_patch(Wedge((0, 0), r1, t1, t2, width=r1 - r0, facecolor=shade,
                           edgecolor="white", linewidth=0.7, zorder=2))

    # Level-1 labels inside the inner ring (radial, white). Short forms; full names in the legend.
    short = {
        "System & Operational Risks": "System &\nOperational",
        "Content Safety Risks": "Content\nSafety",
        "Societal Risks": "Societal",
        "Legal & Rights-Related Risks": "Legal &\nRights",
    }
    for depth, l1, idx, t1, t2, name in nodes:
        if depth != 1:
            continue
        a = math.radians((t1 + t2) / 2)
        r = sum(RINGS[1]) / 2
        deg = (math.degrees(a)) % 360
        rot = deg + 180 if 90 < deg < 270 else deg
        ax.text(r * math.cos(a), r * math.sin(a), short.get(name, name),
                rotation=rot, rotation_mode="anchor", ha="center", va="center",
                fontsize=9.5, color="white", weight="bold", zorder=4, linespacing=0.95)

    # Center hub.
    ax.add_patch(Circle((0, 0), RINGS[1][0], facecolor="white", edgecolor="#d0d0d0",
                        linewidth=1.0, zorder=3))
    ax.text(0, 0.03, "AIR-BENCH", ha="center", va="center", fontsize=12, color=INK, weight="bold", zorder=4)
    ax.text(0, -0.045, "Live", ha="center", va="center", fontsize=12, color=INK, weight="bold", zorder=4)
    ax.text(0, -0.115, "taxonomy", ha="center", va="center", fontsize=8, color=SUBTLE, zorder=4)

    # Legend: the 16 subcategories grouped by category, with leaf counts.
    l1_nodes = children(tree)
    handles, labels = [], []
    for i, l1 in enumerate(l1_nodes):
        handles.append(Line2D([0], [0], marker="s", linestyle="none", markersize=9,
                              markerfacecolor=L1_COLORS[i], markeredgecolor="none"))
        labels.append(l1["name"].upper())
        for l2 in children(l1):
            handles.append(Line2D([0], [0], marker="s", linestyle="none", markersize=7,
                                  markerfacecolor=lighten(L1_COLORS[i], LIGHTEN[2]),
                                  markeredgecolor="none"))
            labels.append(f"   {l2['name']}  ({count_leaves(l2)})")
    leg = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                    frameon=False, fontsize=8.2, handletextpad=0.5, labelspacing=0.28,
                    borderaxespad=0.0)

    ax.text(0, -1.18,
            f"{len(l1_nodes)} categories  ·  {sum(len(children(l1)) for l1 in l1_nodes)} "
            f"subcategories  ·  {total} leaf-level risks",
            ha="center", va="top", fontsize=9.5, color=SUBTLE)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.3, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=0.72, top=1.0, bottom=0.0)
    for ext, dpi in (("png", 230), ("pdf", None), ("svg", None)):
        out = OUT_STEM.with_suffix("." + ext)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()

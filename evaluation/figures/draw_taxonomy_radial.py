#!/usr/bin/env python3
"""Draw the FULL-DEPTH AIR-BENCH taxonomy as a radial dendrogram (phylogenetic style).

Root at the centre; one ring per level (4 top categories -> 16 subcategories -> 43 groups ->
335 leaf risks). Every leaf is a tip on the rim, coloured by its top-level category; the 16
subcategories are labelled as spokes. Reads the live tree so counts stay accurate. Outputs PNG +
PDF/SVG into this directory.
"""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge

ROOT = Path(__file__).resolve().parents[2]
TREE_PATH = ROOT / "tree" / "semantic-tree.json"
OUT_STEM = Path(__file__).with_name("air_bench_taxonomy_radial")

L1_COLORS = ["#3E6D9C", "#4F8A6B", "#C08A3E", "#8B5E83"]
INK = "#2b2b2b"
SUBTLE = "#6b7177"
MAXDEPTH = 4  # root=0, L1=1, L2=2, L3=3, leaf=4


def children(node):
    return node.get("children") or []


def count_leaves(node):
    kids = children(node)
    return 1 if not kids else sum(count_leaves(k) for k in kids)


def main():
    tree = json.loads(TREE_PATH.read_text())
    n_leaves = count_leaves(tree)

    # Assign each leaf an angle (evenly spaced, in DFS order so siblings stay contiguous);
    # each internal node sits at the mean angle of its descendants, at radius = its depth.
    state = {"i": 0}

    def layout(node, depth, l1):
        kids = children(node)
        if not kids:
            ang = (state["i"] + 0.5) / n_leaves * 2 * math.pi
            state["i"] += 1
        else:
            angs = [layout(c, depth + 1, (j if depth == 0 else l1))
                    for j, c in enumerate(kids)]
            ang = sum(angs) / len(angs)
        node["_a"], node["_d"], node["_l1"] = ang, depth, l1
        return ang

    layout(tree, 0, -1)

    def xy(node):
        r = node["_d"] / MAXDEPTH
        return r * math.cos(node["_a"]), r * math.sin(node["_a"])

    fig, ax = plt.subplots(figsize=(13, 13))

    # Edges, coloured by the child's top-level category.
    def draw_edges(node):
        x0, y0 = xy(node)
        for c in children(node):
            x1, y1 = xy(c)
            col = L1_COLORS[c["_l1"]] if c["_l1"] >= 0 else SUBTLE
            lw = 0.5 if c["_d"] >= 3 else (0.9 if c["_d"] == 2 else 1.4)
            ax.plot([x0, x1], [y0, y1], color=col, lw=lw, alpha=0.75,
                    solid_capstyle="round", zorder=1)
            draw_edges(c)

    draw_edges(tree)

    # Collect nodes by depth.
    by_depth = {d: [] for d in range(MAXDEPTH + 1)}

    def collect(node):
        by_depth[node["_d"]].append(node)
        for c in children(node):
            collect(c)

    collect(tree)

    # Leaf tips.
    for leaf in by_depth[MAXDEPTH]:
        x, y = xy(leaf)
        ax.plot([x], [y], "o", ms=2.6, color=L1_COLORS[leaf["_l1"]], zorder=3)
    # Faint inner-node dots (structure cue).
    for d in (1, 2, 3):
        for nd in by_depth[d]:
            x, y = xy(nd)
            ax.plot([x], [y], "o", ms=(4.5 if d == 1 else 2.4), color=L1_COLORS[nd["_l1"]],
                    zorder=3, alpha=0.9 if d == 1 else 0.6)
    ax.plot([0], [0], "o", ms=6, color=INK, zorder=4)

    # L1 grouping arcs just outside the leaf rim (leaves of one category are contiguous).
    r_leaf = 1.0
    for i, l1 in enumerate(by_depth[1]):
        leaf_angs = [lf["_a"] for lf in by_depth[MAXDEPTH] if lf["_l1"] == i]
        pad = (math.pi / n_leaves)  # half a leaf step
        t1 = math.degrees(min(leaf_angs) - pad)
        t2 = math.degrees(max(leaf_angs) + pad)
        ax.add_patch(Wedge((0, 0), r_leaf + 0.055, t1, t2, width=0.03,
                           facecolor=L1_COLORS[i], edgecolor="none", alpha=0.9, zorder=2))

    # Subcategory (L2) spoke labels around the rim.
    r_lbl = r_leaf + 0.085
    for l2 in by_depth[2]:
        a = l2["_a"]
        deg = math.degrees(a) % 360
        flip = 90 < deg < 270
        rot = deg + 180 if flip else deg
        ha = "right" if flip else "left"
        ax.text(r_lbl * math.cos(a), r_lbl * math.sin(a), l2["name"],
                rotation=rot, rotation_mode="anchor", ha=ha, va="center",
                fontsize=7.4, color=INK, zorder=4)

    # Legend: top categories with leaf counts.
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                      markerfacecolor=L1_COLORS[i], markeredgecolor="none",
                      label=f"{l1['name']}  ({count_leaves(l1)})")
               for i, l1 in enumerate(by_depth[1])]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0),
                    frameon=False, fontsize=9.5, handletextpad=0.5, labelspacing=0.6,
                    title="Top-level risk category  (leaves)")
    leg.get_title().set_fontsize(9.5)
    leg.get_title().set_color(INK)

    # Title / caption with live counts.
    depth_n = Counter()

    def cnt(n, d):
        depth_n[d] += 1
        for c in children(n):
            cnt(c, d + 1)

    cnt(tree, 0)
    ax.text(0, -1.47, "The AIR-BENCH risk taxonomy, full depth",
            transform=ax.transData, ha="center", va="top", fontsize=14, color=INK,
            weight="bold")
    ax.text(0, -1.55,
            f"{depth_n[1]} categories  ·  {depth_n[2]} subcategories  ·  "
            f"{depth_n[3]} groups  ·  {depth_n[4]} leaf-level risks",
            transform=ax.transData, ha="center", va="top", fontsize=9.5, color=SUBTLE)

    lim = 1.42
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.62, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    for ext, dpi in (("png", 240), ("pdf", None), ("svg", None)):
        out = OUT_STEM.with_suffix("." + ext)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()

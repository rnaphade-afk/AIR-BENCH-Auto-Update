#!/usr/bin/env python3
"""Methodology figure: the pipeline's data representation --- a depth-4 JSON tree in which every
node carries metadata. Shows one representative root->leaf path plus the JSON schema stored at an
inner node (a recursive summary) and at a leaf (summary + attack prompts + judge prompt + the
scraped policies that justify the category). Sibling counts are read live from the tree.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
TREE = json.loads((ROOT / "tree" / "semantic-tree.json").read_text())
OUT = Path(__file__).with_name("air_bench_taxonomy_representation")
ACCENT, INK, SUBTLE, CARDBG = "#3E6D9C", "#2b2b2b", "#6b7177", "#f6f7f9"
CODE, KEY = "#565c62", "#3E6D9C"          # value/punctuation grey, key accent
_KEY_RE = re.compile(r'"[^"]+"(?=\s*:)')   # a quoted string immediately followed by a colon

PATH = ["System & Operational Risks", "Security Risks", "Confidentiality", "Network intrusion"]


def kids(n):
    return n.get("children") or []


def walk_path(tree, names):
    """Return [(display, level_label, 'k of N')] for root + each named node along the path."""
    out = [("AIR-BENCH root", "root", None)]
    node, level = tree, 0
    labels = ["level 1 (category)", "level 2 (subcategory)", "level 3 (group)", "level 4 (leaf)"]
    for i, name in enumerate(names):
        sibs = kids(node)
        idx = next((j for j, c in enumerate(sibs) if c.get("name") == name), 0)
        out.append((name, labels[i], f"{idx + 1} of {len(sibs)}"))
        node = sibs[idx]
    return out


INNER_JSON = """{
  "name": "Confidentiality",
  "level": 3,
  "summary": "<recursive synthesis of
              child summaries (GPT 5.4-mini)>",
  "children": [ ... ]
}"""

LEAF_JSON = """{
  "name": "Network intrusion",
  "level": 4,
  "summary": "<GPT 5.4-mini category summary>",
  "prompts": [
    { "variant": "base | authority_endorsement",
      "language": "English | ES | JA | PT",
      "prompt": "<attack prompt>" }, ...
  ],
  "judge": "<judge-prompt template>",
  "policies": [
    { "clause": "<scraped policy text>",
      "matched_segment": "<...>",
      "source": { "legislature": "us",
                  "url": "<...>" } }, ...
  ]
}"""


def _keys_only(body):
    """Same text as body but every character blanked except keys (quoted strings before a colon),
    so overlaying it in monospace recolors just the keys."""
    lines = []
    for line in body.split("\n"):
        chars = [" "] * len(line)
        for m in _KEY_RE.finditer(line):
            chars[m.start():m.end()] = line[m.start():m.end()]
        lines.append("".join(chars))
    return "\n".join(lines)


def card(ax, x, y, w, h, title, body, border):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                fc=CARDBG, ec=border, lw=1.4, zorder=2))
    ax.add_patch(Rectangle((x, y), 0.7, h, fc=border, ec="none", zorder=3))  # left stripe
    ax.text(x + 1.4, y + h - 1.6, title, fontsize=9.5, color=border, weight="bold",
            va="top", ha="left", zorder=4)
    tx, ty = x + 1.4, y + h - 4.6
    kw = dict(fontsize=7.4, family="monospace", va="top", ha="left", linespacing=1.32)
    ax.text(tx, ty, body, color=CODE, zorder=4, **kw)                 # values + punctuation (grey)
    ax.text(tx, ty, _keys_only(body), color=KEY, zorder=5, **kw)      # keys (accent), overlaid


def main():
    steps = walk_path(TREE, PATH)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(2, 96, "Data representation: a JSON tree with per-node metadata", fontsize=14,
            color=INK, weight="bold", va="top")
    ax.text(2, 91.5, "A depth-4 tree; inner nodes store a recursive summary, leaves store the "
            "benchmark payload the pipeline reads and writes.", fontsize=9.2, color=SUBTLE, va="top")

    # ---- left: representative root -> leaf path ----
    bx, bw, bh = 3.0, 27.0, 8.5
    ys = [78, 63, 48, 33, 16]
    boxes = {}
    for (name, lvl, kn), y in zip(steps, ys):
        is_leaf = lvl.endswith("(leaf)")
        col = ACCENT if (lvl == "root" or is_leaf) else "#9aa0a6"
        ax.add_patch(FancyBboxPatch((bx, y), bw, bh, boxstyle="round,pad=0.3",
                                    fc="#eef2f7" if (lvl == "root" or is_leaf) else "white",
                                    ec=col, lw=1.6 if is_leaf else 1.2, zorder=3))
        ax.text(bx + bw / 2, y + bh - 2.6, name, fontsize=9.2, color=INK, weight="bold",
                ha="center", va="top", zorder=4)
        tag = lvl if kn is None else f"{lvl}   ·   {kn}"
        ax.text(bx + bw / 2, y + 1.8, tag, fontsize=7.0, color=SUBTLE, ha="center", va="bottom",
                zorder=4)
        boxes[lvl] = (bx, y, bw, bh)
    # downward connectors
    for (a_y), (b_y) in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((bx + bw / 2, a_y), (bx + bw / 2, b_y + bh),
                                     arrowstyle="-|>", mutation_scale=11, color="#9aa0a6",
                                     lw=1.1, shrinkA=0, shrinkB=0, zorder=1))

    # ---- right: metadata cards ----
    card(ax, 45, 55, 53, 33, "inner node  —  metadata", INNER_JSON, "#8a8f94")
    card(ax, 45, 4, 53, 46, "leaf node  —  metadata", LEAF_JSON, ACCENT)

    # dashed leaders from the path to the cards
    lx, ly, lw_, lh_ = boxes["level 3 (group)"]
    ax.add_patch(FancyArrowPatch((lx + lw_, ly + lh_ / 2), (45, 62),
                                 arrowstyle="-|>", mutation_scale=10, color="#8a8f94",
                                 lw=1.0, linestyle=(0, (4, 3)), shrinkA=2, shrinkB=2, zorder=1))
    lx, ly, lw_, lh_ = boxes["level 4 (leaf)"]
    ax.add_patch(FancyArrowPatch((lx + lw_, ly + lh_ / 2), (45, 26),
                                 arrowstyle="-|>", mutation_scale=10, color=ACCENT,
                                 lw=1.0, linestyle=(0, (4, 3)), shrinkA=2, shrinkB=2, zorder=1))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext, dpi in (("png", 220), ("pdf", None), ("svg", None)):
        out = OUT.with_suffix("." + ext)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()

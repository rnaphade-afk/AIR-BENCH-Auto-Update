#!/usr/bin/env python3
"""Methodology figure: the AIR-BENCH taxonomy as a depth-4 tree (the representation the pipeline
starts from). Compact, horizontal root -> category -> subcategory layout, with per-subcategory
level-3 / level-4 counts and a banner stating the four-tier cardinalities.

Based on the ORIGINAL AIR-BENCH 2024 taxonomy (4 / 16 / 43 / 314), derived once from
stanford-crfm/air-bench-2024 and hard-coded here so the figure renders offline.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_STEM = Path(__file__).with_name("air_bench_taxonomy_representation")
L1_COLORS = {
    "System & Operational Risks": "#3E6D9C",
    "Content Safety Risks": "#4F8A6B",
    "Societal Risks": "#C08A3E",
    "Legal & Rights-Related Risks": "#8B5E83",
}
INK, SUBTLE, LINE = "#2b2b2b", "#6b7177", "#9aa0a6"

# (top category, subcategory, #level-3 groups, #level-4 leaves) -- original AIR-BENCH 2024.
ORIG = [
    ("System & Operational Risks", "Security Risks", 3, 12),
    ("System & Operational Risks", "Operational Misuses", 3, 26),
    ("Content Safety Risks", "Violence & Extremism", 6, 24),
    ("Content Safety Risks", "Hate/Toxicity", 4, 36),
    ("Content Safety Risks", "Sexual Content", 4, 9),
    ("Content Safety Risks", "Child Harm", 2, 7),
    ("Content Safety Risks", "Self-harm", 1, 3),
    ("Societal Risks", "Political Usage", 4, 25),
    ("Societal Risks", "Economic Harm", 4, 10),
    ("Societal Risks", "Deception", 3, 9),
    ("Societal Risks", "Manipulation", 2, 5),
    ("Societal Risks", "Defamation", 1, 3),
    ("Legal & Rights-Related Risks", "Fundamental Rights", 1, 5),
    ("Legal & Rights-Related Risks", "Discrimination/Bias", 1, 60),
    ("Legal & Rights-Related Risks", "Privacy", 1, 72),
    ("Legal & Rights-Related Risks", "Criminal Activities", 3, 8),
]


def runs(seq):
    out, i = [], 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        out.append((seq[i], i, j))
        i = j
    return out


def main():
    row_h, gap = 0.62, 0.45
    ys, y, prev = [], 0.0, None
    for (l1, *_r) in ORIG:
        if prev is not None and l1 != prev:
            y -= gap
        ys.append(y)
        y -= row_h
        prev = l1

    x_root, x_l1, x_l2 = -1.6, 0.0, 4.0
    x_g, x_l = 8.6, 9.8               # level-3 / level-4 count columns
    top = max(ys) + 1.0

    fig_h = (max(ys) - min(ys)) * 0.5 + 2.6
    fig, ax = plt.subplots(figsize=(12.5, fig_h))

    def elbow(x0, y0, x1, y1, color, lw=0.9):
        xm = (x0 + x1) / 2
        ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color=color, lw=lw,
                solid_capstyle="round", zorder=1)

    l1_mid = {l1: sum(ys[a:b]) / (b - a) for (l1, a, b) in runs([r[0] for r in ORIG])}
    root_y = sum(l1_mid.values()) / len(l1_mid)

    # root -> L1
    for l1, yy in l1_mid.items():
        elbow(x_root, root_y, x_l1, yy, L1_COLORS[l1], lw=1.3)
    ax.plot([x_root], [root_y], "o", ms=9, color=INK, zorder=3)
    ax.text(x_root - 0.12, root_y, "AIR-BENCH\ntaxonomy", ha="right", va="center",
            fontsize=11, color=INK, weight="bold", linespacing=1.0)

    # L1 nodes
    for (l1, a, b) in runs([r[0] for r in ORIG]):
        c = L1_COLORS[l1]
        ax.plot([x_l1], [l1_mid[l1]], "o", ms=8, color=c, zorder=3)
        ax.text(x_l1 + 0.14, l1_mid[l1] + 0.14, l1.replace(" Risks", ""), ha="left",
                va="bottom", fontsize=9.5, color=INK, weight="bold")

    # L2 rows + counts
    for k, (l1, l2, n3, n4) in enumerate(ORIG):
        c = L1_COLORS[l1]
        elbow(x_l1, l1_mid[l1], x_l2, ys[k], c, lw=0.9)
        ax.plot([x_l2], [ys[k]], "o", ms=5, color=c, zorder=3)
        ax.text(x_l2 + 0.12, ys[k], l2, ha="left", va="center", fontsize=9.0, color=INK)
        ax.text(x_g, ys[k], str(n3), ha="center", va="center", fontsize=8.6, color=SUBTLE)
        ax.text(x_l, ys[k], str(n4), ha="center", va="center", fontsize=8.6, color=SUBTLE)

    # Column banner emphasizing the four tiers.
    chips = [(x_l1, "LEVEL 1", "4 categories"),
             (x_l2 + 0.1, "LEVEL 2", "16 subcategories"),
             (x_g, "LEVEL 3", "43 groups"),
             (x_l, "LEVEL 4", "314 leaf risks")]
    for x, a, b in chips:
        ha = "left" if x in (x_l1, x_l2 + 0.1) else "center"
        ax.text(x, top + 0.25, a, ha=ha, va="bottom", fontsize=8.5, color=INK, weight="bold")
        ax.text(x, top - 0.1, b, ha=ha, va="bottom", fontsize=7.8, color=SUBTLE)
    ax.plot([x_root, x_l + 0.6], [top - 0.35, top - 0.35], color="#dddddd", lw=0.8, zorder=0)

    # Title + leaf-contents note.
    ax.text(x_root - 0.12, top + 1.15, "Data representation: a depth-4 taxonomy tree",
            ha="left", va="bottom", fontsize=13.5, color=INK, weight="bold")
    ax.text(x_root - 0.12, top + 0.62,
            "Each leaf stores attack prompts, a judge prompt, associated policies, and a category "
            "summary; inner nodes store recursive summaries.",
            ha="left", va="bottom", fontsize=8.8, color=SUBTLE)

    ax.set_xlim(x_root - 1.7, x_l + 1.0)
    ax.set_ylim(min(ys) - 0.7, top + 1.9)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext, dpi in (("png", 220), ("pdf", None), ("svg", None)):
        out = OUT_STEM.with_suffix("." + ext)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()

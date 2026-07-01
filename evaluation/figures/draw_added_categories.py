#!/usr/bin/env python3
"""Figure: the 21 novel leaf-level risks AIR-BENCH Live added over AIR-BENCH 2024.

Node-link layout (top category -> subcategory -> novel leaf), colored by top category. The leaf
set was derived once as (our 335 leaves) - (original 314 leaves from stanford-crfm/air-bench-2024)
and is hard-coded here so the figure renders offline.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_STEM = Path(__file__).with_name("air_bench_added_categories")
L1_COLORS = {
    "System & Operational Risks": "#3E6D9C",
    "Content Safety Risks": "#4F8A6B",
    "Societal Risks": "#C08A3E",
    "Legal & Rights-Related Risks": "#8B5E83",
}
INK, SUBTLE, LINE = "#2b2b2b", "#6b7177", "#9aa0a6"

# (top-level category, subcategory, novel leaf)
NOVEL = [
    ("System & Operational Risks", "Security Risks", "Training data exfiltration"),
    ("System & Operational Risks", "Security Risks", "Model weight theft or compromise"),
    ("System & Operational Risks", "Security Risks", "Code compromise"),
    ("System & Operational Risks", "Security Risks", "Adversarial examples / prompts"),
    ("System & Operational Risks", "Security Risks", "Cyber attack coaching for non-experts"),
    ("System & Operational Risks", "Security Risks", "Autonomous cyber attacks"),
    ("System & Operational Risks", "Security Risks", "Sponge example attacks"),
    ("System & Operational Risks", "Security Risks", "Attack volume amplification"),
    ("Content Safety Risks", "Violence & Extremism", "Inciting violence or radicalization"),
    ("Content Safety Risks", "Hate / Toxicity", "Cyber-abuse material"),
    ("Content Safety Risks", "Sexual Content", "Degrading imagery"),
    ("Content Safety Risks", "Child Harm", "Virtual intimate relationships with minors"),
    ("Content Safety Risks", "Child Harm", "Publishing digital forgeries of minors"),
    ("Content Safety Risks", "Child Harm", "Sexual encouragement of minors"),
    ("Societal Risks", "Political Usage", "Election-related digital replica use"),
    ("Societal Risks", "Political Usage", "Deceptive synthetic media harming democracy"),
    ("Societal Risks", "Political Usage", "Criminal justice deception via synthetic media"),
    ("Societal Risks", "Economic Harm", "Extortion"),
    ("Societal Risks", "Manipulation", "Identity verification evasion"),
    ("Legal & Rights-Related Risks", "Privacy", "Nudging children to disclose personal data"),
    ("Legal & Rights-Related Risks", "Privacy", "Manipulative weakening of child privacy"),
]


def runs(seq):
    """Yield (value, start, end) for maximal contiguous equal-key spans."""
    out, i = [], 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        out.append((seq[i], i, j))
        i = j
    return out


def main():
    # y for each leaf row, top-to-bottom, with a gap between top categories.
    ys, y, prev = [], 0.0, None
    for (l1, _l2, _l4) in NOVEL:
        if prev is not None and l1 != prev:
            y -= 0.7
        ys.append(y)
        y -= 1.0
        prev = l1

    x_l1, x_l2, x_leaf = 0.0, 2.2, 4.4
    fig_h = (max(ys) - min(ys)) * 0.42 + 1.8
    fig, ax = plt.subplots(figsize=(12.5, fig_h))

    def elbow(x0, y0, x1, y1, color, lw=0.9):
        xm = (x0 + x1) / 2
        ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color=color, lw=lw,
                solid_capstyle="round", zorder=1)

    # Precompute each top category's vertical midpoint.
    l1_mid = {l1: sum(ys[a:b]) / (b - a) for (l1, a, b) in runs([n[0] for n in NOVEL])}

    # L1 nodes
    for (l1, a, b) in runs([n[0] for n in NOVEL]):
        c = L1_COLORS[l1]
        y_l1 = l1_mid[l1]
        ax.plot([x_l1], [y_l1], "o", ms=8, color=c, zorder=3)
        ax.text(x_l1 - 0.15, y_l1 + 0.16, l1, ha="right", va="bottom", fontsize=10.0,
                color=INK, weight="bold")
        ax.text(x_l1 - 0.15, y_l1 - 0.18, f"+{b - a} leaves", ha="right", va="top",
                fontsize=8.0, color=SUBTLE)

    # L2 nodes + L1->L2 and L2->leaf connectors
    for (key, a, b) in runs([(n[0], n[1]) for n in NOVEL]):
        l1, l2 = key
        c = L1_COLORS[l1]
        y_l2 = sum(ys[a:b]) / (b - a)
        elbow(x_l1, l1_mid[l1], x_l2, y_l2, c, lw=1.0)
        ax.plot([x_l2], [y_l2], "o", ms=5.5, color=c, zorder=3)
        ax.text(x_l2 + 0.12, y_l2, l2, ha="left", va="center", fontsize=8.8, color=INK,
                weight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
        for k in range(a, b):
            elbow(x_l2, y_l2, x_leaf, ys[k], c, lw=0.8)

    # Leaf tips + labels
    for k, (l1, _l2, l4) in enumerate(NOVEL):
        c = L1_COLORS[l1]
        ax.plot([x_leaf], [ys[k]], "o", ms=3.5, color=c, zorder=3)
        ax.text(x_leaf + 0.12, ys[k], l4, ha="left", va="center", fontsize=8.6, color=INK)

    ax.text(x_l1 - 0.15, max(ys) + 1.5,
            "What AIR-BENCH Live added", ha="left", va="bottom", fontsize=14,
            color=INK, weight="bold")
    ax.text(x_l1 - 0.15, max(ys) + 0.9,
            "21 new leaf-level risks from 2024--2026 regulations, by category.",
            ha="left", va="bottom", fontsize=9.5, color=SUBTLE)

    ax.set_xlim(-3.4, x_leaf + 6.6)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 2.1)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext, dpi in (("png", 220), ("pdf", None), ("svg", None)):
        out = OUT_STEM.with_suffix("." + ext)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()

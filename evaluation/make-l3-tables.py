#!/usr/bin/env python3
"""Emit LaTeX tables of safety scores broken down by level-3 category.

Outputs two .tex files under evaluation/tables/ (and prints them):
  1. gpt4o_comparison_by_l3.tex  -- GPT-4o, original AIR-BENCH 2024 vs AIR-BENCH Live (English).
  2. eval_by_l3.tex              -- full leaderboard, all evaluated models (multilingual), with
                                    placeholder columns for the Anthropic models (no API key yet).

Reads the per-model result CSVs in evaluation/results/. Each is the multilingual run
(split-base-variant, 2 prompts/category x 4 languages = 8 rows/category).
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
OUT = ROOT / "evaluation" / "tables"
OUT.mkdir(exist_ok=True)
TREE = json.loads((ROOT / "tree" / "semantic-tree.json").read_text())

# Evaluated models (file, display name). Claude models are placeholders for now.
MODELS = [
    ("openai_gpt-5.5.csv", "GPT-5.5"),
    ("openai_gpt-4o.csv", "GPT-4o"),
    ("openrouter_google_gemini-2.5-pro.csv", "Gemini 2.5 Pro"),
    ("openrouter_google_gemini-2.5-flash.csv", "Gemini 2.5 Flash"),
    ("openrouter_x-ai_grok-4.3.csv", "Grok 4.3"),
    ("openrouter_deepseek_deepseek-v3.2.csv", "DeepSeek V3.2"),
    ("openrouter_deepseek_deepseek-r1.csv", "DeepSeek R1"),
    ("openrouter_qwen_qwen3-235b-a22b-2507.csv", "Qwen3-235B"),
    ("openrouter_moonshotai_kimi-k2.csv", "Kimi K2"),
    ("openrouter_mistralai_mistral-large.csv", "Mistral Large"),
    ("openrouter_meta-llama_llama-3.3-70b-instruct.csv", "Llama 3.3 70B"),
    ("openrouter_meta-llama_llama-3-8b-instruct.csv", "Llama 3 8B"),
]
# Anthropic models: filenames run-eval.py will write once CLAUDE_API_KEY is set. They render as
# placeholders until those CSVs exist, then auto-fill on the next run of this script.
CLAUDE = [
    ("claude_claude-opus-4-8.csv", "Claude Opus 4.8"),
    ("claude_claude-haiku-4-5.csv", "Claude Haiku 4.5"),
]
PLACEHOLDER = "--"
# Models whose score reflects API-level input moderation (all prompts blocked before inference)
# rather than model refusal. Flagged with a dagger and a caption note.
MODERATION_BLOCKED = {"GPT-5.5"}


def kids(n):
    return n.get("children") or []


def l3_order():
    """Ordered list of (l1, l2, l3) and an l3-name -> (l1,l2,l3) key map."""
    order = []
    for l1 in kids(TREE):
        for l2 in kids(l1):
            for l3 in kids(l2):
                order.append((l1["name"], l2["name"], l3["name"]))
    return order


def score(r):
    try:
        return float(r["score"])
    except (TypeError, ValueError):
        return None


def per_l3(path, english_only=False, eng=None):
    """(l2,l3) -> mean score, plus overall."""
    by = defaultdict(list)
    alls = []
    for r in csv.DictReader(open(path)):
        if english_only and r["prompt"] not in eng:
            continue
        s = score(r)
        if s is None:
            continue
        by[(r["l2-name"], r["l3-name"])].append(s)
        alls.append(s)
    means = {k: statistics.mean(v) for k, v in by.items()}
    return means, (statistics.mean(alls) if alls else None)


def esc(s):
    return s.replace("&", r"\&").replace("%", r"\%")


def fmt(x):
    return f"{x:.2f}" if x is not None else PLACEHOLDER


def main():
    order = l3_order()
    eng = set(r["prompt"] for r in csv.DictReader(open(ROOT / "tree" / "air_bench_prompts_english.csv")))

    # ---- Table 1: GPT-4o comparison by L3 (English) ----
    live_m, live_o = per_l3(RESULTS / "openai_gpt-4o.csv", english_only=True, eng=eng)
    leg_m, leg_o = per_l3(RESULTS / "legacy_openai_gpt-4o.csv")
    lines = [
        r"\begin{table}[h]\centering\footnotesize",
        r"\caption{GPT-4o mean safety score by level-3 category: original AIR-BENCH 2024 vs.\ "
        r"AIR-BENCH Live, \textbf{English prompts only} (the original benchmark is English-only), "
        r"same GPT 5.4-mini judge; higher = safer. Per-category $n$ is small, so the Overall row is "
        r"the robust comparison.}",
        r"\label{tab:gpt4o-l3}",
        r"\begin{tabular}{@{}p{6.2cm}ccc@{}}",
        r"\hline",
        r"Level-3 category & AIR-BENCH 2024 & AIR-BENCH Live & $\Delta$ \\",
        r"\hline",
    ]
    cur = None
    for (l1, l2, l3) in order:
        if l1 != cur:
            lines.append(r"\multicolumn{4}{l}{\textit{" + esc(l1) + r"}} \\")
            cur = l1
        old = leg_m.get((l2, l3))
        new = live_m.get((l2, l3))
        d = (new - old) if (old is not None and new is not None) else None
        ds = f"{d:+.2f}" if d is not None else PLACEHOLDER
        lines.append(f"\\quad {esc(l3)} & {fmt(old)} & {fmt(new)} & {ds} \\\\")
    lines += [
        r"\hline",
        f"\\textbf{{Overall}} & {fmt(leg_o)} & {fmt(live_o)} & {f'{live_o-leg_o:+.2f}'} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT / "gpt4o_comparison_by_l3.tex").write_text("\n".join(lines))

    # ---- Table 2: full leaderboard by L3 (multilingual) ----
    # A model with a results file is evaluated; one without renders as a placeholder column.
    model_means, model_overall, present = {}, {}, set()
    for fn, name in MODELS + CLAUDE:
        path = RESULTS / fn
        if path.exists():
            model_means[name], model_overall[name] = per_l3(path)
            present.add(name)
    ordered_names = [n for _, n in MODELS + CLAUDE]
    ranked = sorted((n for n in ordered_names if n in present),
                    key=lambda n: model_overall[n], reverse=True)
    pending = [n for n in ordered_names if n not in present]
    cols = ranked + pending

    def colhead(c):
        return esc(c) + (r"$^\dagger$" if c in MODERATION_BLOCKED else "")
    head = " & ".join([r"Level-3 category"] + [r"\rotatebox{90}{" + colhead(c) + "}" for c in cols])
    lines2 = [
        r"\begin{table}[h]\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Mean safety score by level-3 category on AIR-BENCH Live, \textbf{all four "
        r"languages} (higher = safer; GPT 5.4-mini judge). Columns ordered by overall score; "
        r"Anthropic models pending evaluation. $^\dagger$GPT-5.5's prompts were all blocked by "
        r"OpenAI's API moderation before inference, so its score reflects product-level input "
        r"filtering rather than model refusal.}",
        r"\label{tab:eval-l3}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\hline",
        head + r" \\",
        r"\hline",
    ]
    cur = None
    for (l1, l2, l3) in order:
        if l1 != cur:
            lines2.append(r"\multicolumn{" + str(len(cols) + 1) + r"}{l}{\textit{" + esc(l1) + r"}} \\")
            cur = l1
        cells = [fmt(model_means[c].get((l2, l3))) if c in present else PLACEHOLDER for c in cols]
        lines2.append(f"\\quad {esc(l3)} & " + " & ".join(cells) + r" \\")
    overall_cells = [fmt(model_overall[c]) if c in present else PLACEHOLDER for c in cols]
    lines2 += [
        r"\hline",
        r"\textbf{Overall} & " + " & ".join(overall_cells) + r" \\",
        r"\hline",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    (OUT / "eval_by_l3.tex").write_text("\n".join(lines2))

    print("=== overall safety (ranking) ===")
    for n in ranked:
        print(f"  {n:18s} {model_overall[n]:.3f}")
    print(f"\nwrote {OUT/'gpt4o_comparison_by_l3.tex'}")
    print(f"wrote {OUT/'eval_by_l3.tex'}")


if __name__ == "__main__":
    main()

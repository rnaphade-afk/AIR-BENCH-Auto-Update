#!/usr/bin/env python3
"""Per-model new-vs-original comparison table.

For each model, compute the mean safety score (GPT 5.4-mini judge; higher = safer) on the original
AIR-BENCH 2024 prompts (evaluation/results/legacy_*.csv) and on AIR-BENCH Live (the English subset
of the full multilingual run), then emit a LaTeX table sorted by the Live score. English-only on
both sides, since the original benchmark is English-only. Models without a legacy run yet show "--".
"""
import csv
import statistics
from pathlib import Path

RES = Path("evaluation/results")
ENG = set(r["prompt"] for r in csv.DictReader(open("tree/air_bench_prompts_english.csv")))
OUT = Path("evaluation/tables/newvsold_by_model.tex")

# (display, new-run file, legacy-run file)
MODELS = [
    ("GPT-5.5",          "openai_gpt-5.5.csv",                             "legacy_openai_gpt-5.5.csv"),
    ("GPT-4o",           "openai_gpt-4o.csv",                              "legacy_openai_gpt-4o.csv"),
    ("Grok 4.3",         "openrouter_x-ai_grok-4.3.csv",                   "legacy_openrouter_x-ai_grok-4.3.csv"),
    ("Gemini 2.5 Pro",   "openrouter_google_gemini-2.5-pro.csv",           "legacy_openrouter_google_gemini-2.5-pro.csv"),
    ("Gemini 2.5 Flash", "openrouter_google_gemini-2.5-flash.csv",         "legacy_openrouter_google_gemini-2.5-flash.csv"),
    ("Qwen3-235B",       "openrouter_qwen_qwen3-235b-a22b-2507.csv",       "legacy_openrouter_qwen_qwen3-235b-a22b-2507.csv"),
    ("Llama 3.3 70B",    "openrouter_meta-llama_llama-3.3-70b-instruct.csv","legacy_openrouter_meta-llama_llama-3.3-70b-instruct.csv"),
    ("Llama 3 8B",       "openrouter_meta-llama_llama-3-8b-instruct.csv",  "legacy_openrouter_meta-llama_llama-3-8b-instruct.csv"),
    ("Kimi K2",          "openrouter_moonshotai_kimi-k2.csv",              "legacy_openrouter_moonshotai_kimi-k2.csv"),
    ("DeepSeek V3.2",    "openrouter_deepseek_deepseek-v3.2.csv",          "legacy_openrouter_deepseek_deepseek-v3.2.csv"),
    ("DeepSeek R1",      "openrouter_deepseek_deepseek-r1.csv",            "legacy_openrouter_deepseek_deepseek-r1.csv"),
    ("Mistral Large",    "openrouter_mistralai_mistral-large.csv",         "legacy_openrouter_mistralai_mistral-large.csv"),
    ("Claude Opus 4.8",  "claude_claude-opus-4-8.csv",                     "legacy_claude_claude-opus-4-8.csv"),
    ("Claude Haiku 4.5", "claude_claude-haiku-4-5.csv",                    "legacy_claude_claude-haiku-4-5.csv"),
]
DAGGER = {"GPT-5.5"}


def mean_score(path: Path, english_only: bool):
    if not path.exists():
        return None
    ss = []
    for r in csv.DictReader(path.open()):
        if english_only and r["prompt"] not in ENG:
            continue
        try:
            ss.append(float(r["score"]))
        except ValueError:
            pass
    return statistics.mean(ss) if ss else None


def main():
    rows = []
    for disp, newf, legf in MODELS:
        new = mean_score(RES / newf, english_only=True)     # Live: English subset
        old = mean_score(RES / legf, english_only=False)    # original is already English
        rows.append((disp, old, new))

    fmt = lambda x: "--" if x is None else f"{x:.3f}"
    print(f"{'Model':18s} {'Original':>9} {'Live':>9} {'Δ':>8}")
    for disp, old, new in rows:
        d = "--" if (old is None or new is None) else f"{new - old:+.3f}"
        print(f"{disp:18s} {fmt(old):>9} {fmt(new):>9} {d:>8}")

    def name(d):
        return d + (r"$^\dagger$" if d in DAGGER else "")

    def cell(old, new, which):
        x = old if which == "old" else new
        return "--" if x is None else f"{x:.2f}"

    def delta(old, new):
        return "--" if (old is None or new is None) else f"${new - old:+.2f}$"

    body = "\n".join(
        rf"{name(d)} & {cell(o,n,'old')} & {cell(o,n,'new')} & {delta(o,n)} \\"
        for d, o, n in rows
    )
    tex = "\n".join([
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Model & AIR-BENCH 2024 & AIR-BENCH Live & $\Delta$ \\",
        r"\hline",
        body,
        r"\hline",
        r"\end{tabular}",
        r"\caption{Mean safety score (higher = safer; GPT 5.4-mini judge) on the original "
        r"AIR-BENCH 2024 prompts vs.\ AIR-BENCH Live, \textbf{English prompts only} on both sides. "
        r"$\Delta<0$ indicates the modernized prompts are more challenging. "
        r"$^\dagger$GPT-5.5's prompts are blocked by API moderation before inference.}",
        r"\label{tab:newvsold}",
        r"\end{table}",
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex + "\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()

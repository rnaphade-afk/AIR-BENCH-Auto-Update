#!/usr/bin/env python3
"""Evaluate a model on the ORIGINAL AIR-BENCH 2024 prompts, for apples-to-apples comparison
against AIR-BENCH Live.

The original prompts are pulled from the HuggingFace dataset `stanford-crfm/air-bench-2024`.
Because AIR-BENCH Live added leaves (which renumbers `cate-idx`), each original prompt is remapped
to our `cate-idx` by its (l2, l3, l4) leaf name, so the judge lookup hits the correct template.
Our judge CSV inherited the AIR-BENCH judge prompts for legacy leaves, so the judge here is
identical to the one used by run-eval.py on the new benchmark.

This script does NOT modify run-eval.py; it imports and reuses its target/judge/scoring machinery
(`run_target`, `get_judge_client`, `load_judge_templates`) so the legacy run is byte-for-byte the
same evaluation, only the prompt source differs.

Usage:
  python evaluation/eval-legacy.py --openai_models gpt-4o \
      --judge-model gpt-5.4-mini --sample-per-category 2 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Reuse run-eval.py's machinery without importing by name (the filename is hyphenated).
_RUNEVAL_PATH = Path(__file__).with_name("run-eval.py")
_spec = importlib.util.spec_from_file_location("run_eval", _RUNEVAL_PATH)
re_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(re_mod)

DEFAULT_DATASET = "stanford-crfm/air-bench-2024"
DEFAULT_JUDGE_CSV = "tree/air_bench_judge_prompts.csv"


def leaf_to_cate_idx(judge_csv: Path) -> Dict[tuple, str]:
    """(l2-name, l3-name, l4-name) -> our cate-idx, from the judge CSV."""
    mapping: Dict[tuple, str] = {}
    with judge_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mapping[(r["l2-name"], r["l3-name"], r["l4-name"])] = r["cate-idx"]
    return mapping


def load_legacy_rows(dataset: str, leaf2cdx: Dict[tuple, str],
                     sample_per_category, seed: int, limit) -> List[Dict[str, str]]:
    """Load original prompts, remap each to our cate-idx by leaf, and (optionally) take a
    reproducible random sample of N prompts per category (the original set is English-only, so no
    language bundling). Prompts whose leaf no longer exists in our taxonomy are dropped."""
    from datasets import load_dataset

    ds = load_dataset(dataset)["test"]
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    dropped = 0
    for row in ds:
        cdx = leaf2cdx.get((row["l2-name"], row["l3-name"], row["l4-name"]))
        if cdx is None:
            dropped += 1
            continue
        grouped[cdx].append({
            "cate-idx": cdx, "l2-name": row["l2-name"], "l3-name": row["l3-name"],
            "l4-name": row["l4-name"], "prompt": row["prompt"],
        })

    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    for cdx in sorted(grouped, key=lambda i: [int(p) if p.isdigit() else p for p in i.split(".")]):
        bucket = grouped[cdx]
        if sample_per_category is not None and len(bucket) > sample_per_category:
            bucket = rng.sample(bucket, sample_per_category)
        rows.extend(bucket)
    if limit is not None:
        rows = rows[:limit]
    re_mod.log(f"Legacy prompts: kept {sum(len(v) for v in grouped.values())} across "
               f"{len(grouped)} leaves ({dropped} dropped as no-longer-present); "
               f"evaluating {len(rows)} after sampling.")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--openrouter_models", nargs="+", default=[], metavar="MODEL")
    p.add_argument("--openai_models", nargs="+", default=[], metavar="MODEL")
    p.add_argument("--claude_models", nargs="+", default=[], metavar="MODEL")
    p.add_argument("--judge-model", default=re_mod.DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge-csv", default=DEFAULT_JUDGE_CSV)
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset id for the original AIR-BENCH prompts.")
    p.add_argument("--out-dir", default="evaluation/results")
    p.add_argument("--sample-per-category", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=re_mod.MAX_CONCURRENCY)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    targets = ([("openrouter", m) for m in args.openrouter_models]
               + [("openai", m) for m in args.openai_models]
               + [("claude", m) for m in args.claude_models])
    if not targets:
        sys.exit("No target models. Pass --openai_models / --openrouter_models / --claude_models.")

    judge_csv = Path(args.judge_csv)
    if not judge_csv.exists():
        sys.exit(f"judge CSV not found: {judge_csv}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_templates = re_mod.load_judge_templates(judge_csv)
    leaf2cdx = leaf_to_cate_idx(judge_csv)
    base_rows = load_legacy_rows(args.dataset, leaf2cdx, args.sample_per_category, args.seed, args.limit)
    re_mod.log(f"Targets ({len(targets)}): " + ", ".join(f"{p}:{m}" for p, m in targets))

    judge_client = re_mod.get_judge_client()
    for provider, model in targets:
        safe = f"legacy_{provider}_{model}".replace("/", "_").replace(":", "_")
        out_path = out_dir / f"{safe}.csv"
        if out_path.exists() and not args.no_resume:
            _strip_unscored(out_path)  # drop transient-error rows so resume retries them
        re_mod.run_target(provider, model, base_rows, judge_templates, judge_client,
                          args.judge_model, out_path, args.concurrency, args.no_resume)


def _strip_unscored(path: Path) -> None:
    """Remove rows with no numeric score (transient target/judge errors) so a resumed run retries
    them instead of treating them as done. Content-policy refusals are recorded with a real score,
    so this only re-attempts genuine failures."""
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return
    kept = [r for r in rows if r.get("score") not in ("", "None", None)]
    if len(kept) != len(rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(kept)
        re_mod.log(f"  {path.name}: dropped {len(rows) - len(kept)} error rows for retry.")


if __name__ == "__main__":
    main()

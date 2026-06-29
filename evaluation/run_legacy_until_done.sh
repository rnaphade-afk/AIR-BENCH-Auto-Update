#!/bin/bash
# Self-healing driver for the legacy (original AIR-BENCH) evaluation: re-runs eval-legacy.py
# (which resumes from each model's CSV) until all 12 models reach 628 scored rows, so a transient
# death just restarts where it left off. Run under caffeinate to also block machine sleep.
cd /Users/ro/Desktop/Code/Virtue/AIR-BENCH-Auto-Update || exit 1
PY=venv/bin/python
MODELS_OR="deepseek/deepseek-r1 deepseek/deepseek-v3.2 google/gemini-2.5-flash google/gemini-2.5-pro meta-llama/llama-3-8b-instruct meta-llama/llama-3.3-70b-instruct mistralai/mistral-large moonshotai/kimi-k2 qwen/qwen3-235b-a22b-2507 x-ai/grok-4.3"
MODELS_OAI="gpt-5.5 gpt-4o"
MODELS_CLAUDE="claude-opus-4-8 claude-haiku-4-5"

for attempt in $(seq 1 100); do
  echo "[watchdog] attempt $attempt starting $(date)"
  $PY evaluation/eval-legacy.py \
    --openrouter_models $MODELS_OR \
    --openai_models $MODELS_OAI \
    --claude_models $MODELS_CLAUDE \
    --judge-model gpt-5.4-mini --sample-per-category 2 --seed 0 --concurrency 10
  done=$($PY -c "import csv,glob; print(sum(1 for f in glob.glob('evaluation/results/legacy_*.csv') if sum(1 for r in csv.DictReader(open(f)) if r['score'] not in ('','None'))>=620))")
  echo "[watchdog] attempt $attempt finished: $done/12 models complete"
  if [ "$done" -ge 14 ]; then
    echo "[watchdog] all 14 models complete — generating table"
    $PY evaluation/make-newvsold-table.py
    break
  fi
  sleep 5
done
echo "[watchdog] exiting $(date)"

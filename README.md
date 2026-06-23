# AIR-BENCH Auto Update

A pipeline for automatic maintenance of the AIR-BENCH dataset. It contains functionality for scraping policy sources, filtering and classifying new clauses into a semantic tree, generating attack/judge prompts for novel cateogires, and exporting the updated dataset.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file:

```bash
OPENAI_API_KEY=...

# Optional: translation runs on Qwen 3.7 via OpenRouter when set; otherwise falls back to the OpenAI model.
QWEN_API_KEY=...

# Optional: Congress API discovery (bill text is fetched from the govinfo.gov mirror)
CONGRESS_API_KEY=...

# Optional: EUR-Lex access via LexAPI (lex-api.com). Without it, EUR-Lex is skipped — the
# site bot-walls direct scraping. Free tier is 50 metered calls/day (only document-content
# retrievals are metered; search is free).
LEX_API_KEY=...

# Optional: PersonaHub is sampled for persona-prompting. To use a local persona list instead:
PERSONA_SOURCE_PATH=path/to/personas.txt
```

> All commands should be run from the repo root using the repo-local virtual environment (`venv/bin/python ...`).

> Several sources (NIST, AI Verify, and OECD-linked documents) are PDFs. PDF text extraction uses
> `pypdf`; install **poppler** (`brew install poppler` on macOS, `apt-get install poppler-utils` on
> Debian/Ubuntu) to enable the `pdftotext` fallback for PDFs `pypdf` cannot parse.

---

## Quick Reference

| Goal | Command |
|---|---|
| First-time tree initialization | `./setup.py` |
| Rebuild tree with prompt generation (EXPENSIVE) | `./setup.py --generate-prompts` |
| Scrape, update, and export | `venv/bin/python pipeline.py --scrape --export` |
| Run on an existing policy JSON | `venv/bin/python pipeline.py "POLICY_JSON_PATH"` |
| Export only | `venv/bin/python pipeline.py --export` |
| Classify only, then stop for review | `venv/bin/python pipeline.py --scrape --parallel-policies --classify-only` |
| Apply reviewed classifications + export | `venv/bin/python pipeline.py --scrape --resume --yes --export` |
| Resume an interrupted run | `venv/bin/python pipeline.py --scrape --export --resume --run-dir pipeline-runs/<timestamp>` |
| Launch the web interface | `venv/bin/python webapp/server.py` |
| Proof-of-concept run | See [Proof of Concept](#proof-of-concept) |

---

## Commands

### `setup.py` — Tree Initialization

Use `setup.py` ONLY when initializing or fully rebuilding `tree/semantic-tree.json`. It rewrites the tree, so do not use it for routine updates.

```bash
# Initialize tree and regenerate summaries
./setup.py

# Initialize tree, regenerate all leaf prompts, then regenerate summaries
./setup.py --generate-prompts

# Resume an interrupted setup run
./setup.py --generate-prompts --resume --run-dir pipeline-runs/<timestamp>
```

### `pipeline.py` — Incremental Updates

Use `pipeline.py` for all normal incremental policy updates.

```bash
# Scrape, update, and export
venv/bin/python pipeline.py --scrape --export

# Run on an existing policy JSON (no scrape)
venv/bin/python pipeline.py "POLICY_JSON_PATH"

# Run on an existing policy JSON and export
venv/bin/python pipeline.py "POLICY_JSON_PATH" --export

# Export only (no scrape or update)
venv/bin/python pipeline.py --export

# Limit scraping to one or more specific sources
venv/bin/python pipeline.py --scrape --export --scrape-source "SOURCE_NAME"

# List available scraper sources
venv/bin/python webscraper/multisource_lm_policy_scrape.py --list-sources

# Apply a prompt mutation (currently only authority_endorsement is supported)
venv/bin/python pipeline.py POLICY_JSON_PATH --mutation-type MUTATION_TYPE
```

#### Classification controls

For large runs it is best to classify first, review the matches and (typically over-generated)
novel proposals, and only then apply them — applying is irreversible (it mutates the tree and
generates prompts for every novel leaf).

```bash
# Classify every policy and STOP before touching the tree. Writes one
# pipeline-runs/<run>/policy-NNN-classification.json per policy for review.
venv/bin/python pipeline.py --scrape --run-dir pipeline-runs/<run> --parallel-policies --classify-only

# (review/prune the policy-*-classification.json files — see Classification Review below)

# Apply the reviewed classifications, generate prompts for novel leaves, and export.
venv/bin/python pipeline.py --scrape --run-dir pipeline-runs/<run> --resume --yes --export
```

- `--classify-only` — classify all policies, write their classification JSONs, then stop before applying anything.
- `--parallel-policies` — classify policies concurrently (routing only reads the tree, so this is safe); much faster for large runs.
- `--yes` — run unattended, skipping every human-review pause. The critic/refiner and reconciliation
  loops are the quality layer in this mode.

---

## Web Interface

A browser front-end for orchestrating both `setup.py` and `pipeline.py`, with an integrated JSON editor for the human-in-the-loop review checkpoints. It is a thin driver around the existing CLIs — it spawns them as subprocesses and reuses the same review/resume flow, so behavior is identical to running them in a terminal.

```bash
# Install dependencies (adds flask) and launch the server
venv/bin/pip install -r requirements.txt
venv/bin/python webapp/server.py        # http://127.0.0.1:5000

# Use a different port
PORT=8080 venv/bin/python webapp/server.py
```

Then, in the browser:

1. Pick a tool (`pipeline.py` or `setup.py`) and set flags (toggles cover the common ones; the text box accepts any CLI flag).
2. **Start run.** Logs stream live in the bottom panel.
3. When the run reaches a review checkpoint it pauses, opens the relevant JSON file in the editor, and shows a banner. Edit the file (it is validated as JSON), then **Save & Continue** to resume the run.
4. Repeat until the run reports `done`. Other checkpoint files for the run are listed under "Run-dir checkpoints" and can be opened at any time.

Notes:

- The same review files documented in [Policy Review](#policy-review) and under `pipeline-runs/<timestamp>/` are what the editor opens — the web interface only changes *how* you edit them, not *what* gets reviewed.
- One run at a time. File reads/writes are sandboxed to the repository root, so a `--run-dir` outside the repo will not be editable in the browser.
- `--yes` is never injected; the pause is how the interface lets you edit. The editor (CodeMirror) loads from a CDN and needs internet for its assets. See [webapp/README.md](webapp/README.md) for details.

---

## Pipeline Steps

Running `pipeline.py --scrape --export` executes the following steps:

1. Webscraping across configured policy sources
2. Filtering to policies not seen in previous runs
3. Human review of newly scraped policies
4. Classification into the semantic tree
5. Review checkpoints for classifications and novel leaves
6. Prompt and judge generation for novel leaves
7. Semantic tree update
8. Dataset CSV export

Run artifacts are written to `pipeline-runs/<timestamp>/` by default. Override with `--run-dir`.

---

## Policy Sources

Sixteen sources are configured (`--list-sources` to print them): Congress.gov, Federal Register,
California Legislature, EUR-Lex, EU AI Office, UK AISI, CAC China, NIST AI, IMDA Singapore, AI Verify
Foundation, METI Japan AI Policy, MSIT Korea, Korea Law Information Center, Parliament of Canada
LegisINFO, ISED Canada AI, and OECD AI Policy Observatory.

Most sources are crawled as ordinary HTML (a bounded breadth-first crawl that only follows
AI/model-topical links, to avoid wandering into unrelated legislation). Several need dedicated access
paths because the site blocks scraping or renders content dynamically:

- **EUR-Lex** — bot-walls direct requests, so it is fetched through **LexAPI** (`lex-api.com`, `LEX_API_KEY`). The AI Act is seeded directly; full-text search augments discovery. Free tier = 50 metered calls/day.
- **Congress.gov** — blocks this environment's IP, so a curated list of AI bills is resolved via the Congress API and their text is fetched from the **govinfo.gov** mirror (needs `CONGRESS_API_KEY`).
- **Federal Register** — document pages are JS shells; full rule text is pulled from the API's `raw_text_url` field via relevance-ordered, harm-targeted queries.
- **Korea Law Information Center** — statute bodies load via an internal endpoint, which is called directly to get the English article text (e.g. Korea's AI Basic Act).
- **OECD AI Policy Observatory** — a catalog of national AI policies; each entry's linked official PDFs (on the originating gov hosts) are followed and extracted. Also serves as the practical source of Japanese coverage.
- **METI Japan** — geo-blocked at the IP level (free proxies do not get through); covered indirectly via OECD.

Extraction uses a recall-oriented keyword pre-filter followed by a per-clause LLM KEEP/DROP gate.
Each scrape writes a `webscraper-report.json` with a per-source funnel
(`funnel_by_source`: pages → extractable pages → raw candidates → post-filter → final) so coverage
gaps are visible at a glance.

---

## Policy Review

Each scrape run produces three files:

```
pipeline-runs/<timestamp>/webscraped-policies-all.json
pipeline-runs/<timestamp>/webscraped-policies-new.json
pipeline-runs/<timestamp>/webscraped-policies-new-review.json
```

The review file is generated after novelty filtering. To review:

- **Remove** any policies that should not enter classification.
- **Edit** metadata as needed.
- **Leave** approved records in the `policies` list.

Approved policies are written back to `webscraped-policies-new.json` and fed into classification. Novelty is based on normalized clause text — a previously seen clause from a new URL will be skipped.

---

## Classification Review

Each policy is classified into `pipeline-runs/<run>/policy-NNN-classification.json`, containing
`existing_matches` (fragments mapped to current leaves) and `novel_categories` (proposed new leaves).
Applying these is irreversible — every novel category triggers a full prompt-generation chain — so for
non-trivial runs use `--classify-only` to generate all classification JSONs, review them, and only then
resume to apply.

The reconciliation agent leans slightly toward proposing novel categories (it is given the full
cross-branch leaf catalog and told to prefer existing leaves, never emit catch-all/"general" or
defensive categories, and dedupe near-identical proposals), so expect to **prune** novel proposals:

- Remove proposals that duplicate an existing leaf (in any branch) or another proposal.
- Remove vague/catch-all ("Other …", "General …") or defensive/governance proposals that slipped through.
- Keep genuinely new harms (new mechanism, technology, or victim context) — convert the rest to
  `existing_matches` (with the right `node_id`) or delete them.

Re-running with `--resume` reads the edited JSONs and does **not** re-classify. A backup of the
unedited classifications is handy before bulk-pruning.

---

## Policy History

By default, the pipeline checks for previously seen policies in:

```
pipeline-runs/
webscraper/runs/
webscraper/lm_policy_clauses_multisource.json
```

```bash
# Add extra history files or directories
--previous-policies path/to/old.json
--previous-policies-dir path/to/old-runs

# Disable default history lookup
--no-default-policy-history
```

---

## Prompt Generation Controls

Base attack prompts are generated, then a subset is carried forward to mutation, translation, and
storage. These flags apply to both `setup.py --generate-prompts` and `pipeline.py` (novel categories):

- `--base-count` — how many base prompt candidates to **generate** per category (default `8`).
- `--base-select` — how many of those to **carry forward** (default `5`, matching AIR-BENCH's 5 base
  prompts per category). With `--yes` the first `--base-select` are kept automatically; otherwise edit
  `selected_base_prompts` in the base-prompts review file.
- `--base-review-rounds`, `--mutation-review-rounds`, `--translation-review-rounds` — critic/refiner
  iterations per stage (default `1` each).
- `--mutation-type` — attack mutation(s) to apply after base prompts (repeatable; default
  `authority_endorsement`).

```bash
# Generate 12 base candidates per category, keep the best 6
venv/bin/python pipeline.py --scrape --export --base-count 12 --base-select 6
```

---

## Translation Languages

Reviewed attack prompts are translated into additional languages and stored on each leaf as
extra language-tagged prompt records (the English originals are kept). This applies to both
`pipeline.py` (new/novel categories) and `setup.py --generate-prompts` (full rebuild).

Choose the target languages with `--translation-language`, which accepts **ISO 639-1 codes**
(e.g. `es`, `ja`, `pt`) or full language names (e.g. `Spanish`); both are normalized to a
canonical language name. The flag is repeatable. If omitted, it defaults to Spanish, Japanese,
and Portuguese.

```bash
# ISO codes (repeat the flag per language)
venv/bin/python pipeline.py --scrape --export --translation-language es --translation-language fr

# Names also work
./setup.py --generate-prompts --translation-language Spanish --translation-language German
```

Supported ISO codes are listed in `LANGUAGE_BY_ISO` in `prompt-generation/generate-prompts.py`;
unknown values are passed through to the model unchanged. `--translation-review-rounds` controls
the translation critic/refiner loop (default 1).

**Translation model.** Translations are generated by **Qwen 3.7 via OpenRouter** when `QWEN_API_KEY`
is set (default model `qwen/qwen3.7-plus`; override with the `TRANSLATION_MODEL` env var, e.g.
`qwen/qwen3.7-max`). If `QWEN_API_KEY` is absent, translation falls back to the default OpenAI model.
The translation critic/refiner review still runs on the OpenAI critic model — only the translation
generation/revision uses Qwen, so that work bills to OpenRouter rather than OpenAI. Qwen's reasoning
output is disabled (`reasoning.enabled=false`): it adds no translation-quality benefit but ~12× the
billed tokens, so with it off `qwen3.7-plus` matches `gpt-5.4-mini` on fidelity/language at ~⅓ the cost.

---

## Export Outputs

The exporter computes category IDs from tree order. Default output paths:

```
tree/air_bench_prompts_default.csv
tree/air_bench_prompts_china.csv
tree/air_bench_prompts_eu.csv
tree/air_bench_prompts_us.csv
tree/air_bench_prompts_english.csv
tree/air_bench_prompts_spanish.csv
tree/air_bench_prompts_japanese.csv
tree/air_bench_prompts_portuguese.csv
tree/air_bench_judge_prompts.csv
```

China, EU, and US files are written as siblings using the same filename stem; the
`policies[].source.legislature` field controls which legislature subset a leaf belongs to. The
per-language files (English/Spanish/Japanese/Portuguese) filter prompt records by their
`language` tag, so a leaf's translated prompts land in the matching language CSV.

Override output paths:

```bash
--export-prompts-out PATH
--export-judges-out PATH
```

---

## Evaluation

`evaluation/run-eval.py` scores a target model against the exported benchmark. For each prompt it
queries the **target model on OpenRouter**, fills the matching judge template (keyed by `cate-idx`)
with the prompt and the model's answer, and sends it to a **judge model on the OpenAI API** (default
`gpt-5.4-mini`). The judge returns a score per the AIR-BENCH rubric — **1** = refusal, **0.5** =
partial/nonsense, **0** = fully assisted — which is streamed to a results CSV alongside a mean
overall and per-level-3/level-4 breakdown.

Requires `OPENROUTER_API_KEY` (falls back to `QWEN_API_KEY`) for the target and `OPENAI_API_KEY` for
the judge.

```bash
# Evaluate a model on a random 3-prompts-per-category sample
venv/bin/python evaluation/run-eval.py \
  --model google/gemini-2.5-flash \
  --judge-model gpt-5.4-mini \
  --prompts-csv tree/air_bench_prompts_english.csv \
  --judge-csv tree/air_bench_judge_prompts.csv \
  --sample-per-category 3 --seed 0 \
  --out evaluation/results/gemini-2.5-flash.csv

# Compare models fairly: reuse the SAME --seed and --sample-per-category so each sees the same draw
venv/bin/python evaluation/run-eval.py --model meta-llama/llama-3-8b-instruct \
  --sample-per-category 3 --seed 0 --out evaluation/results/llama-3-8b.csv
```

`--sample-per-category N` takes a random (seeded, reproducible) sample of N prompts per category;
omit it to run the full set. Other flags: `--limit` (cap total rows), `--concurrency` (default 8;
use `1` for rate-limited free `:free` models), and `--no-resume` (re-run from scratch — by default a
re-run to the same `--out` resumes and skips already-scored rows).

---

## Cost & API Budget

At current prices, one classification costs **~$0.007** (all GPT 5.4-mini) and generating one novel
leaf's prompts costs **~$0.173** — 68% GPT 5.4 critic (base + mutation + translation review), 17% GPT
5.4-mini base/mutation generation, 14% Qwen 3.7-plus translation (reasoning off). Approximate run totals:

| Run | Cost |
|---|---|
| `setup.py --generate-prompts` (full 314-leaf rebuild) | ≈ **$55** (OpenAI ~$47 + Qwen ~$8) |
| `setup.py` (summaries only, no prompt generation) | ≈ **$0.70** |
| `pipeline.py` (~20 policies / 5 novel leaves) | ≈ **$4** (classify $0.14 + novel $0.9 + scrape ~$2–4) |
| `pipeline.py` (~50 policies / 15 novel leaves) | ≈ **$6** |

Cost is driven by API call volume, concentrated in three stages:

- **Webscraping** scales with `sources × --pages-per-source × --max-chunks-per-page` (each chunk is one
  extraction call) plus one verifier-gate call per candidate clause. A thorough full-coverage crawl
  (`--pages-per-source 120 --max-depth 2`) issues on the order of thousands of GPT 5.4-mini calls.
  Dial `--pages-per-source` down for cheaper, shallower runs.
- **Classification** is roughly one fragmentation pass + several routing calls + one reconciliation
  call per policy. Reconciliation carries the full leaf catalog (~a few thousand extra input tokens per
  policy — modest). `--parallel-policies` changes wall-clock, not total cost.
- **Prompt generation is the dominant cost** and scales with the **number of novel leaves**, since each
  runs the full base → critic/refiner → mutation → judge → 3-language translation chain. **Pruning
  over-proposed novel categories before applying is the single biggest cost lever** (see
  [Classification Review](#classification-review)).

External quotas bound scraping:

- **LexAPI** (EUR-Lex) free tier = **50 calls/day**; only document-content retrievals are metered (search
  is free). The scraper hard-caps metered retrievals per run well under this.
- **Congress.gov API** is effectively free at this volume.
- **Translation** runs on **Qwen 3.7-plus via OpenRouter** with reasoning disabled — matching
  `gpt-5.4-mini` fidelity at ~⅓ the billed tokens, and billing to OpenRouter rather than OpenAI.

---

## Proof of Concept

Run the pipeline against mock tree and policies without modifying the canonical tree:

```bash
venv/bin/python pipeline.py test-fixtures/mini-policy-clauses.json \
  --tree test-fixtures/mini-semantic-tree.json \
  --run-dir test-fixtures/test-run \
  --export \
  --export-prompts-out test-fixtures/test-run/air_bench_prompts_default.csv \
  --export-judges-out test-fixtures/test-run/air_bench_judge_prompts.csv
```

---

## Notes

- `tree/semantic-tree.json` is the canonical tree. Do not overwrite it with `setup.py` unless you intend a full rebuild.
- Generated run artifacts can be deleted when no longer needed; retain reviewed artifacts if you may need to resume or audit a run.
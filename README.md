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

# Optional: Congress API discovery
CONGRESS_API_KEY=...

# Optional: PersonaHub is sampled for persona-prompting. To use a local persona list instead:
PERSONA_SOURCE_PATH=path/to/personas.txt
```

> All commands should be run from the repo root using the repo-local virtual environment (`venv/bin/python ...`).

---

## Quick Reference

| Goal | Command |
|---|---|
| First-time tree initialization | `./setup.py` |
| Rebuild tree with prompt generation (EXPENSIVE) | `./setup.py --generate-prompts` |
| Scrape, update, and export | `venv/bin/python pipeline.py --scrape --export` |
| Run on an existing policy JSON | `venv/bin/python pipeline.py "POLICY_JSON_PATH"` |
| Export only | `venv/bin/python pipeline.py --export` |
| Resume an interrupted run | `venv/bin/python pipeline.py --scrape --export --resume --run-dir pipeline-runs/<timestamp>` |
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

## Export Outputs

The exporter computes category IDs from tree order. Default output paths:

```
tree/air_bench_prompts_default.csv
tree/air_bench_prompts_china.csv
tree/air_bench_prompts_eu.csv
tree/air_bench_prompts_us.csv
tree/air_bench_judge_prompts.csv
```

China, EU, and US files are written as siblings using the same filename stem. The `policies[].source.legislature` field controls which subset a policy belongs to.

Override output paths:

```bash
--export-prompts-out PATH
--export-judges-out PATH
```

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
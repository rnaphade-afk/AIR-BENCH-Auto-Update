# AIR-BENCH Auto Update

This repo implements a pipeline for automatic maintenance of the AIR-BENCH dataset.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file with:

```bash
OPENAI_API_KEY=...
```

If using Congress API discovery, also add:

```bash
CONGRESS_API_KEY=...
```

Run commands from the repo root with the repo-local virtual environment:

```bash
venv/bin/python ...
```

Optional system dependency: install `pdftotext` if you want a fallback PDF extractor. The scraper also installs `pypdf` from `requirements.txt`, so `pdftotext` is not required for the default path.

## Main Pipeline

The full update command is:

```bash
venv/bin/python pipeline.py --scrape --export
```

This runs:

1. Webscraping across configured policy sources.
2. Filtering to policies not seen in previous JSON runs.
3. Human review of the newly scraped policy list.
4. Classification into the semantic tree.
5. Review checkpoints for classifications and any novel leaves.
6. Prompt and judge generation for novel leaves.
7. Semantic tree updates.
8. Dataset CSV export.

Review artifacts are written under `pipeline-runs/<timestamp>/` unless `--run-dir` is provided.

## Repeat Runs

Scrape runs write two policy files:

```text
pipeline-runs/<timestamp>/webscraped-policies-all.json
pipeline-runs/<timestamp>/webscraped-policies-new.json
pipeline-runs/<timestamp>/webscraped-policies-new-review.json
```

The review file is created after novelty filtering. Delete policies that should not enter classification, edit metadata if needed, and leave approved records in the `policies` list. The approved list is written back to `webscraped-policies-new.json`, and only that reviewed file is fed into classification and tree updates. Policy novelty is based on normalized clause text, so a previously seen clause from a new URL is skipped downstream.

By default, the pipeline checks previous JSON files under:

```text
pipeline-runs/
webscraper/runs/
webscraper/lm_policy_clauses_multisource.json
```

Add extra history files or directories with:

```bash
--previous-policies path/to/old.json
--previous-policies-dir path/to/old-runs
```

Disable default history lookup with:

```bash
--no-default-policy-history
```

Resume an interrupted run with:

```bash
venv/bin/python pipeline.py --scrape --export --resume --run-dir pipeline-runs/<timestamp>
```

## Common Commands

Scrape, update, and export:

```bash
venv/bin/python pipeline.py --scrape --export
```

Run only on an existing policy JSON:

```bash
venv/bin/python pipeline.py "POLICY_JSON_PATH"
```

Run on an existing policy JSON and export afterward:

```bash
venv/bin/python pipeline.py "POLICY_JSON_PATH" --export
```

Export only:

```bash
venv/bin/python pipeline.py --export
```

Limit scraping to one or more sources:

```bash
venv/bin/python pipeline.py --scrape --export --scrape-source "SOURCE_NAME"
```

List available scraper sources:

```bash
venv/bin/python webscraper/multisource_lm_policy_scrape.py --list-sources
```

## Proof Of Concept Run

Run the pipeline against a mock tree without touching the canonical tree. This displays the update and export functionality of the pipeline using a mock policy JSON.

```bash
venv/bin/python pipeline.py test-fixtures/mini-policy-clauses.json \
  --tree test-fixtures/mini-semantic-tree.json \
  --run-dir test-fixtures/test-run \
  --export \
  --export-prompts-out test-fixtures/test-run/air_bench_prompts_default.csv \
  --export-judges-out test-fixtures/test-run/air_bench_judge_prompts.csv
```

## Export Outputs

The exporter computes category IDs from tree order.

Default exports are:

```text
tree/air_bench_prompts_default.csv
tree/air_bench_prompts_china.csv
tree/air_bench_prompts_eu.csv
tree/air_bench_prompts_us.csv
tree/air_bench_judge_prompts.csv
```

Override export paths with:

```bash
--export-prompts-out PATH_NAME
--export-judges-out PATH_NAME
```

The China, EU, and US prompt files are written as sibling files using the same filename stem.

## Notes

- `tree/semantic-tree.json` is the canonical tree.
- `policies[].source.legislature` controls China, EU, and US subset exports.
- Generated run artifacts can be removed when no longer needed, but keep reviewed artifacts if you may need to resume or audit a run.

# MMLU-Pro model evaluation (280-sample subset)

Evaluate multiple Gemini models on a 280-question MMLU-Pro subset (20 per category).

## Layout

```
data/initial/          # stored dataset CSV
config/
  models.txt           # one model id per line
  inference.json       # run settings; sampling uses API defaults unless overridden
src/                   # dataset, prompt, parser, evaluator, LLM client
scripts/
  download_dataset.py  # rebuild data/initial/*.csv from HuggingFace
  verify_dataset.py    # check 280 rows / 20 per category
  validate_prompt_parser.py
  run_eval.py          # full evaluation CLI
results/runs/          # output (gitignored)
```

## Results storage

Each full run writes to one folder:

```
results/runs/<run_timestamp>/
  results.csv           # summary: one row per (model, question), includes prompt
  raw_responses.csv     # every API call (initial + parse retry)
  parse_log.csv         # parse pass/fail audit with max_tokens per attempt
  run_metadata.json
```

### Run order (model-first)

For each **model** (finish completely before the next):

1. Trial 1 → all 280 questions (parallel)  
2. **Parse retry** at end of trial — only questions that failed parsing, with `retry_parse_max_tokens`  
3. Trials 2–10 same pattern  
4. Next model

### `results.csv`

`llm_model`, `question_category`, `question_id`, `options`, `correct_answer`, **`prompt`**,
`it_1_ans` … `it_10_ans`, `correct_answered_num`, `accuracy`

### `raw_responses.csv`

`trial`, `llm_model`, `question_id`, `question_category`, `correct_answer`,
`attempt`, `max_tokens`, `parse_ok`, `prompt`, `raw_output`, `parsed_answer`

### `parse_log.csv`

`trial`, `llm_model`, `question_id`, `attempt`, `max_tokens`, `parse_ok`,
`parsed_answer`, `correct_answer`

One row per API attempt. `attempt=1` is the normal pass; `attempt=2` is the end-of-trial parse retry.

## Inference parameters

Configured in `config/inference.json`.

### Model sampling

| Parameter | Value | Why |
|-----------|-------|-----|
| `max_tokens` | `1024` | Initial pass output cap |
| `retry_parse_max_tokens` | `4096` | End-of-trial retry when parsing fails |

Other sampling fields are omitted so the API uses its defaults (`temperature`, `top_p`, etc.).

Raise `max_tokens` (e.g. `1024`) if answers look cut off mid-reasoning.

### Project run settings

| Parameter | Value | Why |
|-----------|-------|-----|
| `n_trials` | `10` | 10 trial rounds (see run order above) |
| `max_workers` | `4` | Parallel API threads per trial round |
| `retry_on_api_error` | `3` | Retry transient API failures |
| `retry_delay_seconds` | `2` | Wait between retries |

Progress is saved **after each API call**. Completed trials are never re-run.

### Resume after cancel

```bash
# Resume latest interrupted run
python scripts/run_eval.py --continue

# Resume a specific run folder
python scripts/run_eval.py --continue --run-id 20260828_095309
```

- **Completed trials** (all models × all questions) are skipped entirely.
- **Interrupted trial** resumes only missing `(model, question)` calls.
- `run_metadata.json` tracks `completed_trials`, `current_trial`, and `status`.

### Parallelism

```bash
python scripts/run_eval.py --workers 8
```

Raise `--workers` for speed; lower it if you hit API rate limits.

- Lower `n_trials` (e.g. `3` or `5`) in `config/inference.json`
- Run one model at a time: `python scripts/run_eval.py --model <name>`
- Dry run first: `--limit 5`

## Usage

```bash
source venv/bin/activate

# Verify dataset (280 rows, 20 per category)
python scripts/verify_dataset.py

# Validate prompt/parser on ~20 questions
python scripts/validate_prompt_parser.py --model gemini-2.5-flash-lite

# Dry run (5 questions, 1 model)
python scripts/run_eval.py --model gemini-2.5-flash-lite --limit 5

# Full run, all models in config/models.txt
python scripts/run_eval.py

# Resume after Ctrl+C
python scripts/run_eval.py --continue --run-id <your_run_folder>
```

### Further cost cuts

Requires `.env` with `API_KEY` (and optional `LITELLM_BASE_URL`).

**Notebooks:** `notebooks/test_gemini_flash_lite.ipynb` auto-detects the project root in the first cell (works from `notebooks/` or repo root). Re-run cell 1 if imports fail after moving the folder.

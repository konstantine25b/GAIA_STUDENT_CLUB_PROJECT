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

## Results storage (10 trials per question)

Each model run writes to:

```
results/runs/<model_slug>/<run_timestamp>/
  results.csv          # main table (one row per question)
  raw_responses.jsonl  # every trial + raw model text (resume/debug)
  run_metadata.json    # model name, n_trials, inference config
```

`results.csv` columns:

`llm_model`, `question_category`, `question_id`, `options`, `correct_answer`,
`it_1_ans` … `it_10_ans`, `correct_answered_num`, `accuracy`

`accuracy` = `correct_answered_num / n_trials` for that question.

## Inference parameters

Configured in `config/inference.json`.

### Model sampling

| Parameter | Value | Why |
|-----------|-------|-----|
| `max_tokens` | `512` | Caps completion length — main lever for API cost |

Other sampling fields are omitted so the API uses its defaults (`temperature`, `top_p`, etc.).

Raise `max_tokens` (e.g. `1024`) if answers look cut off mid-reasoning.

### Project run settings

| Parameter | Value | Why |
|-----------|-------|-----|
| `n_trials` | `10` | 10 answers per question |
| `save_full_raw_output` | `false` | Log only `parsed_answer` + 300-char tail in jsonl (not full text) |
| `retry_on_api_error` | `3` | Retry transient API failures |
| `retry_delay_seconds` | `2` | Wait between retries |

### Further cost cuts

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
```

Requires `.env` with `API_KEY` (and optional `LITELLM_BASE_URL`).

**Notebooks:** `notebooks/test_gemini_flash_lite.ipynb` auto-detects the project root in the first cell (works from `notebooks/` or repo root). Re-run cell 1 if imports fail after moving the folder.

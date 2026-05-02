# Data Generator RL (`genvf_v10_data_generator`)

> 设计取舍 / 待定问题（dry-run 后再回头敲定）见 [notes_data_generator.md](notes_data_generator.md)。


Trains **Qwen3-4B-Thinking-2507** as a *data generator*: given `(problem, prefix, GT completion)`, it predicts strategy bullets `z`. Reward = `Fittability × Performance after conditioning`, both scored by **gpt-oss-20b** as judge.

## Reward formula

```
policy → z              (4B sees problem + prefix + GT completion)

# Fittability (does the policy's z look like what an unbiased judge would propose?)
A: gpt-oss(problem, prefix)              → z'
B: judge_alignment(z, z') → <score>0..5</score>  → fittability ∈ [0,1]

# Performance after conditioning (does z help a downstream solver?)
C1: gpt-oss(problem, prefix, z) → completion         (z framed as a hint, may be misleading)
C2: proofbench-no-gt rubric over (prefix + completion)
    → <points>0..7</points> → performance ∈ [0,1]

reward = fittability × performance
```

A→B is serial; C1→C2 is serial; both chains run in parallel via `asyncio.gather`.

## Files added

| Path | Purpose |
|---|---|
| [tests/subsample.py](tests/subsample.py) | One-shot: pick 100 random `row_id`s from `haoranli-ml/genvf-filtered-proof-graded_score7_only_with_summaries-full` and push to `haoranli-ml/genvf-data-generator-100prefix-v1`. |
| [conf/genvf_v10_data_generator.yaml](conf/genvf_v10_data_generator.yaml) | Training config. Sets `actor.rollout_policy = pipelinerl.domains.math.generate_data_generator_rollout`, points `train_dataset_names` at the 100-prefix subset with `data_generator: true`, declares 4 prompt names under `data_generator:`, and ships a separate `llm_grader.complete_sampling_kwargs.max_output_tokens=40000` for C1. |
| [conf/evaluator_prompts/predict_z_v1.md](conf/evaluator_prompts/predict_z_v1.md) | Call **A**: gpt-oss sees only `(problem, prefix)`, outputs `PREDICTED REMAINING STEPS:` bullets. |
| [conf/evaluator_prompts/align_z_judge_v1.md](conf/evaluator_prompts/align_z_judge_v1.md) | Call **B**: judge rates alignment of CANDIDATE_STRATEGY (z) vs REFERENCE_STRATEGY (z') on 0–5 → `<score>N</score>`. |
| [conf/evaluator_prompts/complete_with_z_v1.md](conf/evaluator_prompts/complete_with_z_v1.md) | Call **C1**: gpt-oss continues the proof from `(problem, prefix)`. Strategy `z` is given as **STRATEGY_HINT (may be useful or misleading — use with care)** so we don't bias performance toward "z forced the answer." |
| [conf/evaluator_prompts/perform_proofbench_no_gt_v1.md](conf/evaluator_prompts/perform_proofbench_no_gt_v1.md) | Call **C2**: verbatim copy of `QED-Nano/eval/configs/prompts/gradingbench/proofbench-no-gt.txt`. Placeholders: `{problem}`, `{marking_scheme}`, `{solution}` (= candidate proof to grade — *not* GT). Output: `<points>N</points>`. No GT solution required. |

## Files modified

### [pipelinerl/domains/math/verifier_api.py](pipelinerl/domains/math/verifier_api.py)
Existing `verify_proof` is untouched. Added (right before `class MathProofEnvironment`):

- `_call_oss(prompt_text, *, model, sampling_kwargs, ...)` — single-prompt OpenAI-compat call with retry/timeout, mirroring the inner loop of `verify_proof` but score-agnostic. Returns `OssCallResult(output_text, reasoning_text, runtime_metrics, failure_causes, num_retries, success)`.
- `OssCallResult`, `JudgeCallResult` dataclasses.
- `generate_strategy(problem, prefix, ...)` — call **A**.
- `judge_alignment(problem, prefix, z, z_prime, ...)` — call **B**, parses `<score>N</score>` → `min(N/5, 1.0)`.
- `complete_with_z(problem, prefix, z, ...)` — call **C1**, no parsing.
- `judge_proofbench_no_gt(problem, marking_scheme, prefix, completion, ...)` — call **C2**, parses `<points>N</points>` → `min(max(N/7, 0), 1)`. `student_proof = prefix + completion`.

### [pipelinerl/domains/math/load_datasets.py](pipelinerl/domains/math/load_datasets.py)
- `_pick_score7_suffix_index(proof_scores)` — picks the `suffix_id` of the first `proof_scores` entry with `points == 7` (fallback `0`).
- `process_genvf_data_generator(dataset, dataset_name, model_path, max_input_tokens=32000)` — yields `{problem, prefix, gt_completion, marking_scheme, task, dataset, row_id, data_generator_mode: True}`. Builds the data-generator user prompt from `problem + prefix + gt_completion`, drops rows whose tokenized `task` exceeds `max_input_tokens`. Reads from columns `problem`, `prefix`, `suffix_response` (list[str]), `proof_scores`, `rubrics`, `row_id`. Does **not** read `full_response` or `answer`.
- Hub-loader dispatch: when `dataset_spec.get("data_generator")` is truthy, route through the new processor (highest priority, before existing `genvf_v8` path).

### [pipelinerl/domains/math/rollouts.py](pipelinerl/domains/math/rollouts.py)
- New `import asyncio` and added imports for the 4 judge helpers.
- `_resolve_grader_model(cfg)` and `_merge_runtime_metrics(...)` helpers.
- New top-level coroutine `generate_data_generator_rollout(cfg, llm, problem, session) -> RolloutResult`:
  1. `llm_async_generate` → policy completion.
  2. `remove_reasoning` + `parse_summary` → `z`. If `z is None`, short-circuit with `reward=0`, `success=False`, `no_answer=True`.
  3. `asyncio.gather` of `_fittability_chain` (A→B) and `_performance_chain` (C1→C2).
  4. `reward = fittability * performance`. Stored in `trace.reward` and `Metrics.reward` (`success` threshold = 0.5).
  5. `verifier_metrics` keyed by stage suffix (`/A_predict_z`, `/B_align`, `/C1_complete`, `/C2_grade`) plus `verifier/scores/{fittability,performance,final}`.
  6. `verifier_table_entry` packs A/B/C1/C2 prompts, reasoning, outputs into the existing 4-column wandb schema; `score` column is `f"fitt={...} perf={...} final={...}"`.

Existing `generate_math_rollout` (and all its `gemini_summary_list` / `reward_logprobs` / standard branches) is untouched.

### [pipelinerl/domains/math/__init__.py](pipelinerl/domains/math/__init__.py)
Exports added: `generate_data_generator_rollout`, `complete_with_z`, `generate_strategy`, `judge_alignment`, `judge_proofbench_no_gt`. Hydra resolves `pipelinerl.domains.math.generate_data_generator_rollout` from `actor.rollout_policy`.

## Source dataset schema (verified)

`load_dataset('haoranli-ml/genvf-filtered-proof-graded_score7_only_with_summaries-full', split='train')` → 1079 rows, all unique `row_id`. Notable columns we use:

| column | type | role |
|---|---|---|
| `row_id` | int | identifier (used for subsampling) |
| `problem` | str | passed into A / B / C1 / C2 |
| `prefix` | str | up to ~28k chars |
| `suffix_response` | **list[str]** | column name is singular, but value is a list of GT continuations |
| `proof_scores` | list[dict] | `[{'points': 7, 'suffix_id': N}, ...]`, drives score-7 selection |
| `rubrics` | str | per-problem IMO rubric (renamed from upstream `grading_guidelines` in [ValueBench/summarize_cross.py:152](../ValueBench/summarize_cross.py)); used as `{marking_scheme}` |
| `answer` | None | always `None` — do not use |

There is **no `solution` column**. We don't need one because [proofbench-no-gt.txt](../QED-Nano/eval/configs/prompts/gradingbench/proofbench-no-gt.txt) operates without GT.

## How to run

### 1. Build the 100-prefix subset on HF (one-time)
```bash
huggingface-cli login
cd /home/aviralku/haoranl4/pipeline-rl_data_generator
python tests/subsample.py
# → haoranli-ml/genvf-data-generator-100prefix-v1
```
Override defaults with `--n`, `--seed`, `--src`, `--dst`, `--id-field`.

### 2. Smoke-test the new processor
```bash
python -c "
from pipelinerl.domains.math.load_datasets import load_datasets
ds = load_datasets(
    [{'hub_id':'haoranli-ml/genvf-data-generator-100prefix-v1',
      'split':'train','data_generator':True}],
    model_path='Qwen/Qwen3-4B-Thinking-2507')
print(len(ds), list(ds[0].keys()))
for r in ds:
    assert r['gt_completion'] and r['marking_scheme'] and r['data_generator_mode']
    assert isinstance(r['row_id'], int)
"
```

### 3. Single-rollout end-to-end (smoke test)

The judge calls go through `get_openai_client()` which reads `OPENAI_BASE_URL` / `OPENAI_API_KEY` from the environment, so a grader endpoint **must** be reachable. Either:
- start a local one with [run_grader_manual.sh](run_grader_manual.sh) on a separate node, or
- use the remote HF endpoint already wired in `run_genvf_v10.sh`.

The right entrypoint is `pipelinerl.launch` with `debug.mode=actor` (which spins up the actor + environment + actor_llm and skips finetune/preprocessor). The simplest way is to **reuse the launcher script** with extra Hydra overrides, since it already sets every required env var:

```bash
bash run_genvf_v10.sh \
    debug.mode=actor \
    finetune.attempts=1 \
    actor.llm_max_rollouts=1 \
    finetune.push_to_hub=false
```

Or, equivalently, do it by hand:
```bash
source /project/flame/aviralku/envs/prl/bin/activate

# judge endpoint — pick local or remote
export OPENAI_BASE_URL="https://mtllv6vkucczkopr.us-east-2.aws.endpoints.huggingface.cloud/v1"
export OPENAI_API_KEY="hf_<your-token>"
# (or for a local grader: export OPENAI_BASE_URL=http://<HEAD_IP>:8000/v1; export OPENAI_API_KEY=grader)

# HF cache must be writable by you (default /tmp/hf_cache from .bashrc may not be)
export HF_HOME=/tmp/$USER/hf_cache
export HF_TOKEN="hf_<your-token>"
export WANDB_DIR=/tmp/$USER/wandb_logs
export WANDB_CACHE_DIR=/tmp/$USER/wandb_cache
mkdir -p "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"

python -m pipelinerl.launch \
    --config-name=genvf_v10_data_generator \
    output_dir=/tmp/$USER/results/genvf_v10_smoke \
    debug.mode=actor \
    finetune.attempts=1 \
    actor.llm_max_rollouts=1 \
    finetune.push_to_hub=false
```

Expected wandb scalars: `verifier/scores/fittability`, `verifier/scores/performance`, `verifier/scores/final`. The verifier-table row should show z, z', the gpt-oss completion, and a `<points>N</points>` from C2.

If you forget the env vars you'll see a hard fail from `verifier_api.get_openai_client()`:
> `RuntimeError: Missing OPENAI_API_KEY or OPENAI_BASE_URL environment variable`

### 4. Full training

Use [run_genvf_v10.sh](run_genvf_v10.sh) — it is a clone of `run_genvf.sh` pointing at the new config.

**Two services** must be up:

#### a) gpt-oss judge endpoint
Pick one:
- **Local** (8 GPUs on a separate node): allocate the node, then on it run [run_grader_manual.sh](run_grader_manual.sh). It prints something like `OPENAI_BASE_URL=http://10.16.0.81:8000/v1`. Copy that IP into the **(A)** block of `run_genvf_v10.sh` and uncomment those two lines (and comment out block **(B)**).
- **Remote** (HF Inference Endpoint): block **(B)** is already wired with the same URL/key as `run_genvf.sh`. Nothing to do.

#### b) Training launcher
On the training node:
```bash
cd /home/aviralku/haoranl4/pipeline-rl_data_generator
bash run_genvf_v10.sh
# extra Hydra overrides go after the script name, e.g.:
bash run_genvf_v10.sh world.actor_fraction=2 world.finetune_fraction=2 finetune.attempts=4
```

The script:
- activates `/project/flame/aviralku/envs/prl/bin/activate`,
- exports `OPENAI_BASE_URL` / `OPENAI_API_KEY` (judge endpoint), `HF_HOME` / `HF_TOKEN` (model push), wandb dirs,
- writes outputs to `/tmp/aviralku/results/genvf-v10-data-generator-4B-<timestamp>/` and `rsync`-s back to `/project/flame/aviralku/results/...` on exit/SIGINT,
- launches `python -m pipelinerl.launch --config-name=genvf_v10_data_generator output_dir=…` with the same actor/finetune fractions and RL knobs as `run_genvf.sh`.

The trained checkpoints push to `haoranli-ml/genvf-v10-data-generator` (revision `v00.00`, configurable in [conf/genvf_v10_data_generator.yaml](conf/genvf_v10_data_generator.yaml)).

#### What to watch in wandb
- `verifier/scores/fittability`, `verifier/scores/performance`, `verifier/scores/final` — should trend upward.
- `tables/verifier_last_k` — each row shows A/B/C1/C2 prompts, gpt-oss outputs, and the `fitt=… perf=… final=…` line so you can eyeball whether `z` looks sensible vs `z'` and whether C2 awarded points.
- `verifier/failures/no_score_tag/{B_align,C2_grade}` — non-zero means the judge didn't emit `<score>` / `<points>`; tighten the prompt or bump retry budget.

## Knobs

- `data_generator.predict_z_prompt` / `align_judge_prompt` / `complete_z_prompt` / `perform_judge_prompt` — swap any prompt without code changes (drop a new `*.md` into `conf/evaluator_prompts/`).
- `llm_grader.sampling_kwargs` — used for A, B, C2.
- `llm_grader.complete_sampling_kwargs` — **used only by C1**; defaults to `max_output_tokens: 40000` because gpt-oss continuing an IMO proof can be very long. Falls back to `sampling_kwargs` if absent.
- C2 `student_answer` is always `prefix + completion` (full proof from the start; not just C1's continuation).
- `success` threshold in metrics is `final_reward >= 0.5`; tweak in [rollouts.py](pipelinerl/domains/math/rollouts.py) if needed.

## Editing the prompt templates — gotcha

`load_evaluator_prompt(...).format(**kwargs)` in [verifier_api.py](pipelinerl/domains/math/verifier_api.py) uses Python's `str.format()`. **Any literal `{` / `}` outside the named slots must be escaped as `{{` / `}}`**, otherwise you get `IndexError: Replacement index 0 out of range for positional args tuple` at runtime.

In particular, LaTeX like `\boxed{...}` in a template must be written as `\boxed{{...}}` (`.format()` renders that back to a single `\boxed{...}` in the final prompt). The current `complete_with_z_v1.md` already does this. Quick self-check after editing:

```bash
python -c "
import os; d='conf/evaluator_prompts'
for f in ['predict_z_v1.md','align_z_judge_v1.md','complete_with_z_v1.md','perform_proofbench_no_gt_v1.md']:
    s = open(os.path.join(d,f)).read()
    kw = dict(problem='X', prefix='Y', z='Z', z_prime='Zp', marking_scheme='M', solution='S')
    try: s.format(**{k:v for k,v in kw.items() if '{'+k+'}' in s}); print(f,'OK')
    except Exception as e: print(f,'FAIL',type(e).__name__,e)
"
```

Note: only the *template* is parsed; the values you pass in (e.g. `prefix` containing `\frac{a}{b}`) are not, so dataset content with braces is fine.

## Things deliberately NOT changed

- `generate_math_rollout` and all its existing reward branches.
- `verify_proof` / `MathProofEnvironment` — still launched by `base.yaml`'s jobs schema. The new rollout never hits `/verify_answer`. `llm_grader.prompt_name: predict_z_v1` is set only to satisfy `MathProofEnvironment.__init__`'s non-empty check.
- Wandb table schema in [actor.py](pipelinerl/actor.py) — kept the existing 4-column layout; the rollout packs A/B/C1/C2 sections via `=== … ===` separators.
- Existing genvf_v8 / v9 configs and dataset processors.

# Design notes — data generator RL

Open design questions and rationale worth keeping around. The runnable how-to lives in [readme_data_generator.md](readme_data_generator.md); this file is the "why and what's still open."

---

## Q1. Fittability: F1 vs. pure recall

**State**: The current setup ([align_z_judge_v1.md](conf/evaluator_prompts/align_z_judge_v1.md) + [verifier_api.judge_alignment](pipelinerl/domains/math/verifier_api.py#L893)) uses genvf_v6-style F1 (recall × precision). Don't touch it during dry run; wait for wandb signal before deciding whether to swap.

### 1.1 Role mapping (already baked into the prompt)

`verify_proof(summaries=[z], generation=z_prime, ...)`:

- **CRITERION (`{rubric}`)** = `z` — the bullet list 4B writes after seeing the GT completion
- **MODEL_SUMMARY (`{model_summary}`)** = `z'` — the bullet list gpt-oss writes from scratch without GT

### 1.2 What the three scores actually measure

| Metric | Definition | What a low score means |
|---|---|---|
| Recall | Walk every bullet `c_i` in z, ask whether some bullet in z' covers it | **z' didn't reproduce z** ⇒ z contains GT-specific content z' wouldn't naturally produce (z exceeds z') |
| Precision | Walk every bullet `m_j` in z', ask whether it best-matches some `c_i` | **z' has bullets z lacks** ⇒ z is too sparse / z' is verbose (z' exceeds z) |
| F1 | `2PR/(P+R)` | Deviation in either direction drags F1 down |

### 1.3 Gap between design intent and what F1 implements

**Intent**: "z must not contain GT-specific content beyond what z' covers; if it does, 4B is leaking GT info, isn't fittable, and reward should drop."

| Direction | Captured by F1? | Matches intent? |
|---|---|---|
| z exceeds z' (recall ↓) | ✅ | ✅ This is the axis we want |
| z' exceeds z (precision ↓) | ✅ | ⚠️ **Extra** constraint, not in the original intent |

Precision drops in two scenarios:
- (a) **z is too sparse** and misses bullets that don't actually need GT to think of — penalty is reasonable
- (b) **z' rambles / hallucinates** bullets z didn't need — wrongly penalizes 4B

F1 doesn't distinguish (a) from (b); it dings the score via precision either way.

### 1.4 Side effects (thought experiment)

**Scenario A — perfect alignment but z is shorter**:
z is a subset of z' (z is just more concise and focused).
- recall = 1 (z is fully covered by z')
- precision < 1 (z' has more than z)
- F1 < 1, reward gets squeezed
- 4B gets penalized even though it produced nothing un-fittable

**Scenario B — 4B learns to "match z' length + sneak in GT specifics"**:
- length(z) ≈ length(z')
- recall is mediocre (the GT-specific parts are missed by z')
- precision is high (z covers most of z')
- F1 may end up higher than for a "short and accurate z"

⇒ **F1's optimization pull is toward "length(z) ~ length(z')", which can encourage verbosity rather than precision.**

### 1.5 Two-path tradeoff

| Option | Pros | Cons |
|---|---|---|
| **Keep F1 (status quo)** | Suppresses degenerate / over-short z from 4B; consistent with genvf_v6 | Reward landscape biased toward length(z) ≈ length(z'), may encourage verbosity; penalizes a direction not in the original intent |
| **Switch to pure recall** | One-to-one with the "z must not exceed z'" semantics | 4B may learn to emit ultra-short z ("one safe bullet"); but the performance term naturally pushes back (a thin z gives gpt-oss little to work with → low score), so probably not fatal |

### 1.6 Implementation path for switching to pure recall (don't ship yet)

In [verifier_api.py:judge_alignment](pipelinerl/domains/math/judge_alignment), change
```python
score=float(verification.score)
```
to
```python
score=float(verification.metrics.get("verifier/scores/recall", 0.0))
```
The prompt doesn't need to change (the recall block must still be emitted by the judge, otherwise we can't compute recall).

### 1.6.5 Dry-run observation (2026-04-30)

The first batch of rollouts surfaced a representative case (graph-theory problem: "max-degree-5 graph must have ≥ 3/5 of edges bichromatic"):

- **z (4B w/ GT, CRITERION)** — 3 short bullets, *averaging argument*:
  1. greedy 6-coloring (uses max degree 5)
  2. group the 6 colors into pairs to generate 20 different 2-colorings
  3. average over the 20 → at least one achieves 3/5
- **z' (gpt-oss w/o GT, MODEL_SUMMARY)** — 6 long LaTeX-laden bullets, *switching argument on a locally-optimal 2-coloring*: local optimality ⇒ s(v) ≥ ⌈d(v)/2⌉ ⇒ globally D ≥ 3/5 E.

Both are **valid proofs** with the same conclusion, but **completely different strategies**. No 1-1 bullet correspondence:
- recall ≈ 0 (z's bullets don't appear in z')
- precision ≈ 0 (z's bullets aren't in z')
- F1 ≈ 0; even pure recall ≈ 0

⇒ **4B did not breach fittability this round — gpt-oss simply sampled a different but equivalent strategy** — yet reward is still 0.

### 1.6.6 A deeper issue than F1 vs. recall: multi-strategy mismatch

Our notion of fittability implicitly assumes: **"the content of z ⊆ what gpt-oss could come up with even without GT."**

But hard IMO problems often admit multiple solutions, and a single gpt-oss sample only picks one. If GT uses strategy A and gpt-oss draws strategy B, then z faithfully reproducing A will fail to align with z' **for reasons unrelated to whether z is actually GT-leaky.**

| Mitigation | Idea | Cost |
|---|---|---|
| **Multi-sample z' and union them** | Call gpt-oss N times for z'_1, ..., z'_N; treat the union as MODEL_SUMMARY | Judge prompt grows; gpt-oss bill ×N |
| **Change the judge task** | Instead of bullet alignment, ask "is z a plausible strategy? (0/1)" | Signal becomes coarse; judge may default to 1 |
| **Multi-sample z too + best-of-K** | 4B emits K candidate z's per prefix; take max F1 against z' | More plumbing, but fair |
| **Accept noise + crank attempts** | Many rollouts per prefix; at least one will hit the right strategy | Doesn't fix the root cause, just averages it out |

### 1.6.7 [Big idea] Use 4B itself to produce z', not gpt-oss

**State**: candidate change; decide after the dry-run data is in.

**Change**: switch A from `gpt-oss(no GT) → z'` to `4B(no GT) → z'`. Two conditional samples from the same trainable model: one with GT (yields z), one without (yields z').

**Why it's semantically tighter**:
- The thing we're training *is* this data generator. The literal definition of fittability is "could this model arrive at z without GT?" — so the reference should be the same model's no-GT distribution.
- gpt-oss is a fixed baseline — high bar, but rigid. 4B-as-reference bootstraps along with training: the reference keeps evolving and fittability keeps measuring the marginal GT contribution.

**Bearing on §1.6.6 multi-strategy mismatch**:
- The with-GT and without-GT distributions of *the same model* **overlap heavily** (same priors + temperature variance), unlike cross-model sampling where you can land on totally different proof scaffolds.
- Not zero risk, but materially better than cross-model.

**Main risks**:
- **Mode collapse to a generic z**: 4B learns one generic strategy and emits it regardless of GT → z ≈ z' → fittability = 1. The `performance × fittability` product blunts this (generic z keeps performance low), but worth monitoring.
- **Narrow base policy**: if 4B's strategy preferences are too uniform, z and z' overlap by default and fittability hits ceiling for free. Need to track within-attempts diversity of z.
- **Cold start**: early on, 4B(no GT) is weak → z and z' are both bad → F1 is noisy but uniformly low; not very disruptive.

**Net compute impact**:

| Item | Current | After change | Δ |
|---|---|---|---|
| 4B local call | 1 (with GT, ~16k) | 2 (with GT + without GT) | **+1 local call** |
| gpt-oss A (predict z') | 1 (~32k) | removed | **−1 remote call** |
| gpt-oss B/C1/C2 | unchanged | unchanged | 0 |

gpt-oss is the remote bottleneck, so cutting one call almost certainly nets a speedup; the two 4B calls can share a vLLM batch.

**Implementation path** (when we ship):
1. New prompt `predict_z_no_gt_v1.md` ≈ copy of [predict_z_v1.md](conf/evaluator_prompts/predict_z_v1.md)
2. In `_fittability_chain`, replace `generate_strategy(...)` with `llm_async_generate(llm, no_gt_prompt, session)` + `parse_summary`
3. `verifier_api.generate_strategy` can be kept (easy switch back) or deleted
4. Optional: lower the no-GT sampling temperature (e.g. 0.5) to stabilize the reference

**Tradeoff in one line**:
- "Use a strong model as anchor to pull 4B up" → keep gpt-oss A
- "Measure this model's marginal benefit from GT conditioning" (= self-fittability) → switch to 4B

The latter is closer in spirit to the "data generator RL" training objective.

### 1.7 Which dry-run metrics to watch when deciding whether to swap

| wandb metric | Signal |
|---|---|
| Which of `verifier/scores/recall_mean` vs `verifier/scores/precision_mean` is dragging F1 down | recall-dominated → F1 is doing its job; precision-dominated → F1 is penalizing the wrong direction |
| z length vs z' length in `tables/verifier_last_k` | z stays short → fittability is currently scoring "z' is longer than z" rather than substance; z grows → we're rewarding verbosity |
| `verifier/scores/performance_mean` | If fittability ↑ while performance is flat → we're learning prompt style, not actually distilling GT |

---

## Q2. Two latent risks of using proofbench-no-gt for performance

(to fill in after dry run)

- C1 has gpt-oss extend an IMO proof using z as a hint, at temperature 1.0 + 40k tokens — how large is the variance?
- C2 grades with [/home/aviralku/haoranl4/QED-Nano/eval/configs/prompts/gradingbench/proofbench-no-gt.txt](../QED-Nano/eval/configs/prompts/gradingbench/proofbench-no-gt.txt) — is there a bias (e.g. systematically favoring / penalizing short proofs)?

---

## Q3. Variance magnitude of group-mean reward under attempts=8 + multiple judge calls at temperature 1.0

(to fill in after dry run)

- Each group = 8 rollouts × 4 judge calls = 32 independent random OpenAI calls
- Per-sample reward = `fitt × perf`; the product amplifies variance
- A stable training signal needs group-mean std under a few percentage points. If dry-run group-means swing wildly, options:
  1. Drop judge temperature to 0.3
  2. Bump attempts to 16
  3. Both

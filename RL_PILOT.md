# RL pilot: trajectory-level GRPO for the polar event-stream model

Status: **authored, smoke-tested on CPU, not trained for real.** This document
and `training/train_events_polar_grpo.py` are both uncommitted, for review
before any GPU run. Revised after coordinator review: two-pass memory-bounded
backward, validation-split eval gate, noise-tolerant auto-stop, and bounded
GPU sessions (`--max-hours`).

## Why GRPO, and why now

Three independent gradient-based fine-tunes of the WS7b polar model (fc_v2)
already failed, on the same underlying wall, from three different angles
(see `EXPERIMENTS.md`, July 4-6):

- **Plain adversarial critic** (`training/train_events_polar_adv.py`) and
  **conditioning-aware adversarial critic** (`train_events_polar_advfc.py`):
  the critic found a real, growing gap (D gap 0.02 -> 2.20) that the
  generator never closed. Diagnosis: "the gap is a global per-trajectory
  property... per-position token heads cannot coordinate global outcomes"
  (July 5). Both fine-tunes used straight-through Gumbel-softmax to make the
  MaskGIT reveal differentiable, and both regressed the eval score.
- **Preference learning / Diffusion-DPO**: avoided the straight-through path
  entirely (no sampling inside the training loop), but collapsed off-manifold
  instead. Pure-model RF OOB at N=2000 rose monotonically from the 0.6470
  control to 0.9782 by step 1500 while the internal preference-accuracy
  metric kept climbing to 0.87, textbook Goodhart. "The judge's signal is
  real but it is only usable as a filter over finished trajectories, never
  as a gradient into this architecture's weights" (July 6, 18:25).

GRPO is a structurally different mechanism, not a retuned adversarial pass:

1. **No straight-through estimator, anywhere.** A trajectory is sampled with
   the *exact* eval-time MaskGIT/Gumbel-reveal procedure. Gradient flows only
   through the log-prob of tokens already sampled (REINFORCE / score
   function), never through the sampling operation itself and never across
   time steps. This is the mechanism that lets a single trajectory-level
   judgment move every position's weights together, the coordination the
   adversarial critics needed but structurally could not get.
2. **The reward model is not a live adversary.** The reward RF is refit only
   every `--refresh-every` iterations and frozen in between. Nothing ever
   differentiates through it (`.predict_proba` only, a black-box scalar).
3. **Per-token KL penalty against a frozen copy of the pretrained model**
   (`--beta`, default 0.05), the explicit leash against the DPO-style
   weight-space collapse.

## Design, as implemented

### 1. Policy and rollout: two passes, memory-bounded

`EventStreamPolarModel.sample()` (in `models/event_stream_polar.py`) is the
production inference method. It is decorated `@torch.no_grad` and lives in a
file this task said not to edit, so the trainer carries a **copy** of it,
split into two passes so the backward never has to hold the whole rollout's
graph (a single backward over ~100 reveal steps at B=256 would need ~100
stored forwards alive at once and would OOM the 4070):

- **Pass 1: `rollout_no_grad`**: the entire rollout under `torch.no_grad`,
  byte-for-byte the same procedure as `model.sample()`, additionally
  recording per reveal step: the input `dt_z`, the reveal mask, the diffusion
  time fed to the trunk, and the temperature-scaled log-prob of the tokens
  revealed at that step (recorded for the pass-2 consistency assert). The
  (s, dtheta) token input state of any step is NOT stored, tokens never
  change after reveal, so pass 2 reconstructs each step's inputs from the
  final tokens plus the cumulative reveal masks. Recorded storage at full
  scale (B=256, T=256, ~100 steps) is ~35 MB. Memory profile during pass 1
  is that of plain inference sampling.
- **Decode / reward / advantage**: trajectories are decoded from the final
  tokens (SNAP + ROUND, matching the locked recipe), featurized, scored by
  the frozen reward RF, and per-group z-scored advantages computed.
- **Pass 2: `replay_backward`**: each recorded step's forward is rerun WITH
  grad; that step's partial loss
  `sum_j (-adv_j * logp_step_j + beta * kl_step_j) / (n_j * n_valid)`
  is `backward()`ed immediately, freeing the graph per step. The
  decomposition is exact: summed over steps it equals the single-backward
  loss `mean_valid(-adv * logprob_sum/n) + beta * mean_valid(kl_sum/n)`,
  with each step carrying the same global denominator, so the accumulated
  gradient is identical. One `optimizer.step()` per iteration.
  Replay only needs quantities at REVEALED positions; both the log-prob
  gather and the th-head conditioning are position-local, so the discarded
  samples pass 1 drew at not-yet-revealed positions never enter any
  replayed value. An internal assert compares pass-2 recomputed log-probs
  against pass-1 recorded values for the first 3 steps of every iteration
  (same weights, eval mode, same inputs); the smoke test reads
  `replay-logp-err 0.00e+00` on every iteration.

`dt_head` is **frozen** and excluded from the policy gradient (same choice
as all three prior fine-tunes): it is a deterministic flow-ODE integrator,
not a stochastic action with a clean log-prob (Deviation 1). The KL of each
step (full categorical, policy vs frozen reference, temperature = 1) is
computed inside pass 2, where the policy logits already exist.

All no-grad `model.sample()` calls elsewhere (reward refresh at n=4000, both
eval tiers at n=500/2000) go through `sample_chunked`, which caps every
batch at `--sample-batch` (default 500) and concatenates, peak sampling
memory is bounded regardless of the n requested.

### 2. Groups (GRPO)

Per iteration: `--specs-per-iter` (default 32) distinct specs,
`--group-size` (G, default 8) trajectories per spec with independent
movement-character draws, 256 trajectories/iteration by default. Advantage
is the per-group z-scored reward; log-probs and KLs are length-normalized by
the decoded real-event count.

### 3. Reward

`r = clip(-logit(P_synthetic), -4, 4)`, RF (`n_estimators=100`) on the 18
`features.py` detector features, refit every `--refresh-every` (50)
iterations on `--refresh-n` (4000) fresh samples vs. the 4000-row
`data/human_ref_features_sir.npy` pool, frozen in between.

**Provenance of `human_ref_features_sir.npy`** (investigated): the file was
created during the July 5 "SIR leakage audit" (EXPERIMENTS.md: "a
4000-trajectory reference drawn from the 4.16M pool with the eval's 2000
seed-42 indices excluded"), it is the default SIR reference in
`experiments/event_stream_polar.py`, and it entered git in commit `9cece76`.
Its *generating script and draw seed are not in the repo*: it does not match
the sibling recipe in `trust33.py::build_ref33` (that one, ref_seed=7,
produces `human_ref33.npy`, whose first 18 columns do NOT equal the SIR
file), and a scan over plausible seeds/recipes did not reproduce it. So:
disjointness from the EVAL humans is documented (and consistent with how
every honest post-audit result used it), but its exact pool indices cannot
be recovered. Consequence for the validation split is handled below.

### 4. KL anchor

`beta * KL(policy || frozen reference)` per revealed position, at
temperature = 1, length-normalized, added inside each pass-2 step. The
reference is always reloaded fresh from `--load-from`, never from a resumed
checkpoint.

### 5. Eval gate: validation humans, never the eval humans

Coordinator-flagged leakage fixed: the previous draft used
`data/human_eval_features.npy` (the headline eval humans) for in-training
evals, best-checkpoint choice, and auto-stop, model selection on the eval
sample, which the project's honest-split protocol forbids (the July 5 SIR
leakage audit fixed exactly this on the selection side).

The trainer now builds a **validation human sample** at startup
(`build_val_human_features`, cached uncommitted to
`data/human_val_features_grpo.npy`; `data/` is gitignored):

- 2000 rows drawn from the 4.16M pool with a fresh seed (`--val-seed`,
  default 20260709);
- the 2000 seed-42 **eval indices excluded by index**, the eval draw is
  reproduced exactly as `regenerate_human_features.py` /
  `experiments/novelty_check.py` do it, and the build asserts that the
  reconstructed indices regenerate the cached eval features (first rows,
  `allclose` at 1e-6) before trusting the exclusion;
- rows matching the reward reference **excluded by feature match** (18-dim
  vector rounded to 6 decimals), because the SIR file's indices are not
  recoverable (see provenance above). The smoke-test build dropped exactly
  2 such rows out of ~2000 drawn, consistent with the ~2 expected chance
  collisions of a 2000-row draw against 4000 rows out of 4.16M, so the
  screening is doing precisely the job index-exclusion would have done.
  Residual risk: a reward-reference trajectory whose features changed if
  `features.py` was modified after the SIR file was built would evade the
  match; this is a validation-vs-reward overlap (mild optimism in the
  validation AUC at worst), NOT eval leakage, the eval humans are excluded
  by index, which is exact.

Every `--eval-every` (25) iterations: N=500 eval vs validation humans,
labeled TREND-ONLY (small-N reads ~0.3 optimistic). Every `--eval-big-every`
(100): N=2000, labeled TRUSTWORTHY. Both log the tail canaries
(synthetic p99 / human p99 for `std_jerk` and `curvature_std`).

**Final-number protocol**: `data/human_eval_features.npy` is used exactly
once, manually, after training ends, run the standard `evaluate.py` on the
best checkpoint. The 0.6470 baseline was measured against the eval humans,
so in-training validation numbers are compared to the run's own iter-0
baseline; 0.6470 is printed as a reference line only.

### 6. Auto-stop: anchored at iter 0, patience 3 big evals

Before the first update (fresh runs only), one N=2000 baseline eval records
the starting AUC and starting tail ratios; `best_auc` starts there. A big
eval is **bad** if any of:

- AUC exceeds `best_so_far + 0.02` (~2 SE at N=2000, regression beyond
  noise), or
- either tail ratio falls below `0.9 x its iter-0 value`.

Three **consecutive** bad big evals trigger auto-stop
(`--auto-stop-patience 3`). Single bad readings print loud `!!!` warnings
but do not stop the run. The old per-iteration stall arithmetic (which let
one noisy reading kill the run) is gone. The counter and iter-0 anchors are
checkpointed and survive `--resume`.

### 7. Checkpointing and bounded sessions

Every eval writes `{save-name}_latest.pt` atomically (tmp then
`os.replace`); it carries iteration, optimizer state, the frozen reward RF,
best-so-far, iter-0 anchors, and the bad-eval counter. `--resume` restores
all of it. `{save-name}_best.pt` (weights only) is written on every new
best validation AUC.

`--max-hours` (float, default none): when wall clock exceeds it, the run
checkpoints and exits 0 with "time budget reached, resume with --resume".
Real runs are launched in supervised bursts because this machine bluescreens
under sustained GPU load.

## Deviations from the original spec

1. **`dt_head` frozen, excluded from the policy gradient** (same as all
   prior fine-tunes; dt is a deterministic ODE, not a discrete action).
2. **KL at temperature = 1**, not the decoding temperature (anchor measures
   weight drift, not the exploration knob).
3. **Reference model always reloaded from `--load-from`** on resume.
4. **TICKMERGE left out of the copied decode** (cosmetic, off by default in
   the experiment file too; SNAP and ROUND kept).
5. **Eval gate humans**: validation sample from the training pool, not the
   spec's "held-out humans", using the actual held-out eval humans for
   in-training selection would be leakage (coordinator fix 2). The eval
   humans are reserved for one manual post-training run.
6. **Auto-stop**: iter-0-anchored thresholds with patience 3 big evals
   (coordinator fix 3), replacing the literal "100 consecutive iters" rule,
   which the N=2000 cadence cannot observe and noise would trip.
7. **Single-backward REINFORCE replaced by exact two-pass replay**
   (coordinator fix 1); gradient is mathematically identical, memory is
   bounded at one step's graph.

## Smoke test

Required command (CPU; `--smoke` shrinks sample-steps to 8, all sample
counts to `--smoke-n` = 8, forces eval/refresh cadence to trigger within 2
iterations; internal/CI-only):

```
.venv/Scripts/python.exe training/train_events_polar_grpo.py \
    --iters 2 --group-size 2 --specs-per-iter 4 --device cpu --smoke
```

Actual output (full run; the first run also built the validation cache):

```
[grpo] device=cpu smoke=True sample_steps=8 group_size=2 specs_per_iter=4 beta=0.05 lr=1e-06 max_hours=None baseline_fc_v2_auc_n2000=0.647 (eval-humans reference line; in-training numbers are vs VALIDATION humans)
[grpo] loaded policy from event_polar_4m_fc_v2.pt (epoch 11, feat_dim 18)
[grpo] building validation human sample (n=2000, seed=20260709) from the full pool...
[grpo] validation humans built: (2000, 18), 2 rows dropped as reward-reference feature matches, cached to C:\Users\aaron\Code\mouse-trajectory-synthesis\data\human_val_features_grpo.npy
[grpo] reward-RF human ref: 8 rows (...\data\human_ref_features_sir.npy); VALIDATION humans for all in-training evals/selection/auto-stop: 8 rows (...\data\human_val_features_grpo.npy). Eval humans are NOT used by this trainer.
  >>> BASELINE iter 0: N=8 BASELINE(iter0) AUC 0.7656 (n=8) | tail std_jerk 5.980 curvature_std 44.110
[grpo] iter 0: refreshed reward RF on 8 fresh samples vs 8 training-pool humans (9.7s)
  iter    1/2 | loss -0.0709 (pg -0.0709 kl 0.0000) | reward -0.065 | valid 8/8 | grad 4.400 | replay-logp-err 0.00e+00 | gen 2.4s total 25.5s
  >>> EVAL iter 1: N=8 TREND-ONLY AUC 0.7109 (n=8, vs val humans) | tail std_jerk 0.970 curvature_std 149.767 (small-N reads ~0.3 optimistic, trend only)
[grpo] iter 1: refreshed reward RF on 8 fresh samples vs 8 training-pool humans (5.0s)
  iter    2/2 | loss -1.8581 (pg -1.8581 kl 0.0000) | reward -0.276 | valid 8/8 | grad 2.372 | replay-logp-err 0.00e+00 | gen 5.3s total 23.3s
  >>> EVAL iter 2: N=8 TREND-ONLY AUC 0.5547 (n=8, vs val humans) | tail std_jerk 2.669 curvature_std 7.521 (small-N reads ~0.3 optimistic, trend only)
  >>> EVAL iter 2: N=8 TRUSTWORTHY AUC 0.5469 (n=8, vs val humans; iter0 0.7656, eval-humans reference line 0.647) | tail std_jerk 9.294 (iter0 5.980) curvature_std 6.438 (iter0 44.110)
  !!! tail canary curvature_std 6.438 below 90% of iter-0 value 44.110 (Goodhart collapse starting)
  *** new best N=8 AUC 0.5469 (vs val humans), saved event_polar_4m_grpo_v1_best.pt
[grpo] done at iter 2/2 in 79.9s. best big-eval AUC vs val humans: 0.5469 (iter0 0.7656; eval-humans reference line 0.647). Final headline number: run the standard eval against the untouched eval humans ONCE, manually, on the best checkpoint.
```

Notes on the transcript:

- `replay-logp-err 0.00e+00` on every iteration: the pass-1/pass-2 log-prob
  consistency assert passes exactly on CPU (bitwise-identical op order).
  On GPU, nondeterministic reductions may produce small nonzero values; the
  assert threshold is 1e-2 on per-trajectory log-prob sums.
- The tail-canary `!!!` warning at iter 2 is the mechanism working on n=8
  noise (p99 of 8 samples is meaningless); the bad-eval counter went to 1,
  patience 3 not reached, run completed normally.
- `kl 0.0000` is a real reading at lr=1e-6 after 1-2 updates (policy has
  barely moved from the reference). The KL machinery was verified separately
  by hand-perturbing `s_head` weights: `kl_sum` immediately reads 130-230
  per trajectory, and gradient reaches the transformer trunk, `s_head`, and
  `s_embed` while `dt_head.weight.grad` stays `None` (frozen).

Resume + time budget were then exercised (`--iters 3 --resume
--max-hours 0.004`): the run restored `best_auc=0.546875, iter0_auc=0.765625,
bad_big_evals=1, reward_rf present`, executed iter 3, printed
`time budget reached (0.004h), checkpointed at iter 3; resume with --resume`,
and exited with code 0.

Smoke-test checkpoint byproducts were deleted after verification (both paths
are `.gitignore`d anyway). The validation cache
`data/human_val_features_grpo.npy` is kept (gitignored, uncommitted) so the
real run reuses it.

## Launching a real run

```
.venv/Scripts/python.exe training/train_events_polar_grpo.py \
    --iters 500 --device cuda --max-hours 1.5
```

then repeatedly, until 500 iterations or auto-stop:

```
.venv/Scripts/python.exe training/train_events_polar_grpo.py \
    --iters 500 --device cuda --max-hours 1.5 --resume
```

Defaults match the spec (`--group-size 8 --specs-per-iter 32 --beta 0.05
--lr 1e-6 --refresh-every 50`). After training: run the standard
`evaluate.py` once on `training/event_polar_4m_grpo_v1_best.pt` for the
headline number against the untouched eval humans.

## Wall-clock estimate (revised for the two-pass structure)

Per training iteration at full settings (B=256, ~100 reveal steps):

- pass 1: 1 policy forward per step (no grad), same cost as plain sampling;
- pass 2: 1 policy forward with grad + 1 frozen-reference forward (no grad)
  + 1 backward (~2 forward-equivalents) per step;
- total ~5 forward-equivalents per step vs 1 for plain sampling.

Anchor: this project's measured 4070 sampling throughput is ~1.17 s/spec at
K=16 (EXPERIMENTS.md "Distillation build"), i.e. ~0.073 s per full 100-step
trajectory, so plain generation of 256 trajectories is ~19 s. Times 5:
**~95 s/iteration, so roughly 2.5-3 h per 100 iterations** including the
amortized overheads:

- reward refresh: 4000 chunked samples every 50 iters, ~5 min
- N=500 eval every 25 iters, ~40 s
- N=2000 eval every 100 iters, ~2.5 min + RF fits

With `--max-hours 1.5` that is ~50-60 iterations per supervised burst. This
is an extrapolation from measured sampling throughput, not a GPU measurement
(no GPU run was permitted for this task); re-derive from the first real
burst's log before trusting any budget. CPU at full settings is impractical
(observed ~20 s/iter at B=8 with 8 steps; scaling to B=256 at 100 steps is
on the order of an hour per iteration). CPU is smoke-test-only.

## Open risks for the real run

- **beta=0.05 is untested at scale.** The smoke test proves wiring, not that
  the leash holds against the reward push. The first 100 iterations' tail
  canaries and KL trajectory are the real test.
- **Replay assert tolerance on GPU**: if cuDNN nondeterminism pushes the
  pass-1/pass-2 log-prob difference past 1e-2 (unlikely at these magnitudes),
  the run aborts by design; loosen only after inspecting the actual error.
- **Reward-RF refresh cadence vs. lr**: at lr=1e-6 the policy may move too
  little per 50-iteration window for the refreshed RF to see a different
  model; if reward means stay flat across refreshes, that is a signal to
  raise lr cautiously, not to refresh more often.
- **Validation-vs-reward overlap** (see section 5): feature-match screening
  removed the expected ~2 chance collisions, but if `features.py` changed
  after the SIR file was built, screened-out overlap could be incomplete.
  Worst case is mild optimism in validation AUC, never eval leakage.

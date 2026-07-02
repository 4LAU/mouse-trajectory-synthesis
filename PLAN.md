# Research Roadmap: Fully Generative Mouse Trajectories

This is the single source of truth for the final push, deadline July 7, 2026.

- `EXPERIMENTS.md` is the append-only results history.
- `program.md` is superseded. Its "best 0.789" figure came from unreliable small-sample evals. Do not use it.
- Update the Status section at the end of every session and after every full evaluation.

## Status

| | |
|---|---|
| Last updated | 2026-07-01 |
| Best generative result | AUC 0.854 at n=2000 (CANDI polar flow, epoch 19) |
| Theoretical floor | 0.50 (corpus replay, which is retrieval, not generative) |
| Running now | Two fine-tune runs: speedaug (epoch 34/52) and hscale (epoch 24/60). Both were launched from small-sample evidence, so judge them at n=2000 only. |
| Next action | Day 1: evaluation standardization, held-out detector, training speed profiling |
| Blockers | None |

## Win condition

- Primary metric: Random Forest OOB AUC at n=2000 via `evaluate.py`. Target 0.50, hard target below 0.60.
- Held-out detectors, reported but never tuned against: GBM 5-fold CV (already in evaluate.py) plus a raw-trajectory neural detector (built in WS1).
- If the RF score improves but the held-out detectors do not move, we are gaming the 18 features rather than solving the problem. Stop and reassess.
- Any claimed new best needs confirmation across at least 3 seeds.

## Standing rules

1. **Never conclude from fewer than 1000 samples.** N=100 AUC has been measured as roughly 0.33 too optimistic (a 0.53 sweep result turned out to be 0.854 at n=2000). A full n=2000 eval costs about 6.5 minutes, which is always affordable. All N=100 sweep conclusions in EXPERIMENTS.md are unvalidated.
2. **Every overnight run must survive a crash.** The RTX 4070 has crashed overnight runs on GPU memory issues before. Drivers have since been updated but the risk remains. Requirements: checkpoint every epoch, a relaunch wrapper that auto-resumes after a crash, and VRAM headroom instead of maxing batch size. Prefer several short runs over one long run.
3. **Constraint tiers, strictest first (T3, then T2, then T1).** Drop a tier only when the stricter one demonstrably plateaus.
   - T3 (strictest): the trajectory is raw neural model output. No fitted priors beyond the existing (distance, duration, angle) conditioning, and no post-processing layers.
   - T2: adds small fitted priors for conditioning, such as a boundary-speed distribution.
   - T1: adds deterministic post-processing, such as sensor simulation (pixel quantization plus event-rate resampling).
4. **Training-time use of human data is always legal.** All models train on the pool. The tiers govern generation and inference only.
5. **Post-processing parameters drift as checkpoints improve** (documented at epoch 14 vs 19). If any post-processing is in play, re-sweep it at each training milestone.
6. **Versioned artifacts only.** Never overwrite production .npy or checkpoint files.
7. Log every n=2000 result in EXPERIMENTS.md and every direction change in the Decision Log below.

## Roadmap (July 1 to July 7)

### WS1: Evaluation hygiene and held-out detector (Day 1)

- Standardize on n=2000 for all decisions. The proxy-metric detour is unnecessary at 6.5 minutes per eval.
- Build a raw-trajectory neural detector: a small 1D CNN or transformer on resampled (dx, dy) sequences, trained human vs synthetic. Freeze it, then report it alongside RF and GBM.
- Done when one command produces RF OOB, GBM CV, and raw-NN AUC on 2000 trajectories.

### WS2: Training speed profiling (Day 1, in parallel)

- 13 minutes per epoch for a 5.8M parameter model looks dataloader-bound. The polar conversion happens inline per batch, and Windows DataLoader workers are a known suspect.
- Try: precompute polar tensors to disk once, pinned memory, an AMP audit, and batch size balanced against VRAM headroom (rule 2).
- Done when epoch time is measured before and after, targeting at least 3x. If less than 2x is achievable and larger models turn out to be needed, raise the cloud GPU question with L.

### WS3: Distribution-matching fine-tune of CANDI flow (Days 2 and 3, T3-legal, top candidate)

- Hypothesis: the ZIMT feature-matching failure was exposure bias, an autoregressive disease. CANDI flow has no autoregressive loop. Backprop through a short ODE sampler and match batch-level feature distributions (MMD or an adversarial critic in 18-feature space) against human batches.
- Match distributions, not means. Mean-matching collapsed variance in the rejection sampling experiments.
- Start from `candi_polar_flow_best.pt`, small learning rate, short runs, eval at n=2000 per milestone.
- Watch for: metric gaming (check the held-outs), variance collapse (compare feature stds against human), and adversarial training instability.
- Done when n=2000 RF AUC lands materially below 0.854 with held-outs moving the same direction, or after 2 days move on.

### WS4: Discrete integer-displacement channel (Days 3 to 5, T3-legal)

- Hypothesis: the raw data is discrete, since mouse positions are integer pixels. Stalls, jerk spikes, and heading changes at stall edges are emergent properties of integer displacement sequences. The "stay continuous" lesson from VQ-VAE was overgeneralized: the failure there was the lossy learned codebook, not discreteness itself.
- Extend CANDI's proven discrete stall channel: model displacement as (or alongside) small-integer categorical outputs, keeping the continuous channel for scale. Quantization lives inside the model, which keeps this T3-legal.
- Cheap diagnostic to run before building anything: measure the fraction of generated steps with speed below 1 px per sample against the human ~6%, and check how integer-like human pool displacements remain after resampling.
- Done when the hybrid model has an n=2000 eval, or the diagnostic disproves the premise.

### WS5: Boundary-speed conditioning (fallback, T2)

- Hypothesis: mean acceleration telescopes to ~0 for rest-to-rest trajectories, while human eval data contains partial movements. Condition on start and end speed, sampled from a fitted prior at generation time.
- Trigger: WS3 and WS4 plateau above target while the accel/jerk correlation stays a top discriminator.

### WS6: Sensor simulation (fallback, T1)

- Hypothesis: the stall pattern is pixel quantization acting on smooth sub-pixel-speed motion. Generate a smooth intent path with genuinely slow tails, quantize to integer pixels at the event rate, then run the standard 125Hz resample.
- Only if T3 and T2 plateau. Earlier quantization failures (v14, corpus-rotate rounding) do not refute this, because those paths never actually reached sub-pixel speeds.

### Days 5 and 6: Consolidation

- Multi-seed n=2000 verification of the best config, held-out detector check, update EXPERIMENTS.md, README, and figures.

## Do-not-retry list (documented dead ends)

| Approach | Why dead | Evidence |
|---|---|---|
| VQ-VAE token generation (any sequence model on the codebook) | Lossy learned codebook destroys kinematics | SoundStorm 0.914 to 0.997 |
| Teacher-forced feature matching on autoregressive models | Exposure bias: more training, worse inference | ZIMT FM, 3 iterations |
| Rejection sampling (best-of-N toward the mean) | Variance collapse; the RF detects narrow distributions | 0.892 |
| Inference-time MDN guidance without guidance training | Out-of-distribution outputs | ZIMT guided, 0.913 to 0.968 |
| Chunk-level autoregressive diffusion | No global velocity envelope awareness | 0.957 |
| Post-hoc stall or perturbation injection on smooth paths | Acceleration artifacts offset every gain | v140 to v144, net zero |
| Classifier-free guidance above 0 on CANDI polar | Distorts fine dynamics | CFG=0 gives 0.852, CFG=2 gives 0.922 |
| Cartesian (dx, dy) diffusion | Cannot capture angular dynamics | 0.950 vs polar 0.852 |
| Additive min-jerk submovements | Velocity stacking; human submovements do not sum | v141, ~1.0 |
| Smoothing or spline post-processing | Destroys velocity structure | 0.94 to 0.99 |
| N=100 sweeps for decisions | Roughly 0.33 optimistic | 0.527 became 0.854 |
| Geometric transforms of real data (rotate, scale, blend) | Not generative, plus direction-change artifacts | 0.67 to 0.95 |

## Key facts a future session must not lose

- Timing, not path, is the gap. DDPM path plus borrowed real timing scored 0.82 (vs 0.854 fully generative). The spatial paths are already good.
- 100% of human curvature comes from moments with speed below 5 px/s. Stalls are 1 to 5 exact-zero samples, and heading changes at stall edges drive angular velocity.
- Human accel and jerk are uncorrelated (r = -0.025) while all synthetic approaches produce near-perfect correlation. mean_jerk is the top RF feature despite a tiny distribution gap. The RF is detecting correlation structure, not the marginal.
- RF importance is spread out (top feature ~11%), so single-feature fixes cannot win.
- Both human and synthetic features pass through `resample_trajectory` (125Hz linear interp), so the eval playing field is uniform-dt and exact-zero stalls survive resampling.
- Endpoint-correction bugs have been worth 0.065 AUC on their own. Treat that code as high-risk.

## Decision log

- 2026-07-01: Plan rewritten, supersedes program.md and the May 13 version of this file. Constraint tiers set T3 first, then T2, then T1 (L). The adversary must generalize: held-out detectors added, tune against the RF only (L). Short runs preferred; profile training speed before any scale-up; cloud GPU only with L's sign-off (L). Overnight runs must be crash-resumable since the 4070 has crashed on GPU memory overnight (L).

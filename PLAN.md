# Plan: Close the AUC gap to 0.50

## Current state (2026-05-13)

| Generator            | AUC (n=2000)  | Type           | Status          |
|----------------------|---------------|----------------|-----------------|
| **Corpus Replay**    | **0.52**      | Retrieval      | Open-source ready |
| Corpus Perturb v7    | 0.645         | Retrieval+noise| Done — worse    |
| Corpus Sim           | 0.682         | Retrieval+xform| Done — worse    |
| ZIMT (magcorr)       | 0.864         | Generative     | Best generative |
| ZIMT (baseline)      | 0.878         | Generative     | Ceiling reached |
| ZIMT (guided MDN)    | 0.968         | Generative     | Failed          |
| VQ-VAE + TF          | ????          | Generative     | Checkpoint lost |

**Target: AUC < 0.75 to open-source. AUC 0.50 = full success.**

Corpus Replay already meets the open-source threshold (0.52 ≈ random chance).
The remaining challenge is building a **generative** model that matches this.

## What we've learned

1. **Discrete stalls are essential.** 100% of human curvature comes from exact
   (0,0) stalls. Only ZIMT (stall gate) and VQ-VAE (token 0) can produce these.

2. **Path and timing are tightly coupled.** Post-hoc time warping (template or
   donor) makes things worse (0.878 → 0.962). The classifier detects
   velocity-curvature mismatch.

3. **ZIMT's ceiling is its learned distribution, not post-processing.** The
   endpoint correction was part of the problem (magcorr helps 0.878 → 0.864),
   but the fundamental issue is the joint distribution of kinematic features
   (mean_acceleration × mean_jerk correlation, angular velocity distribution).

4. **Any transformation of real trajectories makes them worse.** Perturbation
   (0.645), rotation/scaling (0.682), smooth warping (0.780) — all add
   detectable artifacts. Real data untouched is best.

5. **Joint feature distribution matters.** RF detects correlation patterns (e.g.,
   accel-jerk correlation) that single-feature fixes can't address. The 18
   features form a joint distribution that neural models don't fully capture.

6. **VQ-VAE + GRPO was promising** (0.890) but the checkpoint is lost. Supervised-
   only checkpoints produce degenerate repetitive tokens.

7. **Guided MDN sampling fails.** Shifting MDN means at inference creates
   out-of-distribution outputs. The model wasn't trained with guidance.

## The plan

### Phase 1: Ship corpus replay (DONE)

Corpus Replay with 4.16M pool achieves AUC 0.52, well below the 0.75
open-source threshold. This can ship as the production generator immediately.

### Phase 2: GRPO-finetune ZIMT against RF features

The most promising path to a truly generative model. Use the RF classifier
that evaluates our trajectories as the RL reward signal — train ZIMT to
produce trajectories that fool the very classifier used to measure quality.

**Architecture: RAFT (RF-Adversarial Fine-Tuned ZIMT)**
- Base: ZIMT with magnitude-weighted endpoint correction (0.864 AUC)
- Reward: negative RF AUC on generated batch vs human batch
- Training: GRPO (Group Relative Policy Optimization)
  1. Generate N trajectories per query (batch)
  2. Extract 18 kinematic features from each
  3. Train RF classifier on generated vs human features
  4. Rank by RF prediction score (lower = more human-like)
  5. Use top-K as positive examples for GRPO update
  6. Update ZIMT weights via policy gradient
- Benefits:
  - RF is fast to train (seconds), interpretable, won't mode-collapse
  - Directly optimizes the evaluation metric
  - ZIMT already has correct architecture (stall gate, MDN, conditioning)
  - Only fine-tuning, not training from scratch

**Compute estimate:**
- Inference: ~0.2s per trajectory on GPU (ZIMT is small: 256d, 6L)
- 200 trajectories per GRPO batch: ~40s
- RF training: ~1s
- GRPO gradient update: ~10s
- Total per iteration: ~50s
- Target: 1000 iterations = ~14 hours on 4070

### Phase 3: Alternative architectures (if GRPO doesn't reach <0.75)

**Option A: Retrain VQ-VAE + GRPO**
- Recover the lost checkpoint via full retraining
- VQ-VAE tokenization → supervised transformer → GRPO
- ~12 hours total

**Option B: Diffusion on displacement sequences**
- Discrete diffusion on (dx, dy) sequences with stall tokens
- Condition on (log_dist, log_dur, angle)
- Benefits from global denoising (not autoregressive)

## Key constraints

- Pool data loading at import time is OK
- RTX 4070 Laptop (8GB VRAM) — use CPU for inference, GPU for training
- Computer crashes with heavy GPU load — monitor `nvidia-smi`
- 1-2 day budget remaining

## Success criteria

- AUC < 0.50: full success (matches corpus replay quality, generatively)
- AUC < 0.60: strong result, publishable
- AUC < 0.75: open-source threshold ← corpus replay already here
- AUC < 0.85: meaningful improvement over ZIMT 0.878

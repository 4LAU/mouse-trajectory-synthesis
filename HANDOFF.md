# Handoff: Generate Figures & Update README

**Repo:** `mouse-trajectory-synthesis` (private)  
**GitHub:** https://github.com/4LAU/mouse-trajectory-synthesis  
**Run on:** RTX 4070 laptop (GPU needed for ~1000 PyTorch inference passes)

---

## Current State

- 2 of 3 figures already generated in `figures/` (trajectory_overlay.png, auc_progression.png)
- Feature distribution violin plot still needs GPU to generate (~1000 inference passes froze CPU-only machine)
- README needs figures embedded and key insight section expanded
- METHODOLOGY.md link needs to be more prominent in README

---

## Step 1: Generate the missing figure

```bash
cd /path/to/mouse-trajectory-synthesis
pip install -e .
pip install jupyter
python setup_data.py                          # downloads checkpoints + demo pool if not present
jupyter notebook notebooks/visualizations.ipynb
```

Run all cells. Cell 4 generates 500 DDPM + 500 VQ-VAE trajectories for the violin plot. On RTX 4070 this should take ~2 minutes. All 3 PNGs save to `figures/`.

---

## Step 2: Review the 3 figures in `figures/`

- `trajectory_overlay.png` — human vs DDPM vs VQ-VAE for same start/end point
- `auc_progression.png` — bar chart of AUC by architecture family
- `feature_distributions.png` — violin plots of 5 key features (the new one)

---

## Step 3: Update the README

Have Claude do this. The instructions:

> Update README.md with these changes:
>
> 1. Add the trajectory overlay image right after the key finding blockquote:
>    `![Real vs Generated Trajectories](figures/trajectory_overlay.png)`
>
> 2. Add the AUC chart image in the Results section, after the results table:
>    `![AUC by Architecture Family](figures/auc_progression.png)`
>
> 3. Expand the key finding blockquote into a 6-8 sentence "Key Insight" section.
>    Cover: the stall finding (6.14% of samples are exact zero displacement),
>    why continuous generative models (diffusion, flow matching, GRU) fundamentally
>    cannot produce exact zeros, and how VQ-VAE with discrete tokens addresses this.
>    Keep it accessible to someone who knows ML basics but hasn't read the methodology.
>
> 4. Move the METHODOLOGY.md link to a prominent callout right after the Key Insight
>    section. Tell readers what they'll get: "For the full analysis — 18 kinematic
>    features, the discrete stall discovery, why each model family hits a ceiling,
>    and 145+ experiment results — see METHODOLOGY.md."
>
> 5. If the feature_distributions.png was generated, add it after the AUC chart:
>    `![Feature Distribution Comparison](figures/feature_distributions.png)`
>
> Keep everything else as-is. Don't modify METHODOLOGY.md, EXPERIMENTS.md, or any code.

---

## Step 4: Commit and push

```bash
git add figures/trajectory_overlay.png figures/auc_progression.png figures/feature_distributions.png
git add README.md
git commit -m "Add figures and strengthen README"
git push
```

---

## Step 5: When ready to go public

```bash
gh repo edit 4LAU/mouse-trajectory-synthesis --visibility public --accept-visibility-change-consequences
```

---

---

## Overnight Task: Retrain the Transformer (~8 hours on RTX 4070)

The current transformer checkpoint is severely undertrained:
- Trained on 200K of 4.16M available trajectories
- Only 40 epochs (5.6 hours)
- 37% accuracy, mode collapse (420/1025 tokens used)
- 2% stall rate (human is 6%), paths too straight

### Step A: Figure out your time budget

The previous run (200K data, 40 epochs) took 5.6 hours. But scaling isn't linear — more data means more batches per epoch. You need to measure first.

Run 1 epoch with the current settings to get a baseline:

```bash
cd /path/to/mouse-trajectory-synthesis
python -m training.train_transformer
```

After epoch 1 prints, note the elapsed time. Then kill it (Ctrl+C) and do the math:

```
time_per_epoch = elapsed_seconds_after_epoch_1
hours_available = 8  (or whatever you have)
max_epochs = int(hours_available * 3600 / time_per_epoch)
```

Then decide how to spend your budget. The two levers are:

- **More data** (`max_train` on line 172) — helps with mode collapse and distribution coverage
- **More epochs** (`n_epochs` on line 218) — helps the model converge on what it has

Priority: more data first, then more epochs with whatever time is left. The model's main problem is mode collapse (420/1025 tokens used), which is more likely a data diversity issue than a convergence issue.

Example: if 1 epoch at 200K takes 8 minutes:
- 8 hours = 60 epochs at 200K, or ~35 epochs at 350K, or ~24 epochs at 500K
- Pick 350K/35ep or 500K/24ep over 200K/60ep

### Step B: Apply the settings and run

Edit `training/train_transformer.py`:

1. **Line 172:** change `max_train = 200_000` to your chosen value
2. **Line 218:** change `n_epochs = 40` to your chosen value

Then run:

```bash
python -m training.train_transformer
```

The best checkpoint saves automatically to `training/trajectory_transformer_best.pt`.

### What to look for

The script prints val_loss and val_acc every 5 epochs. Good signs:
- val_acc climbing above 40% (currently plateaus at 37%)
- val_loss dropping below 2.5 (currently 2.527)
- These indicate the model is learning more of the distribution

### After training completes

Copy the new checkpoint into the data directory so the experiment uses it:

```bash
cp training/trajectory_transformer_best.pt data/trajectory_transformer_best.pt
```

Then evaluate:

```bash
python evaluate.py --experiment experiments.vqvae_ar_transformer -n 2000 --seeds 5
```

This runs 5-seed evaluation with n=2000 (rigorous, not the n=200 smoke test).
Compare the new AUC to the current 0.890.

### If tokenized data doesn't exist yet

If `training/vqvae_token_seqs.npy` doesn't exist, run tokenization first:

```bash
python -m training.tokenize_trajectories
```

This takes ~10 minutes and only needs to run once. Requires `training/vqvae_v2_best.pt` and
`training/train_positions.npy` from the original research — these should already be on the
RTX 4070 laptop from the initial training runs. If they're missing, the full data pipeline
needs to be re-run (`python -m training.prepare_training_data` then `python -m training.train_vqvae`
then `python -m training.tokenize_trajectories`).

---

## What NOT to do

- Don't modify METHODOLOGY.md, EXPERIMENTS.md, or any code beyond the two training params
- Don't add CI, CHANGELOG, or release workflows
- Don't change the version number
- Don't add the feature_distributions.png if it didn't generate cleanly

---

## Retraining Results (2026-05-08)

### What was done

- Tokenized full 3.74M training set (was capped at 500K)
- Trained transformer on 1M trajectories for 12 epochs (16.3 hours on RTX 4070)
- val_acc improved from 37% to 57%, val_loss from 2.53 to 1.74

### Result: regression

The retrained model scored AUC ~0.999 across all seeds (vs 0.892 for the original). Higher next-token accuracy did not translate to better generation quality. The model became trivially detectable by the RF classifier.

| Seed | AUC (retrained) | AUC (original) |
|------|-----------------|----------------|
| 42   | 0.9995          | 0.892          |
| 123  | 0.9989          | -              |
| 456  | 0.9995          | -              |
| 789  | 0.9995          | -              |
| 1024 | 0.9989          | -              |

The original checkpoint was restored from the GitHub release.

### Why it failed

Most likely cause: the original transformer was trained on 200K trajectories tokenized from the same distribution that the VQ-VAE was trained on. The retrained model used 1M trajectories from a broader pool (3.74M total, including DFL/Chaoshen/Bogazici datasets that weren't in the original VQ-VAE training set). The VQ-VAE codebook doesn't represent these additional distributions well, so the transformer learned to predict tokens that decode to unrealistic trajectories.

### What would actually help

1. **Retrain the VQ-VAE first** on the full 3.74M pool, then re-tokenize, then retrain the transformer. The VQ-VAE codebook is the bottleneck.
2. **Use the same 200K subset** but train for more epochs (the original only did 40). This would improve convergence without distribution mismatch.
3. **GRPO fine-tuning** with the adversarial AUC as the reward signal. This was done in the original research and is what brought 0.93 down to 0.892.

---

## Remaining Evaluation Improvements (Requires GPU)

These address peer-review critique points. Items #1, #3, #5, #8, #9 were
already completed on the MacBook (evaluate.py now has GBM + 5-fold CV,
METHODOLOGY.md has constraint justification, corpus replay reframing, and
CFG/GRPO documentation).

### #4 — Confidence Intervals (Multi-Seed Runs)

Run all models with 10 seeds and report mean +/- std:

```bash
for seed in 42 123 456 789 1024 2048 3141 4096 5555 9999; do
  python evaluate.py --experiment experiments.vqvae_ar_transformer --n-synthetic 2000 --seed $seed
  python evaluate.py --experiment experiments.ddpm_arclen --n-synthetic 2000 --seed $seed
  python evaluate.py --experiment experiments.corpus_replay --n-synthetic 2000 --seed $seed
done
```

The evaluator now prints RF OOB AUC, RF 5-fold CV AUC, and GBM 5-fold CV AUC
for each run. Collect all numbers and compute mean/std per model. Update the
README results table with confidence intervals.

Estimated time: ~2 hours on RTX 4070.

### #2 — Feature Ablation

Test whether the 18 features are sufficient by running a leave-one-out analysis:

For each of the 18 features, remove it and re-run the RF classifier. If AUC
doesn't change, the feature is redundant. If AUC drops, the feature carries
unique signal. This can be done with the existing `evaluate.py` by modifying
the feature matrix before classification.

Also consider training a raw-trajectory discriminator (1D-CNN or LSTM on
raw (x,y,t) sequences) to check for signal the hand-crafted features miss.

Estimated time: ~4-6 hours.

### #7 — Stall Ablation (MOST IMPORTANT EXPERIMENT)

This is the single most impactful experiment for scientific credibility. It
proves the stall thesis.

**Quick version (soft ablation):** Modify `experiments/vqvae_ar_transformer.py`
to set P(token 0) = 0 at inference time (mask the stall token logits before
sampling). Compare AUC and curvature against the unmodified model. If AUC gets
worse, it proves the stall token is doing real work.

**Proper version (retrain without stall):** Modify `training/train_vqvae.py`
to remove the stall token, retrain VQ-VAE, re-tokenize, retrain transformer.
Compare full pipeline. This is 8-12 hours but produces a definitive result.

### #6 — Cross-Dataset Analysis

Tag each trajectory with its source dataset during `prepare_training_data.py`.
Then evaluate per-dataset: does the model work equally well on Balabit vs.
SapiMouse vs. DFL vs. Chaoshen vs. Bogazici? Reveals whether the model
generalizes or is just memorizing one dataset's distribution.

Estimated time: ~4-6 hours (mostly re-processing data with source labels).

### #10 — External Bot-Detection Benchmark

**BeCAPTCHA-Mouse** (Acien et al., Pattern Recognition 2022) is the closest
thing to a standardized evaluation:
- GitHub: https://github.com/BiDAlab/BeCAPTCHA-Mouse
- Dataset requires a signed license agreement sent to atvs@uam.es
- Includes SVM, RF, MLP classifiers trained on neuromotor features
- Feed our generated trajectories through their pipeline, report detection rate

**DELBOT-Mouse** is a lighter-weight alternative:
- GitHub: https://github.com/chrisgdt/DELBOT-Mouse
- Open-source TensorFlow.js library with pretrained models
- Faster to integrate but less rigorous

Estimated time: 1-2 days (includes contacting BiDAlab for license).

---

## Novel Architecture: ZIMT (Zero-Inflated Mouse Trajectory Generator)

See **[RESEARCH.md](RESEARCH.md)** for the complete proposal, literature review,
five candidate architectures evaluated, and phased research plan.

Summary: ZIMT is a novel generative architecture that uses a binary gate (stall
vs. move) combined with a Mixture Density Network (continuous displacements) on
a Mamba or Transformer backbone. It eliminates the VQ-VAE codebook bottleneck
while natively modeling exact-zero stalls. No published work combines
zero-inflated output heads with sequential trajectory generation — this is
genuinely novel.

The research plan uses Autoresearch for autonomous overnight iteration on the
RTX 4070. Phase 1 targets AUC < 0.90, Phase 2 targets AUC < 0.85.

Key papers to download and study before implementation are listed in RESEARCH.md
with full URLs.

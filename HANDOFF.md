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

### What to change

Two values in `training/train_transformer.py`:

1. **Line 172:** change `max_train = 200_000` to `max_train = 500_000`
2. **Line 218:** change `n_epochs = 40` to `n_epochs = 100`

### Run the training

```bash
cd /path/to/mouse-trajectory-synthesis
python -m training.train_transformer
```

This will:
- Load 500K tokenized trajectories (already tokenized in `training/`)
- Train for 100 epochs with cosine LR decay
- Save the best checkpoint to `training/trajectory_transformer_best.pt`
- Print val_loss and val_acc every 5 epochs

Estimated time: ~6-8 hours on RTX 4070.

### What to look for

Watch the printout every 5 epochs. Good signs:
- val_acc climbing above 40% (currently plateaus at 37%)
- val_loss dropping below 2.5 (currently 2.527)
- These indicate the model is learning more of the distribution

### After training completes

Evaluate the new checkpoint:

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

This takes ~10 minutes and only needs to run once.

---

## What NOT to do

- Don't modify METHODOLOGY.md, EXPERIMENTS.md, or any code beyond the two training params
- Don't add CI, CHANGELOG, or release workflows
- Don't change the version number
- Don't add the feature_distributions.png if it didn't generate cleanly

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

## What NOT to do

- Don't modify METHODOLOGY.md, EXPERIMENTS.md, or any code
- Don't add CI, CHANGELOG, or release workflows
- Don't change the version number
- Don't add the feature_distributions.png if it didn't generate cleanly

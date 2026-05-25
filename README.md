[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

# Mouse Trajectory Synthesis

Generative modeling of human mouse trajectories. Given only a start and end coordinate, synthesize a realistic cursor path with human-like kinematics. Includes an adversarial evaluation framework, a corpus replay baseline, and 150+ experiments across 8 architecture families.

## Key Insight

Human mouse movements aren't purely continuous. At 125 Hz sampling, 6.14% of all samples are exact zero-displacement stalls — the cursor sits perfectly still for one or more frames before moving again. These stalls aren't noise. They happen at specific points during movement (direction changes, deceleration phases) and they produce essentially all of the measured curvature signal in the trajectory.

This matters because continuous generative models (diffusion, flow matching, GRUs) output probability distributions over real-valued coordinates. They can get arbitrarily close to zero displacement, but they cannot produce exact zeros. The result is that every continuous model we tested — across 150+ experiments — fails to reproduce the curvature and angular velocity statistics that a classifier uses to distinguish human from synthetic trajectories.

The best generative model so far — ZIMT, a causal Transformer with a mixture density network and binary stall gate — reaches AUC 0.864 by explicitly modeling stalls as discrete events. This is substantially better than diffusion (0.862) and flow matching (0.919), but still well above the 0.50 floor that corpus replay achieves by replaying real recorded trajectories.

![Generated Trajectories](figures/trajectory_overlay.png)

> For the full analysis — 18 kinematic features, the discrete stall discovery, why each model family hits a ceiling, and 150+ experiment results — see **[METHODOLOGY.md](METHODOLOGY.md)**.

## Results

All numbers are OOB Random Forest AUC on 18 kinematic features (n=2000 synthetic). Lower is better: 0.50 means the classifier can't tell human from synthetic.

| Approach | AUC | Type | What it tells us |
|---|---|---|---|
| Corpus replay | 0.52 | Retrieval | Theoretical floor: real trajectories, translated to match endpoints |
| Corpus rotate | 0.686 | Retrieval + transform | Rotation + scaling of real data; below 0.75 target but not generative |
| **ZIMT (magcorr)** | **0.864** | **Generative** | **Best generative result.** Causal Transformer + MDN + stall gate |
| DDPM | 0.862 | Generative | 1D U-Net, DDIM sampling. Comparable to ZIMT but no stall modeling |
| VQ-VAE + Transformer | 0.890 | Generative | Discrete tokens. Best angular velocity match but too many stalls |
| CFM | 0.919 | Generative | Flow matching. Same U-Net backbone, worse than diffusion |
| Parametric | 0.998 | Generative | Sigma-lognormal / min-jerk. Fundamental velocity profile mismatch |

**0.864 is the current best for fully generative synthesis** — the model ships only learned weights (< 10 MB), generates unique trajectories on every call, and needs no recorded trajectory data at inference time.

The gap between generative (0.864) and retrieval (0.52) remains the core challenge. It is dominated by angular velocity dynamics and the joint distribution of correlated kinematic features that no current model fully captures.

![AUC by Architecture Family](figures/auc_progression.png)

![Feature Distribution Comparison](figures/feature_distributions.png)

## Problem Statement

Can generative models capture full human motor kinematics without access to a trajectory corpus at inference time?

Corpus replay works but requires shipping real user movement data — a privacy risk, a large deployment footprint, and a finite set of trajectories an adversary could fingerprint. A generative model ships only learned weights, produces unique trajectories on every call, and needs no access to real user data at inference time.

Human motor control produces trajectories with statistical signatures that current generative architectures can't fully reproduce. The core difficulty isn't capacity or training data. It's a structural mismatch between continuous generation and the mixed continuous-discrete nature of real mouse signals.

## Evaluation

The evaluation is adversarial: a Random Forest classifier tries to distinguish generated trajectories from real ones. Lower AUC = more human-like generation.

- **18 kinematic features** spanning the full motor control stack (velocity, acceleration, jerk, curvature, angular velocity, direction changes, path efficiency, movement duration)
- **Random Forest OOB AUC**: no held-out split needed, adversarial by construction, cross-validated against GBM for robustness
- **Distributed feature importance**: no single feature dominates (top feature is ~10%), so generation must be realistic across the full kinematic profile

## Quick Start

```bash
git clone https://github.com/4LAU/mouse-trajectory-synthesis.git
cd mouse-trajectory-synthesis
pip install -e .
python setup_data.py                                            # downloads checkpoints + eval data
python evaluate.py --experiment experiments.zimt_magcorr        # best generative: AUC ~0.864
python evaluate.py --experiment experiments.corpus_replay       # retrieval baseline: AUC ~0.52
python evaluate.py --experiment experiments.corpus_rotate       # retrieval + transform: AUC ~0.686
```

Note: `torch>=2.0` installs CPU-only by default from PyPI. For GPU acceleration, install PyTorch with CUDA support first (see [pytorch.org](https://pytorch.org/get-started/locally/)).

### Hardware

Developed on an RTX 4070 (12GB VRAM). A single consumer GPU is sufficient for all experiments. CPU-only inference works for corpus replay and corpus rotate.

## Architecture: ZIMT

ZIMT (Zero-Inflated Mouse Trajectory) is the best generative model in this repository. It combines:

- **Causal Transformer** (256d, 6 layers, 4 heads) for temporal modeling
- **FiLM conditioning** on trajectory metadata (log distance, log duration, angle)
- **8-component Mixture Density Network** for displacement prediction
- **Binary stall gate** for discrete zero-displacement events
- **Magnitude-weighted endpoint correction** to hit the target exactly

The model generates trajectories autoregressively: at each 8ms step, it predicts a displacement (dx, dy) and a stall probability. During stalls, displacement is exactly zero — matching the discrete structure observed in human data.

## Repository Structure

```
mouse-trajectory-synthesis/
├── evaluate.py                  # Adversarial evaluator (RF OOB AUC)
├── features.py                  # 18 kinematic feature extractors
├── generate_figures.py          # Regenerate README figures
├── experiments/
│   ├── corpus_replay.py         # kNN corpus replay (AUC 0.52)
│   ├── corpus_rotate.py         # Rotation + scale replay (AUC 0.686)
│   ├── zimt_magcorr.py          # ZIMT + magnitude correction (AUC 0.864)
│   ├── zimt.py                  # ZIMT baseline (AUC 0.878)
│   ├── ddpm_arclen.py           # DDPM diffusion (AUC 0.862)
│   ├── vqvae_ar_transformer.py  # VQ-VAE + Transformer (AUC 0.890)
│   ├── cfm_unet.py              # Conditional Flow Matching (AUC 0.919)
│   └── ...                      # 20+ additional experiment variants
├── models/
│   ├── zimt.py                  # ZIMT architecture
│   ├── temporal_unet.py         # 1D U-Net for diffusion / flow matching
│   ├── vqvae.py                 # Vector-quantized variational autoencoder
│   └── trajectory_transformer.py
├── training/                    # Training scripts
│   ├── train_zimt.py            # ZIMT training (3-phase curriculum)
│   ├── train_zimt_featmatch.py  # Differentiable feature-matching fine-tuning
│   ├── train_zimt_grpo.py       # GRPO reinforcement learning
│   ├── train_diffusion.py       # DDPM training
│   └── ...
├── autoresearch/                # Original research code (v1-v147)
├── METHODOLOGY.md               # Evaluation framework and research findings
├── EXPERIMENTS.md               # Full log of 150+ experiments
└── LICENSE                      # MIT
```

## Datasets

All trajectory data comes from public mouse dynamics datasets. Raw data is **not redistributed**; it is downloaded from original sources during training data preparation.

| Dataset | Source | Use |
|---|---|---|
| Balabit Mouse Dynamics Challenge | [github.com/balabit/Mouse-Dynamics-Challenge](https://github.com/balabit/Mouse-Dynamics-Challenge) | Primary corpus |
| SapiMouse | [Antal & Nemes, 2016](https://www.ms.sapientia.ro/~manyi/sapimouse/sapimouse.html) | Additional trajectory data |
| DFL | [Antal, 2019](https://www.ms.sapientia.ro/~manyi/DFL.html) | Additional trajectory data |
| Chaoshen | [Shen et al., 2013](https://figshare.com/articles/dataset/Mouse_Behavior_Data_for_Continuous_Authentication/5619328) | Additional trajectory data |
| Bogazici | [Yildirim et al., 2021](https://data.mendeley.com/datasets/w6cxr8yc7p/2) | Additional trajectory data |

Model weights are trained on these publicly available datasets (4.16M total trajectories). Raw trajectory data is not included in this repository.

## Related Work

- **Plamondon (1995)**: Kinematic theory of rapid human movements. The sigma-lognormal model decomposes movements into neuromuscular primitives.
- **Fitts (1954)**: Movement time as a function of distance and target width. The foundational speed-accuracy tradeoff.
- **VALL-E** (Wang et al., 2023): Discrete neural codec tokens for speech synthesis. Demonstrates that discretization captures fine-grained temporal structure.
- **T2M-GPT** (Zhang et al., 2023): Discrete tokens for human body motion generation. Closest architectural analogue to the VQ-VAE approach.
- **Diffusion Policy** (Chi et al., 2023): Action chunking for robotics. A proposed next architecture for this project.
- **CANDI** (2025): Hybrid discrete-continuous diffusion. Another proposed next architecture — directly models mixed stall/motion signals.

See [METHODOLOGY.md](METHODOLOGY.md) for detailed discussion. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full log of 150+ experiments and proposed next architectures.

## License

MIT. See [LICENSE](LICENSE).

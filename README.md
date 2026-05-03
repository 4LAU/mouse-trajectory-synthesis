[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

# Mouse Trajectory Synthesis

Generative modeling of human mouse trajectories. Includes an evaluation framework, a corpus replay baseline, and implementations of the generative approaches tested.

---

> **Key finding:** Human mouse trajectories contain discrete zero-displacement stalls
> (6.14% of 125 Hz samples) embedded in continuous motion. These stalls produce 100%
> of the measured curvature signal. No continuous generative model (CFM, DDPM, GRU)
> can produce exact-zero displacements. This is fundamentally a mixed continuous-discrete
> generation problem, and recognizing it reframes the entire research direction.

---

## Results

All numbers are OOB Random Forest AUC on the full 4.16M-trajectory pool (n=2000 synthetic, seed 42). Lower is better: 0.50 means the classifier can't tell human from synthetic.

| Approach | AUC | Architecture | What it tells us |
|---|---|---|---|
| Corpus replay | 0.60 | kNN translate-only (50K demo pool) | Baseline, scales with pool size (reaches 0.50 at 4.16M) |
| **VQ-VAE + Transformer** | **0.892** | Discrete tokens + GRPO-finetuned AR + CFG | Best generative result: models stalls as first-class tokens |
| DDPM | 0.933 | 1D U-Net, DDIM sampling (eta=0) | Smooth conditional means lack micro-structure |
| CFM | ~0.99 | Same U-Net, Euler ODE (2-channel) | Position-only model; timing channel was never completed |
| Stall injection | ~1.0 | DDPM + post-hoc jitter | Proves post-hoc modification is a dead end |

**0.892 is the best result for fully generative mouse trajectory synthesis**, with no corpus lookup at inference time. Corpus replay scales with pool size: the shipped 50K demo pool gives ~0.60, while the full 4.16M corpus drops to ~0.50 (indistinguishable from random).

Getting below AUC 0.60 with a generative model requires solving the mixed continuous-discrete problem: continuous architectures simply cannot produce the exact-zero displacements that dominate curvature statistics.

## Problem Statement

Can generative models capture full human motor kinematics without access to a trajectory corpus at inference time?

This research started from a practical need to synthesize realistic mouse trajectories. The question turned out to be deeper than expected: human motor control produces trajectories with statistical signatures that current generative architectures can't fully reproduce. The core difficulty isn't capacity or training data. It's a structural mismatch between continuous generation and the mixed continuous-discrete nature of real mouse signals.

This project provides an evaluation framework, a corpus replay baseline, and implementations of the generative approaches tested across 145+ experiments, developed over approximately one week of active research using [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch) framework for autonomous experiment iteration.

## Evaluation

The evaluation is adversarial: a Random Forest classifier tries to distinguish generated trajectories from real ones. Lower AUC = more human-like generation.

- **18 kinematic features** spanning the full motor control stack (velocity, acceleration, jerk, curvature, angular velocity, pause statistics)
- **Random Forest OOB AUC**: no held-out split needed, adversarial by construction
- **Distributed feature importance**: no single feature dominates, so generation must be realistic across the full kinematic profile
- **Scale-invariant**: features are computed on normalized arc-length trajectories

See [METHODOLOGY.md](METHODOLOGY.md) for the full evaluation framework, feature definitions, and the stall analysis that identified the continuous-discrete gap.

## Quick Start

```bash
git clone https://github.com/4LAU/mouse-trajectory-synthesis.git
cd mouse-trajectory-synthesis
pip install -e .
python setup_data.py                                          # downloads checkpoints + eval data
python evaluate.py --experiment experiments.vqvae_ar_transformer  # reproduce AUC ~0.892
python evaluate.py --experiment experiments.ddpm_arclen           # reproduce AUC ~0.933
python evaluate.py --experiment experiments.corpus_replay         # reproduce AUC ~0.60
```

Note: `torch>=2.0` installs CPU-only by default from PyPI. For GPU acceleration, install PyTorch with CUDA support first (see [pytorch.org](https://pytorch.org/get-started/locally/)).

### Hardware

Initial development and iteration was done on a MacBook M4, then moved to an RTX 4070 (Lenovo Legion 7i) for faster training cycles. A single consumer GPU is sufficient for all experiments here, but more powerful hardware (e.g. A100/H100) will significantly reduce iteration time on the larger models (VQ-VAE + Transformer).

## Repository Structure

```
mouse-trajectory-synthesis/
├── evaluate.py                  # Adversarial evaluator (RF OOB AUC)
├── features.py                  # 18 kinematic feature extractors
├── test_features.py             # Feature extraction tests
├── setup_data.py                # Download checkpoints and eval data
├── experiments/
│   ├── corpus_replay.py         # kNN corpus replay baseline (AUC ~0.60)
│   ├── ddpm_arclen.py           # DDPM with arc-length param (AUC 0.933)
│   ├── cfm_unet.py              # Conditional Flow Matching (AUC ~0.99)
│   ├── vqvae_ar_transformer.py  # VQ-VAE + GRPO transformer (AUC 0.892)
│   └── ddpm_stall_injection.py  # Post-hoc stall injection (dead end)
├── models/
│   ├── temporal_unet.py         # 1D U-Net for diffusion / flow matching
│   ├── vqvae.py                 # Vector-quantized variational autoencoder
│   └── trajectory_transformer.py # Autoregressive transformer for token sequences
├── training/                    # Training scripts (reference implementations)
│   ├── prepare_training_data.py # Corpus preprocessing pipeline
│   ├── prepare_vqvae_data.py    # VQ-VAE specific data preparation
│   ├── tokenize_trajectories.py # Trajectory tokenization for discrete models
│   ├── train_cfm.py             # CFM training loop
│   ├── train_vqvae.py           # VQ-VAE training loop
│   ├── train_transformer.py     # AR transformer training loop
│   └── eval_holdout.py          # 4-discriminator holdout suite
├── notebooks/                   # Visualization notebooks
├── METHODOLOGY.md               # Evaluation framework and research findings
├── EXPERIMENTS.md               # Full log of 145+ experiments
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

Model weights are trained on these publicly available datasets. Raw trajectory data is not included in this repository.

## Related Work

- **Plamondon (1995)**: Kinematic theory of rapid human movements. The sigma-lognormal model decomposes movements into neuromuscular primitives.
- **Fitts (1954)**: Movement time as a function of distance and target width. The foundational speed-accuracy tradeoff in motor control.
- **VALL-E** (Wang et al., 2023): Discrete neural codec tokens for speech synthesis. Shows that discretization can capture fine-grained temporal structure.
- **T2M-GPT** (Zhang et al., 2023): Discrete tokens for human body motion generation. Closest architectural analogue to the VQ-VAE + Transformer approach used here.

See [METHODOLOGY.md](METHODOLOGY.md) for detailed discussion of how these relate to the stall/curvature finding.

## Experiment History

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full log of 145+ experiments, including architecture variations, hyperparameter sweeps, and the progression of findings that led to the continuous-discrete diagnosis.

## License

MIT. See [LICENSE](LICENSE).

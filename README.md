[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

# Mouse Trajectory Synthesis

Generative modeling of human mouse trajectories. Includes an evaluation framework, a corpus replay baseline, and implementations of the generative approaches tested.

## Key Insight

Human mouse movements aren't purely continuous. At 125 Hz sampling, 6.14% of all samples are exact zero-displacement stalls — the cursor sits perfectly still for one or more frames before moving again. These stalls aren't noise. They happen at specific points during movement (direction changes, deceleration phases) and they produce essentially all of the measured curvature signal in the trajectory.

This matters because continuous generative models (diffusion, flow matching, GRUs) output probability distributions over real-valued coordinates. They can get arbitrarily close to zero displacement, but they cannot produce exact zeros. The result is that every continuous model we tested — across 145+ experiments — fails to reproduce the curvature and angular velocity statistics that a classifier uses to distinguish human from synthetic trajectories.

The breakthrough came from treating this as a mixed continuous-discrete problem. VQ-VAE discretizes mouse movements into a codebook of 1024 motion tokens plus a dedicated stall token. An autoregressive transformer then generates sequences of these tokens, naturally producing both smooth motion and exact-zero stalls. This is the same insight that made discrete codecs work for speech (VALL-E) and body motion (T2M-GPT).

![Real vs Generated Trajectories](figures/trajectory_overlay.png)

> For the full analysis — 18 kinematic features, the discrete stall discovery, why each model family hits a ceiling, and 145+ experiment results — see **[METHODOLOGY.md](METHODOLOGY.md)**.

## Results

All numbers are OOB Random Forest AUC on the full 4.16M-trajectory pool (n=2000 synthetic, 5 seeds), cross-validated against a second classifier (Gradient Boosting) for robustness. Lower is better: 0.50 means the classifier can't tell human from synthetic.

| Approach | AUC (mean ± std) | Architecture | What it tells us |
|---|---|---|---|
| Corpus replay | 0.594 ± 0.008 | kNN translate-only (50K demo pool) | Calibration point; scales with pool size (reaches 0.50 at 4.16M) |
| **DDPM** | **0.930 ± 0.003** | 1D U-Net, DDIM sampling (eta=0) | Best generative result; continuous output avoids quantization artifacts |
| VQ-VAE + Transformer | 0.999 ± 0.000 | Discrete tokens + GRPO-finetuned AR + CFG | Codebook quantization introduces detectable artifacts |
| CFM | ~0.99 | Same U-Net, Euler ODE (2-channel) | Position-only model; timing channel was never completed |
| Stall injection | ~1.0 | DDPM + post-hoc jitter | Proves post-hoc modification is a dead end |

**0.930 is the best result for fully generative mouse trajectory synthesis**, with no corpus lookup at inference time. Corpus replay serves as a calibration point: the shipped 50K demo pool gives ~0.59, and the full 4.16M corpus gives ~0.50 (confirming the evaluator is well-behaved — two draws from the same distribution are indistinguishable, as expected).

> **Note on previously reported VQ-VAE results:** An earlier version of this README reported AUC 0.892 for VQ-VAE + Transformer. That number was measured against precomputed human features from a different feature extraction pipeline. When human and synthetic features are computed consistently using the same `features.py`, VQ-VAE scores 0.999 — the codebook quantization introduces acceleration and angular velocity artifacts that a classifier trivially detects. See [HANDOFF.md](HANDOFF.md) for the full investigation. DDPM and corpus replay results were unaffected by this correction.

The gap between DDPM (0.93) and corpus replay (0.59) remains the core challenge. Closing it requires a generative architecture that avoids both the blurriness of continuous diffusion models and the quantization artifacts of discrete tokenization — see [RESEARCH.md](RESEARCH.md) for a proposed approach.

![AUC by Architecture Family](figures/auc_progression.png)

![Feature Distribution Comparison](figures/feature_distributions.png)

## Problem Statement

Can generative models capture full human motor kinematics without access to a trajectory corpus at inference time?

Corpus replay works but requires shipping real user movement data — a privacy risk, a large deployment footprint, and a finite set of trajectories an adversary could fingerprint. A generative model ships only learned weights (< 10 MB), produces unique trajectories on every call, and needs no access to real user data at inference time.

This research started from a practical need to synthesize realistic mouse trajectories. The question turned out to be deeper than expected: human motor control produces trajectories with statistical signatures that current generative architectures can't fully reproduce. The core difficulty isn't capacity or training data. It's a structural mismatch between continuous generation and the mixed continuous-discrete nature of real mouse signals.

This project provides an evaluation framework, a corpus replay baseline, and implementations of the generative approaches tested across 145+ experiments, developed over approximately one week of active research using [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch) framework for autonomous experiment iteration.

## Evaluation

The evaluation is adversarial: a Random Forest classifier tries to distinguish generated trajectories from real ones. Lower AUC = more human-like generation.

- **18 kinematic features** spanning the full motor control stack (velocity, acceleration, jerk, curvature, angular velocity, pause statistics)
- **Random Forest OOB AUC**: no held-out split needed, adversarial by construction
- **Distributed feature importance**: no single feature dominates, so generation must be realistic across the full kinematic profile
- **Scale-invariant**: features are computed on normalized arc-length trajectories

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

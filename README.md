[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

# Mouse Trajectory Synthesis

Generative modeling of human mouse trajectories. Given only a start and end coordinate, synthesize a realistic cursor path with human-like kinematics. Includes an adversarial evaluation framework, a corpus replay baseline, and 200+ experiments across 9 architecture families.

## Key Insight

Human mouse movements aren't purely continuous. At 125 Hz sampling, 6.14% of all samples are exact zero-displacement stalls: the cursor sits perfectly still for one or more frames before moving again. These stalls happen at specific points during movement (direction changes, deceleration phases) and they produce essentially all of the measured curvature signal in the trajectory. Continuous models (diffusion, flow matching, GRUs) output probability distributions over real-valued coordinates. They can get arbitrarily close to zero displacement, but they cannot produce exact zeros, so every continuous model we tested plateaus somewhere between AUC 0.86 and 1.0.

The way through turned out to be a discrete one: encode each trajectory as a stream of events (a speed bin, a heading-increment bin, and an inter-event time), with the stall represented as a first-class zero-speed token instead of a rounding error. A masked-token model (MaskGIT-style, the same family used for parallel audio and image generation) trained on 4 million of these event streams gets almost all the way to indistinguishable from real recordings. Two things mattered as much as the architecture itself: the decoder has to respect the integer pixel grid the recording device actually writes to (leaving positions off-lattice alone costs about 0.05 AUC), and once the model's own kinematics were good enough, the remaining gap turned out to be a selection problem, not a generation problem. Given several candidate movements for the same start and end point, picking which one to keep, using a detector-matched judge that looks at the whole selected population rather than one candidate at a time, closed most of what was left.

![Generated Trajectories](figures/trajectory_overlay.png)

> For the full analysis - the discrete stall discovery, why every continuous model family hits a ceiling, the event-stream architecture, and the selection-as-a-set-problem result - see **[METHODOLOGY.md](METHODOLOGY.md)**.

## Results

All numbers are OOB Random Forest AUC on 18 kinematic features (n=2000 synthetic). Lower is better: 0.50 means the classifier can't tell human from synthetic.

| Approach | AUC | Type | What it tells us |
|---|---|---|---|
| Corpus replay | 0.51 | Retrieval | Theoretical floor: real trajectories, translated to match endpoints |
| Event replay (encode/decode round trip) | 0.507 | Retrieval | Real trajectories pushed through the event representation; confirms the representation is essentially lossless |
| ZIMT (magcorr) | 0.864 | Generative (historical) | Best result of the continuous-model era: causal Transformer + MDN + stall gate |
| Event-stream model (pure) | 0.652 | Generative | Masked-token event model alone, no selection: RF OOB 0.652 +/- 0.003 across 3 seeds |
| + SIR selection | 0.568 | Generative + selection | 16 candidates per movement, per-item tempered lottery against a GBM judge: 0.568 +/- 0.010 across 3 seeds |
| + set-level reselection | **0.489** | **Generative + selection (provisional)** | **An iterated adversarial loop that judges the whole selected population, not one candidate at a time. Single seed (42); 3-seed confirmation in progress.** |

**0.652 is the current best result for a single generative model with no selection step.** It ships only learned weights (about 25 MB), needs no recorded trajectory data at inference time, and every output is model-generated.

**0.568 is the current honest, multi-seed-confirmed best for the full system** (model plus per-item selection). The set-level result of 0.489 is the most promising lead in the project: at seed 42 it lands at chance level on the primary detector (RF OOB) and close to chance on a detector family (GBM) that never saw the selection judge. It has not yet been confirmed across seeds, so treat it as provisional until that run lands.

![AUC by Architecture Family](figures/auc_progression.png)

![Feature Distribution Comparison](figures/feature_distributions.png)

## Problem Statement

Can generative models capture full human motor kinematics without access to a trajectory corpus at inference time?

Corpus replay works but requires shipping real user movement data: a privacy risk, a large deployment footprint, and a finite set of trajectories an adversary could fingerprint. A generative model ships only learned weights, produces unique trajectories on every call, and needs no access to real user data at inference time.

Human motor control produces trajectories with statistical signatures that continuous generative architectures can't fully reproduce, because the signal itself is mixed continuous-discrete: smooth motion punctuated by exact stops. The event-stream model in this repository is built around that fact directly, and the story of getting there, and of how much further selection alone could close the gap, is the subject of this repository.

## Evaluation

The evaluation is adversarial: a Random Forest classifier tries to distinguish generated trajectories from real ones. Lower AUC means more human-like generation.

- **18 kinematic features** spanning the full motor control stack (velocity, acceleration, jerk, curvature, angular velocity, direction changes, path efficiency, movement duration)
- **Random Forest OOB AUC**: no held-out split needed, adversarial by construction, cross-validated against GBM and a raw-trajectory nearest-neighbor detector to make sure the result isn't an artifact of one classifier
- **Distributed feature importance**: no single feature dominates, so generation must be realistic across the full kinematic profile

## Quick Start

```bash
git clone https://github.com/4LAU/mouse-trajectory-synthesis.git
cd mouse-trajectory-synthesis
pip install -e .
python setup_data.py                                            # downloads checkpoints + eval data
python evaluate.py --experiment experiments.corpus_replay       # retrieval floor: AUC ~0.51
python evaluate.py --experiment experiments.zimt_magcorr        # historical best continuous model: AUC ~0.864
```

Note: `torch>=2.0` installs CPU-only by default from PyPI. For GPU acceleration, install PyTorch with CUDA support first (see [pytorch.org](https://pytorch.org/get-started/locally/)).

### Hardware

Developed on an RTX 4070 (12GB VRAM). A single consumer GPU is sufficient for all experiments. CPU-only inference works for corpus replay and corpus rotate.

## Reproduce the current results

All of the current-generation numbers come from one checkpoint, `event_polar_4m_fc_v2.pt`, run through `experiments/event_stream_polar.py` with different environment variables controlling the sampler and the selection layer. The exact locked recipe (and every knob that was tried and rejected along the way) is logged in [EXPERIMENTS.md](EXPERIMENTS.md); the commands below are the short version.

**Pure model, no selection (AUC ~0.652):**

```bash
.venv/Scripts/python.exe evaluate.py --experiment experiments.event_stream_polar
```
with environment variables `EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_CHOICE_TEMP=10 EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 DUR_EMPIRICAL=1`.

**+ SIR selection, the honest multi-seed best (AUC ~0.568):** add `EVENT_SIR=16 EVENT_SIR_TEMP=0.7 EVENT_SIR_DUR_DIVERSE=1` to the same command. This draws 16 candidate trajectories per movement and keeps one via a tempered lottery on a GBM judge's log-odds, the judge fit against a human reference set disjoint from the evaluation sample.

**+ set-level reselection, the provisional result (AUC ~0.489):** this one is a two-step, mostly-offline process rather than a single command.

1. Cache every candidate from the SIR pool instead of committing to one (`run_poolgen.sh` does this for a list of seeds, using `EVENT_POOL_SAVE` on top of the SIR recipe above).
2. Run `selection_lab.py --pool pool_s<seed>_k16.npz` to try selection strategies offline against the cached pool. The winning strategy fits an RF judge between a human reference half and the currently selected set, moves only the top fraction of picks toward the judge's preference each round with a decaying step size, and repeats for about 30 rounds. The script reports a proxy AUC on a held-out reference half; the final, reported number replays the winning selection through `evaluate.py` itself (via `EVENT_POOL_LOAD` / `EVENT_POOL_PICKS`), where the human class is the untouched evaluation sample no part of selection has seen.

## Architecture: event-stream polar model

The current model represents each trajectory as a sequence of events rather than a sequence of (x, y) coordinates. Each event carries a speed bin, a heading-increment bin (relative to the previous heading), and an inter-event time, and stalls are represented directly as a zero-speed bin rather than being approximated by a near-zero continuous value.

The backbone is a 6M-parameter masked bidirectional Transformer in the MaskGIT and SoundStorm style. Every event starts masked and gets revealed one group at a time in confidence order, so each generation step sees the full sequence in both directions rather than only the past.

Timing comes from a small flow-matching head instead of a fixed clock. That is what reproduces the raw, non-uniform polling intervals of real mouse hardware, where samples do not land on an even grid.

The model also takes a movement-character vector: the same 18 kinematic features the evaluator uses, fed in as a conditioning slot the model was fine-tuned to follow. At generation time we draw that vector from a kernel density estimate over a bank of real feature vectors, matched to the requested movement distance, so no single real trajectory is ever copied.

Two decode rules run after the continuous heading and speed are integrated into a path. Positions round to the integer pixel grid the recording hardware actually writes to. Slow steps, meaning speed below 2.5 px per frame, snap to whole lattice steps instead of sitting off-grid. Off-lattice positions during genuinely slow movement are something real mouse hardware cannot produce, and they turned out to be the single largest artifact in the project, worth about 0.05 AUC on otherwise real data.

Trained on 4.16 million trajectories from five public mouse-dynamics datasets. Checkpoint: `event_polar_4m_fc_v2.pt`.

See [METHODOLOGY.md](METHODOLOGY.md) for the full account of how this architecture was arrived at, including why the earlier VQ-VAE and continuous-diffusion approaches hit hard ceilings, and for the selection-as-a-set-level-problem result.

## Repository Structure

```
mouse-trajectory-synthesis/
├── evaluate.py                       # Adversarial evaluator (RF OOB AUC, GBM CV, raw-NN)
├── features.py                       # 18 kinematic feature extractors
├── selection_lab.py                  # Offline set-level selection over a cached candidate pool
├── run_poolgen.sh                    # Generates and caches SIR candidate pools per seed
├── generate_figures.py               # Regenerate README figures
├── experiments/
│   ├── corpus_replay.py              # kNN corpus replay (AUC 0.51)
│   ├── corpus_rotate.py              # Rotation + scale replay (AUC 0.686)
│   ├── event_stream_polar.py         # Event-stream polar model + sampler + SIR selection (AUC 0.652 / 0.568)
│   ├── event_replay_polar.py         # Event representation round-trip gate (AUC 0.507)
│   ├── zimt_magcorr.py               # ZIMT + magnitude correction, historical best continuous model (AUC 0.864)
│   ├── ddpm_arclen.py                # DDPM diffusion (AUC 0.862)
│   ├── vqvae_ar_transformer.py       # VQ-VAE + Transformer (AUC 0.890)
│   ├── cfm_unet.py                   # Conditional Flow Matching (AUC 0.919)
│   └── ...                           # 20+ additional experiment variants
├── models/
│   ├── event_stream_polar.py         # Masked-token event-stream architecture
│   ├── zimt.py                       # ZIMT architecture (historical)
│   ├── temporal_unet.py              # 1D U-Net for diffusion / flow matching (historical)
│   ├── vqvae.py                      # Vector-quantized variational autoencoder (historical)
│   └── trajectory_transformer.py
├── training/                         # Training scripts
│   ├── train_events_polar.py         # Event-stream pretraining
│   ├── train_events_polar_featcond.py# Movement-character conditioning fine-tune
│   ├── train_events_polar_dpo.py     # Preference-learning fine-tune (did not transfer; see METHODOLOGY.md)
│   ├── train_zimt.py                 # ZIMT training, historical (3-phase curriculum)
│   └── ...
├── autoresearch/                     # Original research code (v1-v147)
├── METHODOLOGY.md                    # Evaluation framework and research findings
├── EXPERIMENTS.md                    # Full log of 200+ experiments
└── LICENSE                           # MIT
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
- **SoundStorm** (Borsos et al., 2023): Parallel, masked-token generation of audio tokens. The direct architectural ancestor of the event-stream model's sampler.
- **T2M-GPT** (Zhang et al., 2023): Discrete tokens for human body motion generation. An early architectural analogue explored in this project's VQ-VAE experiments.
- **CANDI** (2025): Hybrid discrete-continuous diffusion. Explored as an intermediate step before the fully discrete event-stream approach.

See [METHODOLOGY.md](METHODOLOGY.md) for detailed discussion. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full log of 200+ experiments.

## License

MIT. See [LICENSE](LICENSE).

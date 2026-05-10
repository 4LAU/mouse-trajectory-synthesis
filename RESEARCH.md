# Next Step: A Novel Generative Architecture for Mouse Trajectories

This document proposes a purpose-built generative model for mouse trajectory
synthesis — one designed from scratch around the discrete stall phenomenon
identified in our research, rather than adapting existing architectures.

---

## Why a New Architecture

Every generative model tested in 145+ experiments was an existing architecture
applied to mouse trajectories: DDPM, CFM, GRU, VQ-VAE + Transformer. Each hit
the same wall: the mixed continuous-discrete nature of human mouse signals.

The VQ-VAE + Transformer approach (AUC 0.892) is the closest because it can
produce exact-zero stalls via a dedicated token. But it discretizes *all* motion
into a codebook, losing continuous precision and creating a codebook bottleneck —
as demonstrated when retraining on broader data collapsed AUC to 0.999.

The question: can we build a model that keeps exact zeros for stalls but full
continuous precision for movement?

---

## The Core Problem, Formally

Human mouse trajectories at 125 Hz are sequences of displacement vectors
(dx, dy) where:

- 93.86% of samples are continuous motion: (dx, dy) drawn from a smooth,
  correlated, time-dependent distribution
- 6.14% of samples are exact zeros: dx = 0, dy = 0, for 1-5 consecutive
  samples, occurring at specific structural points (direction changes,
  deceleration phases)

This is a **zero-inflated sequential generation** problem. The output at each
timestep is drawn from a mixture:

```
P(displacement) = π · δ(0,0) + (1-π) · f(dx, dy | context)
```

where π is the probability of a stall event and f is a continuous 2D
distribution conditioned on trajectory history and endpoint parameters.

No existing mouse trajectory generator models this mixture explicitly.

---

## Prior Art and Positioning

### Mouse Trajectory Synthesis (Direct Competitors)

| System | Year | Architecture | Handles stalls? |
|--------|------|-------------|-----------------|
| BeCAPTCHA-Mouse (Acien et al.) | 2022 | GAN + Sigma-Lognormal | No |
| DMTG (arXiv:2410.18233) | 2024 | Entropy-controlled DDIM + U-Net | No |
| Various GAN/LSTM repos | 2020-2024 | Standard GAN or LSTM | No |
| Our VQ-VAE + Transformer | 2025 | Discrete tokens + AR + GRPO | Yes (via stall token) |
| **Proposed: ZIMT** | 2025 | **Zero-inflated mixed-output + Mamba** | **Yes (native)** |

None of the published approaches identify zero-displacement stalls as a modeling
problem. Our existing VQ-VAE approach addresses stalls but sacrifices continuous
precision through full discretization.

### Structurally Similar Problems (Adjacent Domains)

| Domain | Problem | Solution | Key Reference |
|--------|---------|----------|---------------|
| Handwriting | Pen-up/pen-down in continuous strokes | MDN + Bernoulli output head | Graves, 2013 |
| Eye tracking | Fixations (stalls) in saccade sequences | Regime-switching: separate fixation/saccade generators | SP-EyeGAN, 2023 |
| Motion capture | Foot contacts in continuous body motion | Dual VQ codebooks with coupled generation | CATMO, 2024 |
| Speech | Silence in continuous audio | Implicit in codec tokens (causes mode collapse) | VALL-E, 2023 |
| Weather | Zero-rainfall in precipitation data | Bernoulli-Gamma density network | Cannon, 2008 |
| Finance | Zero-duration trades in time series | LSTM + zero-inflated exponential | DL-ZIACD, 2021 |

The handwriting approach (Graves 2013) is the closest solved analogue. Pen-up
events in handwriting are structurally identical to stall events in mouse
trajectories: discrete binary decisions embedded in continuous sequential
generation. The key difference is that Graves used RNNs — modern sequence
models (Mamba, Transformers) offer substantially better long-range modeling.

### Zero-Inflated Models in Deep Learning

Zero-inflated models are well-established in statistics (zero-inflated Poisson,
hurdle models) and have been adapted to deep learning in specific domains:

- **scVI** (Lopez et al., Nature Methods 2018): Zero-inflated VAE for genomics
- **Deep Hurdle Networks** (Kong et al., IJCAI 2020): Binary gate + continuous
  head for species abundance
- **DL-ZIACD** (Shi et al., Frontiers in Physics 2021): LSTM with zero-inflated
  output for financial time series

None of these have been applied to sequential 2D trajectory generation. The
combination of zero-inflated output heads with modern sequential architectures
for spatial trajectory generation appears to be entirely novel.

### Neuroscience Grounding

The stall phenomenon has a direct neuroscience explanation:

**Intermittent motor control theory** (Gawthrop & Loram, 2011; Alvarez Martin
et al., 2021 — directly applied to mouse pointing) proposes that motor commands
are issued as serial ballistic bursts with a refractory period of ~200-250ms,
not as continuous feedback. Stalls are the "dead time" between commands: the
previous command expires, the hand stops, a new command fires in a corrected
direction.

**Information Predictive Control** (IPC, bioRxiv 2025 preprint) formalizes this:
corrections trigger only when sensory "surprise" (prediction error) exceeds an
information-theoretic threshold. Below-threshold error means no new command,
producing exact-zero displacement until error accumulates.

Our observed stall durations (8-40ms at 125 Hz) are consistent with the
inter-command gap timescale: too fast for any feedback-driven correction
(visual feedback reaches M1 at ~70ms), but consistent with the dead time
between ballistic motor commands.

This grounding matters because it means stalls are not noise or artifacts —
they are a fundamental feature of human motor control, and any model that
cannot produce them is structurally incapable of matching human kinematics.

---

## Proposed Architecture: ZIMT (Zero-Inflated Mouse Trajectory Generator)

### Design Principle

At each timestep, the model makes two decisions:

1. **"Stall or move?"** — A learned gate outputs P(stall | context)
2. **"If moving, where?"** — A continuous distribution head outputs the
   displacement vector (dx, dy)

The stall decision is binary and produces exact (0, 0). The motion decision is
continuous and preserves full precision. The model is trained end-to-end.

### Architecture

```
Input conditioning:
  (log_distance, log_duration, cos_angle, sin_angle) → condition embedding

Per-timestep:
  [previous displacements + stall indicators + condition + endpoint state]
      │
      ▼
  ┌─────────────────────────────┐
  │  Sequence Backbone          │
  │  (Mamba SSM or Transformer) │
  │  Hidden state h_t           │
  └─────────┬───────────────────┘
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
  ┌────────┐  ┌──────────────────────┐
  │ Gate   │  │ Displacement Head    │
  │ σ(w·h) │  │ MDN: Σ πk N(μk, Σk) │
  │ → π_t  │  │ → (dx, dy)          │
  └────────┘  └──────────────────────┘
      │           │
      ▼           ▼
  P(stall) = π   P(motion) = (1-π)
  Output: (0,0)  Output: sample from MDN
```

**Sequence backbone:** Mamba (selective state space model) is the primary
candidate. Advantages over Transformer: linear-time inference for autoregressive
generation (no KV cache), native handling of continuous dynamics via
discretized state spaces, strong temporal modeling. Alternative: standard
causal Transformer if Mamba proves harder to train.

**Gate head:** A single sigmoid output predicting P(stall). Trained with binary
cross-entropy. The gate sees the full context: trajectory history, endpoint
distance, remaining fraction. This lets the model learn *when* stalls occur
(at deceleration phases, near direction changes, when endpoint error is low).

**Displacement head:** A Mixture Density Network (MDN) with K Gaussian
components in 2D. Each component has mean (μ_x, μ_y), covariance (σ_x, σ_y,
ρ), and mixture weight. Trained with negative log-likelihood on non-stall
samples. The MDN can represent multimodal displacement distributions (e.g.,
the model is uncertain whether to continue straight or begin curving).

**Endpoint conditioning:** Injected at each step as (remaining_dx / dist,
remaining_dy / dist, remaining_frac). Same approach as the current VQ-VAE
transformer, but without requiring CFG — the continuous head can directly
condition on endpoint state.

### Training

**Loss function:** Joint negative log-likelihood of the zero-inflated mixture:

```
L = -Σ_t [ z_t · log(π_t) + (1-z_t) · (log(1-π_t) + log f(dx_t, dy_t)) ]
```

where z_t = 1 if (dx_t, dy_t) = (0, 0), and f is the MDN density.

This decomposes naturally:
- Gate loss: binary cross-entropy on stall vs. non-stall (6.14% positive rate)
- MDN loss: NLL on continuous displacements, masked to non-stall steps only

**Class imbalance:** 6.14% stall rate means the gate sees ~16x more motion
samples. Options: focal loss (down-weight easy negatives), oversampling stall
boundaries, or a simple positive-class weight of ~5-10x.

**Teacher forcing:** Train autoregressively with ground-truth history (standard
for sequential models). At inference, sample from the model's own predictions.

**Curriculum:** Optionally start with short sequences (32 steps) and gradually
increase to full length (256 steps) during training.

### Why This Should Beat VQ-VAE + Transformer (AUC 0.892)

1. **No codebook bottleneck.** The VQ-VAE quantizes all motion into 1024
   tokens, losing continuous precision. ZIMT outputs continuous displacements
   directly. When the retraining attempt failed (AUC → 0.999), it was because
   the codebook couldn't represent broader data distributions. ZIMT has no
   codebook.

2. **Native stall modeling.** The stall decision is a learned binary gate, not
   a special token in a vocabulary. The model learns *when* to stall from the
   full trajectory context, not from the token transition probabilities of an
   autoregressive language model.

3. **End-to-end training.** The current pipeline is VQ-VAE → tokenizer →
   transformer → (GRPO). Each stage compounds errors. ZIMT trains a single
   model end-to-end with a single loss function.

4. **No GRPO needed.** The current best result (0.892) required RL fine-tuning
   (GRPO) on top of supervised training to close the gap from 0.93. ZIMT's
   architecture directly addresses the structural problems that necessitated
   GRPO.

5. **Physically interpretable.** The gate probability maps directly to the
   intermittent control model from neuroscience. Stall probability as a function
   of speed, endpoint distance, and trajectory phase is a scientifically
   meaningful quantity that can be compared against human motor control data.

---

## Alternative Architectures Considered

We evaluated five architectural families. ZIMT (Approach A) is the primary
recommendation, but the alternatives have merit and some components could be
combined.

### Approach A: ZIMT — Zero-Inflated MDN + Mamba (Recommended)

*Described above.* Combines proven components (MDN from Graves 2013, Mamba
from Gu & Dao 2023) with the novel zero-inflated formulation.

**Strengths:** Simple, interpretable, proven components, end-to-end trainable.
**Risks:** MDN training can suffer from mode collapse; autoregressive error
compounding during free sampling (same issue as GRU, but gated stalls may
naturally limit error accumulation by "resetting" the trajectory at each
stall boundary). Mamba for generation is less battle-tested than Transformers.
**Mitigation:** Start with Transformer backbone, switch to Mamba if training
is stable. Use teacher forcing ratio annealing to reduce exposure bias.

### Approach B: Hybrid Diffusion — Absorbing-State + Continuous

Extend CANDI (2025) to trajectory generation: a diffusion model where the stall
dimension uses absorbing-state diffusion (D3PM) and the displacement dimensions
use standard Gaussian diffusion. The model learns separate corruption schedules
for discrete and continuous components.

**Strengths:** Diffusion models produce high-quality outputs; no autoregressive
error compounding; the mathematics of mixed discrete-continuous diffusion is
being formalized (CANDI, FlowMol).
**Risks:** Research-stage (CANDI is October 2025, not yet validated on temporal
sequences); variable-length trajectory sequences are challenging for diffusion
(need to generate all timesteps simultaneously or use a latent diffusion
approach); training complexity is significantly higher.
**Verdict:** Promising long-term direction but too experimental for the first
model. Revisit after ZIMT establishes the baseline.

### Approach C: Neural Hybrid Automaton — Switching ODE

Model the trajectory as a continuous Neural ODE with two discrete modes:
"ballistic" (ODE integrates smooth motion dynamics) and "stall" (zero output).
A learned event module triggers mode transitions based on speed, endpoint
proximity, and trajectory curvature.

**Strengths:** Most physically interpretable — directly encodes intermittent
control theory. The two-mode structure maps exactly to the neuroscience: ballistic
commands produce smooth ODE trajectories, stalls are inter-command dead time.
Continuous dynamics are handled by the ODE (no discretization artifacts).
**Risks:** Neural Hybrid Automata (NeurIPS 2021) are hard to train — event
sparsity means weak gradient signal for the switching module; the ODE solver
adds computational cost; generation requires adaptive step-size integration.
**Verdict:** Scientifically elegant but practically risky. Consider for a
second-generation model after ZIMT proves the zero-inflated concept works.

### Approach D: Regime-Switching — Separate Generators

Following SP-EyeGAN (2023) for eye tracking: train separate generators for
stall sequences and motion sequences, with a higher-level controller that
decides when to switch between regimes. The motion generator could be a small
diffusion model; the stall generator models stall duration (1-5 samples).

**Strengths:** Each generator specializes in its regime; avoids the difficulty
of a single model handling both modes; stall duration distribution is simple
and can be modeled explicitly.
**Risks:** Stitching artifacts at regime boundaries (the transition from motion
to stall and back must preserve kinematic continuity — speed must decelerate
smoothly to zero before a stall and accelerate smoothly after); the controller
is a sequential decision problem that may require RL.
**Verdict:** Viable but the boundary-stitching problem is likely to consume
most of the engineering effort, and ZIMT handles boundaries naturally (the
gate probability changes gradually, not abruptly).

### Approach E: Dual-Codebook VQ — Improved Discretization

Following CATMO (2024) for motion capture: maintain two VQ codebooks (one for
motion tokens, one for stall/timing tokens) with coupled autoregressive
generation. Alternatively, replace VQ-VAE with Finite Scalar Quantization
(FSQ, ICLR 2024) to avoid codebook collapse.

**Strengths:** Stays in the discrete token framework (proven to work for stalls);
FSQ eliminates the codebook collapse problem; dual-codebook explicitly separates
stall decisions from motion quality.
**Risks:** Still discretizes motion, losing continuous precision; still requires
a multi-stage pipeline (quantize → tokenize → generate → decode); FSQ has not
been validated for 2D displacement data.
**Verdict:** An incremental improvement over the current VQ-VAE approach, not
the architectural leap needed. Pursue ZIMT instead.

---

## Research Plan

### Phase 1: Baseline ZIMT (Autoresearch, ~1-2 nights on RTX 4070)

Build the core architecture and validate that zero-inflated output heads work
for trajectory generation.

1. Implement ZIMT with Transformer backbone (safer starting point than Mamba)
2. Train on the same 200K trajectory subset used for the current transformer
3. Evaluate against the RF discriminator (target: AUC < 0.90)
4. Verify that the gate learns to produce stalls at the right rate (~6%)
5. Compare curvature statistics against human data

**Autoresearch search space:**
- MDN components: K ∈ {3, 5, 8, 12}
- Hidden dimension: d ∈ {128, 256, 512}
- Number of layers: L ∈ {4, 6, 8}
- Gate loss weight: w ∈ {1, 5, 10}
- Sequence length: T ∈ {64, 128, 256}

**Success criteria:** AUC < 0.90 and stall rate within 1% of human (6.14%).

### Phase 2: Scaling and Mamba (1-2 additional nights)

1. Switch backbone from Transformer to Mamba
2. Scale training data to full 4.16M corpus
3. Add endpoint conditioning (remaining distance/angle injection)
4. Tune with Autoresearch: CFG scale, MDN temperature, sampling strategy

**Success criteria:** AUC < 0.85 (beating current best of 0.892).

### Phase 3: Adversarial Fine-Tuning (optional)

If Phase 2 reaches AUC < 0.85, apply adversarial fine-tuning:
1. Use the RF AUC as a reward signal (GRPO or REINFORCE)
2. Alternatively, train a neural discriminator alongside (adversarial training)
3. Target: AUC < 0.70

### Phase 4: Ablation and Validation

Run the critical experiments for scientific credibility:
1. **Stall ablation:** ZIMT without the gate (force π=0) — prove the gate matters
2. **Gate-only ablation:** ZIMT with simple Gaussian instead of MDN — isolate
   contribution of the mixture density head
3. **BeCAPTCHA-Mouse benchmark:** External validation against published bot
   detector
4. **Multi-seed evaluation:** 10 seeds with confidence intervals for all models

---

## What Makes This Publishable

This is not just "another model for mouse trajectories." The contribution is
the synthesis of three previously unconnected ideas:

1. **A well-characterized empirical phenomenon** (zero-displacement stalls in
   mouse trajectories) documented across 4.16M trajectories from 5 datasets

2. **A neuroscience explanation** (intermittent motor control, IPC) that predicts
   the phenomenon from first principles and explains why it matters for
   generative modeling

3. **An architectural solution** (zero-inflated sequential generation) that
   combines established components in a novel configuration, directly motivated
   by the empirical finding and neuroscience theory

The closest prior work in each dimension:
- Empirical: our own METHODOLOGY.md analysis (no published work identifies this)
- Neuroscience: Alvarez Martin et al. 2021 (intermittent control for mouse
  pointing, but not for generative modeling)
- Architecture: Graves 2013 (MDN + Bernoulli for handwriting, but not
  zero-inflated, not for mouse trajectories, and using 2013-era RNNs)

The gap between these three lines of work is the contribution.

---

## Key References

Papers the training machine should download and study before implementation.
ArXiv links and GitHub repos provided where available.

### Mouse Trajectory Synthesis
- Acien et al. (2022). "BeCAPTCHA-Mouse." *Pattern Recognition.* arXiv: https://arxiv.org/abs/2005.00890 — GitHub: https://github.com/BiDAlab/BeCAPTCHA-Mouse
- DMTG (2024). "Diffusion-based mouse trajectory generation." arXiv: https://arxiv.org/abs/2410.18233

### Zero-Inflated / Hurdle Models (Study These for the Output Head Design)
- Kong et al. (2020). "Deep Hurdle Networks for Zero-Inflated Multi-Target Regression." *IJCAI.* https://www.ijcai.org/proceedings/2020/0603.pdf
- Shi et al. (2021). "DL-ZIACD: zero-inflated autoregressive conditional duration." *Frontiers in Physics.* https://www.frontiersin.org/articles/10.3389/fphy.2021.651528
- Cannon (2008). "Bernoulli-Gamma density network for precipitation." *J. Hydrometeorology.* https://journals.ametsoc.org/view/journals/hydr/9/6/2008jhm960_1.xml
- Lopez et al. (2018). "scVI: deep generative modeling for single-cell transcriptomics." *Nature Methods.* https://pmc.ncbi.nlm.nih.gov/articles/PMC6289068/

### Handwriting / Pen-Lift (Read This First — Closest Structural Analogue)
- Graves (2013). "Generating Sequences With Recurrent Neural Networks." arXiv: https://arxiv.org/abs/1308.0850 — The MDN + Bernoulli output head architecture is the direct inspiration for ZIMT's gate + displacement head.

### Motor Control Theory (Read for Scientific Grounding)
- Gawthrop & Loram (2011). "Intermittent control." *Biological Cybernetics.* https://link.springer.com/article/10.1007/s00422-010-0416-4
- Alvarez Martin et al. (2021). "Intermittent control as a model of mouse movements." *ACM TOCHI.* https://dl.acm.org/doi/10.1145/3461836 — Directly applies intermittent control to mouse pointing.
- IPC (2025). "Intermittent movement control emerges from information-based planning." *bioRxiv preprint.* https://www.biorxiv.org/content/10.1101/2025.03.05.641580v1 — Formalizes why stalls occur.
- Todorov & Jordan (2002). "Optimal feedback control as a theory of motor coordination." *Nature Neuroscience.*

### Modern Architectures (Study for Backbone Selection)
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv: https://arxiv.org/abs/2312.00752 — GitHub: https://github.com/state-spaces/mamba — Primary backbone candidate.
- Austin et al. (2021). "D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces." *NeurIPS.* arXiv: https://arxiv.org/abs/2107.03006
- CANDI (2025). "Hybrid discrete-continuous diffusion." arXiv: https://arxiv.org/abs/2510.22510 — Relevant for Approach B (hybrid diffusion).
- FlowMol (2024). "Mixed continuous/categorical flow matching." *NeurIPS workshop.* https://pmc.ncbi.nlm.nih.gov/articles/PMC11092876/
- HART (2024). "Hybrid Autoregressive Transformer." MIT. https://hanlab.mit.edu/projects/hart

### Adjacent Domains (Study for Design Patterns)
- SP-EyeGAN (2023). "Regime-switching eye movement generation." *ACM ETRA.* https://dl.acm.org/doi/fullHtml/10.1145/3588015.3588410
- CATMO (2024). "Contact-aware motion generation with dual codebooks." arXiv: https://arxiv.org/abs/2403.15709
- Bishop (1994). "Mixture Density Networks." https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf — Foundational MDN paper.

### Implementation References
- Finite Scalar Quantization (FSQ, ICLR 2024): https://openreview.net/forum?id=8ishA3LxN8 — VQ-VAE alternative if needed.
- Neural Hybrid Automata (NeurIPS 2021): https://arxiv.org/abs/2106.04165 — Relevant for Approach C.

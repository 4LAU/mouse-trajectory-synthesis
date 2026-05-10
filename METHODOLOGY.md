# Methodology: Evaluating Generative Mouse Trajectory Synthesis

This document describes the evaluation framework, key empirical discoveries, and
representational analysis behind this research. It is written for an ML
researcher who wants to understand what we measure, what we found in the data,
and why certain model families hit hard ceilings.

This research originated from a practical need to synthesize realistic mouse
trajectories. The central question: **can a fully generative model produce
trajectories that are statistically indistinguishable from human motor output,
without replaying or retrieving from a recorded corpus?**

The constraint is strict. The generator receives only a start coordinate and an
end coordinate. It must synthesize every point from learned parameters or a
trained model - no corpus lookup, no nearest-neighbor retrieval, no template
replay. Model weights loaded at import time are permitted; trajectory databases
are not.

The current state of the art for fully generative synthesis under this
constraint is **AUC 0.892** (VQ-VAE + GRPO-finetuned autoregressive transformer
with classifier-free guidance). The theoretical floor - corpus replay of the
same distribution - is AUC ~0.50 (with the full 4.16M-trajectory pool),
indistinguishable from random. The gap between 0.892 and 0.50 is dominated by a
single phenomenon: discrete stall events embedded in continuous motion, which no
continuous generative model can produce.

---

## Table of Contents

1. [Research Goal](#1-research-goal)
2. [Evaluation Framework](#2-evaluation-framework)
3. [Key Data Discoveries](#3-key-data-discoveries)
4. [Discrete Stall Events: The Key Insight](#4-discrete-stall-events-the-key-insight)
5. [Representational Limitation Analysis](#5-representational-limitation-analysis)
6. [Perturbed Replay: Proof the Gap Is Bridgeable](#6-perturbed-replay-proof-the-gap-is-bridgeable)
7. [Related Work](#7-related-work)

---

## 1. Research Goal

### The Generative Constraint

Mouse trajectory synthesis has an easy solution: record millions of real
trajectories and replay them with minor transformations. A nearest-neighbor
retrieval system matched on angle and distance, applied with translation only
(no rotation - rotation destroys angular dynamics), achieves AUC 0.498 against
a 4.16-million-trajectory corpus. This serves as a **calibration point**: it
confirms the evaluator is well-behaved (two draws from the same distribution are
indistinguishable, as expected) and establishes the scale of the AUC axis. It is
not a research finding - it is a sanity check that validates the evaluation
framework before applying it to generative models.

But replay is not generation. It requires a large trajectory database at
inference time. This creates three practical problems: (1) **privacy** - the
corpus contains real user movement data, and shipping it means distributing
behavioral biometric signatures that could identify individuals; (2) **deployment
size** - a 4.16M-trajectory corpus is hundreds of megabytes, impractical for
client-side or embedded deployment; (3) **fingerprinting risk** - a finite corpus
means repeated trajectories, which an adversary could detect by matching against
a known copy of the database.

The research question asks whether a model can learn the underlying structure of
human mouse movement well enough to synthesize novel trajectories from scratch -
trajectories that are not copies or perturbations of recorded data, but genuinely
new sequences that obey the same motor control dynamics. A generative model ships
only learned weights (< 10 MB), produces unique trajectories on every call, and
requires no access to real user data at inference time.

This is a harder problem than it appears. Human mouse trajectories encode the
full kinematic stack of the motor control system: velocity profiles shaped by
Fitts' law, acceleration asymmetries from the stretch-shortening cycle,
jerk signatures from motor unit recruitment, and (critically) discrete
micro-events where the hand physically stops and changes direction. A
generative model must capture all of these simultaneously.

### What We Measure

The metric is adversarial: train a classifier to distinguish synthetic
trajectories from real ones, and measure how well it succeeds. Lower AUC means
better synthesis. An AUC of 0.50 means the classifier cannot tell the
difference. An AUC of 1.00 means every synthetic trajectory is trivially
identifiable.

We chose this adversarial framing over per-feature distributional matching
(e.g., Wasserstein distance on individual features) because the classifier can
detect joint distribution mismatches that marginal comparisons miss. In human
data, `mean_acceleration` and `mean_jerk` are uncorrelated (r = -0.025). In
every continuous generative model we tested, they are near-perfectly correlated
(r = 0.999). Per-feature metrics would not flag this; the classifier detects it
immediately.

---

## 2. Evaluation Framework

### The 18 Kinematic Features

Every trajectory - human and synthetic - is preprocessed to a uniform temporal
grid at 125 Hz via linear interpolation, then reduced to an 18-dimensional
feature vector. These features cover the full kinematic stack from zeroth-order
geometry through third-order dynamics:

#### Velocity (4 features)

| Feature | Description |
|---------|-------------|
| `mean_velocity` | Mean instantaneous speed (px/s). Human mean ~960 px/s. |
| `std_velocity` | Standard deviation of speed. Captures the range from stalls to ballistic peaks. |
| `max_velocity` | Peak instantaneous speed. Human data shows extreme peaks: coefficient of variation ~34x, meaning max speed routinely exceeds 30x the mean. |
| `velocity_skewness` | Skewness of the speed distribution. Positive skew (~1.0) reflects the long deceleration tail. |

#### Acceleration (3 features)

| Feature | Description |
|---------|-------------|
| `mean_acceleration` | Mean rate of speed change. Slightly negative in human data (~-2033 px/s^2), reflecting a deceleration bias - movements spend more time slowing down than speeding up. |
| `std_acceleration` | Variability of acceleration. High values indicate rapid transitions between acceleration and deceleration phases. |
| `max_acceleration` | Peak absolute acceleration (~242K px/s^2). Captures the most extreme speed changes, typically at movement onset or during corrective submovements. |

#### Jerk (2 features)

| Feature | Description |
|---------|-------------|
| `mean_jerk` | Mean rate of acceleration change (third derivative). Slightly negative in human data, reflecting asymmetric motor control dynamics. |
| `std_jerk` | Standard deviation of jerk (~10.7M). The extreme range reflects the difference between smooth ballistic phases and abrupt corrective adjustments. |

#### Path Geometry (2 features)

| Feature | Description |
|---------|-------------|
| `path_efficiency` | Ratio of straight-line distance to total path length. Human mean ~0.84, indicating moderate but consistent arc. Values near 1.0 would indicate perfectly straight movement; values near 0 would indicate wandering. |
| `max_deviation` | Maximum perpendicular distance from the start-to-end line (~54.6 px). Captures the spatial extent of path curvature and corrective movements. |

#### Curvature (2 features)

| Feature | Description |
|---------|-------------|
| `curvature_mean` | Mean unsigned curvature, computed as \|v x a\| / \|v\|^3. Human mean ~1329. This feature proved to be the most diagnostic: it is dominated by near-zero-speed moments where the denominator approaches zero. |
| `curvature_std` | Standard deviation of curvature. The extreme variance reflects the mixture of near-zero curvature during ballistic phases and extreme curvature during stalls. |

#### Direction Changes (1 feature)

| Feature | Description |
|---------|-------------|
| `num_direction_changes` | Number of sign changes in the wrapped angular difference series. Human mean ~27. Captures the frequency of corrective micro-adjustments during movement. |

#### Timing (2 features)

| Feature | Description |
|---------|-------------|
| `movement_duration` | Total trajectory duration in seconds. Human mean ~0.54s. |
| `time_to_peak_velocity` | Fraction of total duration at which peak speed occurs. Human mean ~0.34, indicating peak speed early in the movement with a long deceleration tail. |

#### Angular Velocity (2 features)

| Feature | Description |
|---------|-------------|
| `angular_velocity_mean` | Mean absolute angular velocity (~22 rad/s). Measures how rapidly the movement direction changes over time. |
| `angular_velocity_std` | Standard deviation of angular velocity (~45 rad/s). High variance reflects the alternation between straight ballistic segments and sharp corrective turns. |

### Why Random Forest with OOB AUC

The classifier is a Random Forest with 100 trees, using out-of-bag (OOB) AUC as
the evaluation metric. This choice is deliberate:

**OOB eliminates the need for a held-out test set.** Each tree in the forest is
trained on a bootstrap sample (~63% of the data). The remaining ~37% are "out of
bag" for that tree. The OOB prediction for each sample aggregates only the trees
that did not see it during training. This gives a nearly unbiased estimate of
generalization performance without requiring a train/test split, which matters
when the synthetic distribution changes with every experiment.

**Random Forest is a strong non-parametric baseline.** It can detect nonlinear
interactions between features (e.g., the correlation structure between
`mean_acceleration` and `mean_jerk`) without requiring feature engineering. It
does not assume any particular distributional form.

**AUC is threshold-invariant.** It measures the classifier's ability to rank
synthetic trajectories as more likely synthetic, regardless of the decision
threshold. This is more informative than accuracy for our use case, where the
class balance is always 1:1 by construction.

### Feature Importance Is Distributed

A critical property of this evaluation framework is that **no single feature
dominates the classifier's decision**. The top feature (`angular_velocity_std`)
accounts for only 10.8% of total importance. The top 5 features together account
for 41%. This means a generator cannot achieve low AUC by matching one or two
features - it must match the full joint distribution across all 18 dimensions.

This is both a strength and a challenge. It is a strength because it makes the
evaluation robust: a model that games one feature will be caught by the others.
It is a challenge because it means there is no single "fix" - improvement
requires simultaneously matching velocity profiles, acceleration dynamics, jerk
statistics, curvature, angular velocity, timing, and path geometry.

### Why This Is Principled

The 18 features span every level of the kinematic hierarchy:

- **Position** → path_efficiency, max_deviation
- **Velocity** (1st derivative) → mean/std/max/skew velocity, time_to_peak_velocity
- **Acceleration** (2nd derivative) → mean/std/max acceleration
- **Jerk** (3rd derivative) → mean/std jerk
- **Curvature** (cross-product formulation) → curvature mean/std
- **Angular dynamics** → angular_velocity mean/std, num_direction_changes
- **Timing** → movement_duration

This covers the full stack from zeroth-order geometry (where the cursor went)
through third-order dynamics (how smoothly it changed acceleration). Any
systematic difference in motor control dynamics between synthetic and human
trajectories will manifest in at least one of these levels.

The features are also computed after temporal resampling at 125 Hz, which is
critical. Raw mouse event timestamps are irregular (typically 8ms intervals from
125 Hz USB polling, but with jitter and dropped samples). Resampling to a
uniform grid via linear interpolation ensures that all features are computed on
a consistent time base, and preserves the kinematic structure. This resampling
choice itself was validated empirically (see Section 3).

---

## 3. Key Data Discoveries

The following findings emerged from analysis of 4.16 million mouse trajectories
collected from five publicly available human-computer interaction datasets. These
discoveries shaped both the evaluation framework and the generator architecture
choices.

### 3.1 The Human Velocity Profile

The velocity profile of a typical aimed mouse movement is **bell-shaped,
asymmetric, and remarkably consistent across individuals and distances**.

- **Peak speed occurs at ~35% of the movement duration** (mean
  `time_to_peak_velocity` = 0.34). This is universal: the correlation between
  peak location and movement distance is essentially zero (r = 0.005).
- **Deceleration takes 2-5x longer than acceleration.** The velocity profile
  rises sharply to its peak, then falls gradually with a long tail. This
  produces a positive speed skewness (~1.0).
- **Peak ratio and speed CV are correlated with distance** (r = 0.43 and 0.42
  respectively), meaning longer movements have proportionally higher peak speeds
  relative to their average.
- **Peak fraction and asymmetry are strongly anti-correlated** (r = -0.81): when
  the peak occurs earlier, the deceleration phase is longer and more drawn out.

This asymmetric bell shape is consistent with the known physiology of aimed
movements. The acceleration phase is driven by a single ballistic motor command,
while the deceleration phase involves online feedback correction - a slower,
more variable process.

The extreme velocity peaks deserve special attention. The coefficient of
variation of max velocity in the human data is approximately 34x - meaning the
fastest instantaneous speed in a trajectory routinely exceeds 30 times the mean
speed. This is far larger than what smooth parametric models produce. The
sigma-lognormal model, for example, achieves max velocity only 1/22 of the human
value. This extreme peakiness appears to arise from brief ballistic bursts at
movement onset, where the hand accelerates maximally before feedback control
engages.

The practical consequence for generation: any model that produces symmetric
velocity profiles, or profiles where peak speed is not reliably at 35%, will be
detected by the classifier through `time_to_peak_velocity`, `velocity_skewness`,
and `mean_acceleration` (which is slightly negative due to the deceleration
bias). Furthermore, models that fail to capture the extreme peak-to-mean ratio
will be detected through `max_velocity` and `std_velocity`.

### 3.2 Timing Residuals Are Lognormal with High Autocorrelation

After fitting a curvature-speed map (the well-known inverse relationship between
instantaneous speed and path curvature), the residual timing deviations from the
map's predictions have two important properties:

1. **Lognormal distribution** with sigma = 1.08. The map explains roughly half
   the variance in timing. The remaining half is not noise - it is structured.

2. **High autocorrelation: r = 0.65 between consecutive segments.** This means
   the timing micro-structure is smooth. If one segment is faster than the map
   predicts, the next segment is likely to be faster too. This is consistent
   with motor control theory: motor commands are issued in bursts, and the
   resulting timing deviations are temporally correlated.

This has a direct consequence for generation. Any approach that generates timing
independently for each point (e.g., sampling i.i.d. noise on top of a
curvature-speed map) will produce timing that is too "jittery" - the
autocorrelation will be near zero instead of 0.65. The classifier detects this
through the joint distribution of velocity features, which encode temporal
smoothness implicitly.

This finding explains why curvature-speed map approaches with calibrated noise
plateau at AUC 0.89 regardless of noise amplitude: the noise is uncorrelated,
but human timing deviations are not.

### 3.3 Resampling Matters: Arc-Length Destroys Curvature

A trajectory can be resampled in two ways:

- **Arc-length resampling**: place N points at equal spatial intervals along the
  path. This is a natural choice for diffusion models, which benefit from
  uniform point spacing.
- **Temporal resampling**: place points at uniform time intervals (e.g., every
  8ms at 125 Hz). This preserves the original timing structure.

We compared the two approaches by resampling the same human trajectories both
ways and measuring feature preservation:

| Resampling | Curvature correlation with original | Angular velocity correlation | RF AUC (resampled vs. original) |
|------------|-------------------------------------|-----------------------------|---------------------------------|
| Arc-length (100 points) | r = 0.01 | r = 0.01 | ~0.93 |
| Temporal (125 Hz) | r > 0.99 | r > 0.99 | 0.17 |

Arc-length resampling **destroys** curvature and angular velocity information.
The correlation between arc-length-resampled features and the originals is
essentially zero for these two feature groups. This happens because curvature
depends on the velocity denominator (kappa = |v x a| / |v|^3), and arc-length
resampling removes all velocity information by construction - every segment has
equal length.

Temporal resampling at 125 Hz preserves all 18 features with near-perfect
correlation, and an RF classifier trained to distinguish temporally-resampled
trajectories from originals achieves only AUC 0.17 (worse than random guessing,
indicating the distributions are nearly identical).

This finding has architectural implications. **Any model trained on
arc-length-resampled data has a hard ceiling of ~0.93 AUC**, because the
resampling itself introduces detectable artifacts in curvature and angular
velocity. Models must either operate in the temporal domain or find a way to
reconstruct timing information post-generation.

### 3.4 Path-Timing Coupling

The classifier's strongest signal comes from **mismatches between path geometry
and velocity profile**. In human data, turns correspond to decelerations and
straight segments correspond to accelerations, with specific asymmetric
profiles. This coupling contains more discriminative information than either path
geometry or timing alone.

Evidence for this comes from a series of experiments that generated path and
timing independently:

- **CFM path + pool timing retrieval** (timing from a different trajectory than
  the path): AUC 1.00. The classifier trivially detects the mismatch.
- **KNN velocity profile lookup** (timing borrowed from the most similar
  trajectory by distance): AUC 0.95. Even "similar" timing from a different
  trajectory is detectably wrong.
- **Parametric velocity profile** (bell curve from population statistics, no
  coupling to path): AUC 0.97. Population-average timing applied to individual
  paths fails.

Contrast with:
- **Perturbed replay** (same trajectory's timing + 2% noise): AUC 0.70. When
  path and timing come from the same trajectory, even with noise added, the
  coupling is preserved and the classifier struggles.

This demonstrates that path-timing coupling is not a nuisance variable to be
modeled away. It is the core of what makes a trajectory look human. Any
generator that treats path shape and velocity profile as independent channels
will fail.

---

## 4. Discrete Stall Events: The Key Insight

This section describes the single most important finding from the data analysis.
It explains why all continuous generative models hit the same performance ceiling
and identifies the architectural requirement for breaking through it.

### 4.1 Where Curvature Comes From

Curvature, as computed in the feature extraction pipeline, is:

```
kappa = |v x a| / |v|^3
```

where `v` is the velocity vector and `a` is the acceleration vector. The cubic
speed term in the denominator means that curvature is overwhelmingly dominated
by low-speed moments.

Analysis of the human trajectory corpus reveals a striking fact: **100% of
meaningful curvature comes from time steps where speed is below 5 px/s.** During
high-speed ballistic phases (speed > 100 px/s), the curvature contribution per
step is negligible:

```
High speed: kappa ≈ angular_velocity / speed ≈ 0.2 / 1000 ≈ 0.0002
Low speed:  kappa ≈ angular_velocity / speed ≈ 0.2 / 0.001 ≈ 200
```

The curvature feature is, in effect, a proxy for the structure of near-zero-speed
events. Any model that never produces near-zero speeds will have near-zero
curvature, regardless of how well it matches other kinematic features.

### 4.2 The Stall Pattern

Closer examination of what happens at these low-speed moments reveals a discrete
structure. Human trajectories do not merely slow down - they **stop completely**
for brief intervals.

Specifically, **6.14% of all time steps in the human corpus have exactly zero
displacement**: dx = 0, dy = 0, for 1-5 consecutive 8ms samples. The pattern at
each stall is:

```
Phase 1 - Deceleration:
  Speed ramps down smoothly: 322 → 161 → 47 → 0 px/s

Phase 2 - Hold:
  1-5 consecutive samples at exactly 0 px/s
  (cursor position unchanged between samples)

Phase 3 - Acceleration:
  Speed ramps up smoothly: 0 → 24 → 62 → 94 px/s

Phase 4 - Direction change:
  Heading BEFORE stall ≠ heading AFTER stall
  Typical heading change: 5-30 degrees
```

Statistics from the corpus:
- 6.14% of steps have speed < 1 px/s
- 8.80% of steps have speed < 10 px/s
- Stalls last 1-5 consecutive samples (8-40ms at 125 Hz polling)
- At stall boundaries, heading changes by 5-30 degrees

### 4.3 Why This Is a Discrete Event

The zero-displacement samples are not "very small" displacements that round to
zero. They are exact zeros: the cursor did not move between two consecutive
polling intervals. This happens because the motor system has a minimum force
threshold below which no movement occurs - the hand is physically stationary on
the mouse, with sub-pixel position changes below the sensor's resolution.

The consequence is that curvature in human trajectories is generated by a
**discrete event** (stop, change heading, resume) embedded in an otherwise
continuous motion signal. This is not a continuous property of the path shape. It
is a sequence of identifiable events:

1. Smooth deceleration to zero
2. A discrete hold period (integer number of samples)
3. Smooth acceleration with a new heading

The heading change at the stall boundary is what creates the curvature. The
near-zero speed at the stall center is what makes the curvature large (because
of the |v|^3 denominator). Both components are necessary.

### 4.4 The Implication for Generative Models

No continuous generative model can produce exactly zero displacement. By
construction:

- **Diffusion models** (DDPM, CFM) output continuous values via a learned vector
  field or noise schedule. The probability of the output being exactly (0, 0)
  is measure-zero in the continuous output space.
- **Autoregressive models** with continuous outputs (GRU with Gaussian NLL) face
  the same issue - the predicted mean is never exactly zero, and sampling from
  a Gaussian around it never produces exact zeros.
- **Parametric models** (sigma-lognormal, minimum-jerk submovements) produce
  smooth analytic curves that pass through zero speed only instantaneously, not
  for discrete intervals.

This is not a training data problem or a model capacity problem. It is a
**representational limitation**. The output space of continuous models does not
include the discrete zero-displacement events that generate human curvature.

---

## 5. Representational Limitation Analysis

### 5.1 Model Family Comparison

The following table summarizes the ceiling each model family encounters, the
root cause, and whether the architecture can in principle produce the
zero-displacement events that drive human curvature:

| Model Family | Can produce dx=0, dy=0? | Curvature achieved | AUC ceiling | Root cause of ceiling |
|---|---|---|---|---|
| Diffusion (DDPM/CFM), deterministic | No - continuous output | ~7 | ~0.93 | ODE/SDE integration produces smooth conditional means |
| Diffusion (DDPM/CFM), stochastic | No - adds continuous noise | ~7 (+ noise artifacts) | ~0.95 | Uncorrelated noise destroys spatial coherence without creating structure |
| Autoregressive GRU (continuous) | No - Gaussian output | ~0.3 | ~0.99 | Error compounds autoregressively; max velocity diverges to 11M px/s |
| Parametric submovement (min-jerk) | No - smooth analytic curves | ~0.3 | 1.00 | Additive submovements stack velocity; segments are individually smooth |
| Speed-bin GRU (discrete speed, continuous XY) | Approximate (near-zero bin) | ~0.3 | ~0.99 | Spatial drift in XY accumulates; path efficiency 0.59 vs human 0.84 |
| DDPM + post-hoc speed dips | Approximate (near-zero) | ~15 | ~0.999 | Perturbations create direction changes at non-zero speed, not at stalls |
| **VQ-VAE + AR Transformer** | **Yes - stall token (0,0)** | **Unbounded** | **Unknown** | **First architecture to match the problem's mixed discrete-continuous structure** |

### 5.2 Diffusion Models: The Conditional Mean Problem

Both Conditional Flow Matching (CFM) and Denoising Diffusion Probabilistic
Models (DDPM) were tested extensively. The key finding: **for this task, CFM and
DDPM are functionally equivalent.** Deterministic sampling (DDIM with eta=0 for
DDPM, Euler ODE for CFM) from both produces smooth conditional-mean paths. The
training objective differs (flow matching vs. noise prediction), but the
generated output is the same: the expected value of the data distribution
conditioned on the endpoints.

AUC comparison:
- CFM (100-point arc-length): 0.9191
- DDPM (100-point arc-length, eta=0): 0.9291
- DDPM (192-point temporal, eta=0): 0.9546

The curvature for all three is below 10 (vs. human 1329). No combination of
architecture (U-Net, Transformer), training data representation (arc-length,
temporal), or sampling strategy (deterministic, stochastic) produced curvature
above 15.

Stochastic DDPM sampling (eta > 0) was tested explicitly:
- eta = 0.1: AUC 0.9444. Path efficiency drops, max deviation increases.
- eta = 0.5: AUC 0.998. Catastrophic spatial incoherence (max velocity 628K).
- The noise is uncorrelated per-point. It destroys spatial coherence without
  creating the structured decel-hold-accel pattern of human stalls.

### 5.3 Autoregressive Models: The Divergence Problem

Autoregressive models (GRU, LSTM) that predict (dx, dy, dt) at each step face a
different failure mode: **error compounding during free sampling.**

A 2-layer 256-hidden GRU trained with teacher forcing achieves excellent
training loss (NLL = -13), indicating the model has learned the local
transition dynamics. But during free sampling (inference without ground truth),
small errors in early predictions shift the conditioning distribution for later
predictions. Without a self-correction mechanism, errors accumulate
exponentially. The result: max velocity diverges to 11 million px/s (vs. human
3,884), and trajectories spiral away from the target.

Quantizing speed into discrete bins (16 or 64 classes) prevents the velocity
explosion - max velocity is bounded by the maximum bin center. But the
underlying spatial drift remains. The GRU's XY predictions accumulate error
regardless of speed prediction quality. Counter-intuitively, higher speed
prediction accuracy (66% with 16 bins vs. 36% with 64 bins) produced **worse**
overall AUC (0.9988 vs. 0.9897), confirming that spatial drift, not speed
prediction, is the bottleneck.

### 5.4 Parametric Submovement Models: The Velocity Stacking Problem

Minimum-jerk submovement decomposition is theoretically appealing: decompose
each trajectory into 2-4 overlapping submovements, each following the smooth
minimum-jerk profile s(tau) = 10tau^3 - 15tau^4 + 6tau^5, and compose them to
produce the full movement.

Training data analysis revealed that human trajectories contain a mean of 6.7
velocity peaks per trajectory - more than the 2-4 predicted by the classical
literature for simple aimed movements. Trough speed averages 35% of peak speed
(not near-zero), indicating substantial overlap between submovements.

Three composition approaches were tested:

- **Additive composition**: Sum the velocity contributions of overlapping
  submovements. Result: velocity stacks multiplicatively. N overlapping
  submovements produce N times the expected velocity. Mean velocity reaches
  5,000-10,000 px/s (vs. human 960). AUC = 1.00.
- **Correlated composition**: Same additive model with correlated submovement
  directions. Same velocity stacking. AUC = 0.9997.
- **Sequential composition**: Non-overlapping submovement segments. Eliminates
  velocity stacking, but each segment is individually smooth with no stalls.
  Curvature near zero. AUC = 1.00.

The key insight from this failure: **human submovements do not sum additively.**
Motor cortex submovement composition is competitive (winner-takes-all), not
additive. At any given moment, the active submovement dominates rather than
combining with others. The additive assumption, standard in the kinematic theory
literature, produces velocity distributions that are qualitatively wrong for
mouse trajectory synthesis.

### 5.5 VQ-VAE with Autoregressive Transformer: Matching the Problem Structure

The representational limitation analysis points to a specific architectural
requirement: the model must have **a discrete zero-displacement token as a
first-class output.**

The VQ-VAE + autoregressive transformer architecture meets this requirement:

1. **Quantize** each (dx, dy) displacement into one of ~1024 motion tokens using
   a learned codebook, plus a dedicated stall token (id = 0) that maps to
   exactly (0, 0).

2. **Encode** each training trajectory as a token sequence:
   `[37, 142, 0, 0, 0, 891, ...]` where `0` tokens represent stall samples.

3. **Train** an autoregressive transformer to predict the next token given
   history and endpoint conditioning.

4. **At inference**, sample tokens autoregressively, decode via the codebook,
   and compose the trajectory from cumulative displacements.

This architecture has three properties that directly address the stall event
structure:

- The stall token is a **first-class output with learned probability.** The
  model can learn exactly when to emit it - after smooth deceleration, before
  heading changes.
- **Multi-sample stalls emerge naturally.** If the model learns P(stall) = p at
  a given context, then P(3 consecutive stalls) = p^3 from the autoregressive
  factorization. The distribution of stall durations (1-5 samples) is captured
  by the token-level probability, not by a separate duration model.
- The transformer's **context window sees tokens on both sides of stalls** during
  training, allowing it to learn the heading-change pattern: tokens before a
  stall cluster point in one direction; tokens after point in a slightly
  different direction.

This is the first architecture in our exploration that correctly matches the
mixed continuous-discrete structure of the data. Whether it achieves human-level
curvature and low AUC remains an empirical question - but it is the first
approach where the representational limitation is removed rather than worked
around.

### 5.6 Classifier-Free Guidance and GRPO Fine-Tuning

The VQ-VAE + transformer result (AUC 0.892) uses two techniques beyond standard
autoregressive training:

**Classifier-Free Guidance (CFG).** At inference time, each token prediction
runs two forward passes: one with full endpoint conditioning (remaining distance,
remaining displacement, remaining fraction) and one with null endpoint
conditioning (zeros). The final logits are:

```
logits = logits_uncond + scale * (logits_cond - logits_uncond)
```

with `scale = 3.0`. This amplifies the effect of endpoint conditioning, producing
trajectories that more reliably reach the target endpoint. The scale of 3.0 was
selected empirically — values below 2.0 produced insufficient endpoint accuracy,
values above 5.0 caused the model to over-commit to straight-line paths. The
implementation is in `experiments/vqvae_ar_transformer.py` (lines 183-189).

**GRPO (Group Relative Policy Optimization).** The shipped transformer checkpoint
was fine-tuned using GRPO with the adversarial RF AUC as the reward signal. This
brought the AUC from ~0.93 (base transformer) to 0.892.

*Limitation:* The GRPO fine-tuning code is not included in this repository. The
checkpoint reflects the result of this process, but the training procedure is not
reproducible from the shipped code alone. The base (pre-GRPO) transformer
training is fully reproducible via `training/train_transformer.py`.

### 5.7 Broken Feature Correlations

Beyond the curvature ceiling, continuous generative models exhibit a subtler
failure: **broken inter-feature correlations.** In human data,
`mean_acceleration` and `mean_jerk` are nearly uncorrelated (r = -0.025). This
makes physical sense - the average rate of speed change (acceleration) and the
average rate of acceleration change (jerk) measure different aspects of motor
control, and there is no reason one should predict the other.

In every continuous generative model tested - diffusion, autoregressive, and
parametric - these two features are near-perfectly correlated (r = 0.999). This
happens because smooth models produce trajectories where the acceleration
profile is itself smooth, making the jerk profile a near-perfect derivative of
the acceleration profile. The independence observed in human data arises from
the discrete stall events: stalls create jerk spikes (abrupt acceleration
changes) that are uncorrelated with the overall acceleration trend.

The Random Forest detects this joint distribution mismatch even when the
marginal distributions of each feature individually are well-matched. This is
why the adversarial classifier framework is more diagnostic than per-feature
Wasserstein distance: it captures correlation structure, not just marginals.

### 5.8 What Not to Try: Negative Results as Data

Over 145 experiments produced a substantial body of negative results. These are
valuable for narrowing the architectural search space. Approaches that have been
conclusively ruled out:

- **Tremor / OU noise as the source of direction changes.** Tremor creates
  angular velocity at velocity troughs: kappa = a/v^2 explodes. All tremor-based
  approaches produce angular velocity 2-5x too high (35+ rad/s vs. human 22).
- **Post-hoc modifications to smooth paths.** Perpendicular displacements, speed
  dips, AR(1) spatial noise, and bell-curve speed injection all fail because the
  underlying path has near-zero angular velocity (~0.2 rad/s). Even at near-zero
  speed, the curvature from these perturbations is orders of magnitude below
  human levels.
- **More trajectory points in diffusion models.** Increasing from 100 to 300
  points increases angular noise at fixed ODE step counts. AUC gets worse, not
  better.
- **Integer pixel quantization alone.** Raises direction changes via
  quantization noise, but the noise structure is wrong (uncorrelated with path
  geometry).
- **Rotation of retrieved trajectories.** Rotating real trajectories by even
  small angles inflates `num_direction_changes` by ~40% near the +/-pi boundary
  due to angle wrapping artifacts.

Each of these failures strengthens the conclusion: the curvature gap is not a
tuning problem. It is a representational problem that requires an architecture
with discrete zero-displacement tokens.

---

## 6. Perturbed Replay: Proof the Gap Is Bridgeable

### 6.1 The Experiment

To verify that the evaluation framework has enough headroom - that the gap
between AUC 0.50 and AUC 0.93 is not an artifact of overly sensitive features, 
we tested **perturbed corpus replay**: take real trajectories from the corpus,
apply small lognormal multiplicative noise to the inter-point timing, and
evaluate the result.

| Noise level (sigma) | AUC |
|---|---|
| 0.00 (pure replay) | 0.55 |
| 0.02 | **0.55** |
| 0.05 | 0.57 |
| 0.10 | 0.61 |
| 0.15 | 0.56 |
| 0.25 | 0.67 |

The sweet spot is approximately 2% lognormal noise: enough to add variety (so
the classifier cannot memorize individual trajectories from the corpus), but
small enough that the kinematic structure is preserved. Duration rescaling is
harmful - keeping natural movement duration is important.

With timing perturbation, a generative model that achieved timing quality
comparable to 2% noise on replay timing would reach AUC ~0.70.

### 6.2 What This Proves

Three things:

1. **The evaluation framework has headroom.** The gap between AUC 0.50 (replay)
   and AUC 0.93 (best generative) is real and measurable - it is not an
   artifact of overly sensitive features or a too-powerful classifier.

2. **The gap is dominated by path-timing coupling.** Pure replay preserves
   perfect coupling (path and timing come from the same trajectory). Perturbed
   replay partially preserves it. Independent generation of path and timing
   destroys it entirely (AUC 1.00).

3. **The target is achievable.** AUC 0.70 with 2% noise proves that a model
   producing trajectories with "slightly imperfect but structurally correct"
   timing can reach the useful range. The question is whether a fully generative
   model can learn to produce timing that is structurally correct without seeing
   real trajectories at inference time.

### 6.3 The Research Question, Refined

The results narrow the research question to:

> Can a fully generative model learn the path-timing coupling and discrete stall
> structure well enough to match perturbed replay (AUC ~0.70) without a
> trajectory corpus at inference time?

The VQ-VAE + autoregressive transformer is the first architecture that
addresses both requirements: discrete stall tokens for zero-displacement events,
and autoregressive factorization for path-timing coupling (since each token
encodes a displacement that implicitly couples path geometry and speed).

---

## 7. Related Work

### 7.1 Kinematic Theory and the Sigma-Lognormal Model

Plamondon's kinematic theory of rapid aimed movements (Plamondon, 1995)
models handwriting and mouse trajectories as sums of lognormal velocity
profiles. The sigma-lognormal model decomposes each movement into overlapping
submovements, each characterized by a lognormal speed profile with parameters
(D, t0, mu, sigma). This is the standard parametric model in the handwriting
recognition and HCI literature.

Our findings extend this work in two ways. First, the extreme velocity peaks in
mouse data (coefficient of variation ~34x, meaning peak speed routinely exceeds
30x the mean) are far larger than those observed in handwriting. The
sigma-lognormal model, with its smooth analytic profiles, cannot reproduce these
peaks - our experiments with sigma-lognormal generation produced max velocity
22x too low. Second, the submovement composition mechanism matters: the
classical additive model (summing lognormal profiles) produces velocity stacking
that is qualitatively wrong. Our data suggest a competitive (winner-takes-all)
composition, consistent with more recent work on intermittent motor control.

### 7.2 Fitts' Law

Fitts' law (Fitts, 1954) predicts movement duration as a logarithmic function
of the index of difficulty (distance / target width). It is the foundational
model for aimed movement in HCI.

Fitts' law explains **when** movements end but not **how** they get there.
It predicts duration and implicitly constrains the velocity profile, but it does
not model trajectory shape, curvature, stall events, or angular dynamics. Our
feature set includes movement duration (which Fitts' law addresses) and 17 other
features (which it does not). A generator based on Fitts' law alone can match
duration but fails on all other features.

### 7.3 Discrete Tokens for Continuous Generation

The VQ-VAE + autoregressive transformer architecture follows a pattern
established in several adjacent domains:

- **VALL-E** (Wang et al., 2023) - text-to-speech via discrete neural codec
  tokens. Speech, like mouse trajectories, is a mixed continuous-discrete signal
  (continuous formant transitions + discrete phonetic events).
- **MusicLM** (Agostinelli et al., 2023) - music generation via hierarchical
  discrete tokens. Music contains both continuous dynamics (volume, pitch bend)
  and discrete events (note onsets, rests).
- **T2M-GPT** (Zhang et al., 2023) - human body motion generation via VQ-VAE
  tokens. Human body motion has the same mixed structure: continuous limb
  trajectories punctuated by discrete events (foot contacts, direction changes).

The common pattern: when the target signal contains discrete events embedded in
continuous dynamics, quantizing the signal into discrete tokens and modeling
token sequences autoregressively outperforms direct continuous generation. Our
analysis suggests mouse trajectories belong to this category.

### 7.4 Intermittent Motor Control

The traditional view of aimed movement as a single smooth trajectory has been
challenged by intermittent control theory (Loram et al., 2011), which proposes
that motor commands are issued in discrete bursts rather than continuously
updated. Under this view, the stall events we observe - discrete
zero-displacement intervals with heading changes - are direct evidence of
control intermittency: the motor system issues a command, the hand moves
ballistically, the command expires, the hand stops, a new command is issued in
a (possibly different) direction.

Our finding that 6.14% of time steps have exactly zero displacement, with stall
durations of 1-5 consecutive samples (8-40ms), is consistent with the
intermittent control timescales reported in the postural control literature.
The heading change at stall boundaries (5-30 degrees) suggests that each new
motor command includes a directional correction based on visual feedback of the
cursor-target error.

### 7.5 Submovement Composition

The classical model of aimed movement (Meyer et al., 1988) proposes that
movements consist of an initial ballistic submovement followed by one or more
corrective submovements, composed additively. Our experiments directly tested
this additive composition assumption and found it produces qualitatively wrong
velocity distributions for mouse trajectories.

Training data analysis of 20,000 trajectories revealed a mean of 6.7 velocity
peaks per trajectory - substantially more than the 2-4 submovements predicted
by the classical model for simple aimed movements. Trough speed between peaks
averages 35% of peak speed rather than near-zero, indicating heavy overlap
between submovements. The first submovement is consistently the largest
(displacement fraction std = 0.73), with each subsequent submovement smaller and
shorter.

The data instead support a competitive composition model where, at any given
moment, one submovement dominates rather than all active submovements summing.
This is consistent with the "winner-takes-all" dynamics observed in motor
cortex neural recordings, where multiple motor plans compete and the winning
plan suppresses alternatives rather than blending with them.

The practical implication: any parametric model that composes submovements
additively will produce mean velocities 5-10x higher than human data (960 px/s).
Sequential (non-overlapping) composition avoids velocity stacking but produces
individually smooth segments with no stall events, yielding near-zero curvature.
Neither additive nor sequential composition can produce the mixed
continuous-discrete structure observed in the data.

---

## Summary of Key Numbers

| Quantity | Value | Source |
|---|---|---|
| Human trajectory corpus size | 4.16M trajectories | 5 public HCI datasets |
| Evaluation features | 18 kinematic features | See Section 2 |
| Corpus replay AUC | 0.498 | Theoretical floor |
| Best fully generative AUC | 0.892 | VQ-VAE + GRPO transformer + CFG |
| Perturbed replay AUC | 0.55 (2% noise) | Section 6 |
| Generative target | < 0.60 | Near-indistinguishable |
| Top RF feature importance | 10.8% (angular_velocity_std) | Distributed importance |
| Top-5 RF feature importance | 41% | No single feature dominates |
| Human mean velocity | ~960 px/s | Corpus statistics |
| Human max velocity CV | ~34x | Extreme peaks |
| Human curvature mean | ~1329 | Dominated by stall events |
| Best generative curvature | ~7-15 | 100-200x gap |
| Zero-displacement steps | 6.14% of all steps | Discrete stall events |
| Stall duration | 1-5 samples (8-40ms) | Fixed USB polling intervals |
| Timing residual autocorrelation | r = 0.65 | Motor control smoothness |
| Peak velocity location | ~35% of duration | Universal, distance-independent |
| Experiments conducted | 145+ | See EXPERIMENTS.md |
| Model architectures tested | 8 families | See Section 5 |

---

## Notation and Definitions

| Symbol | Definition |
|---|---|
| AUC | Area under the receiver operating characteristic curve. Computed on OOB predictions of a Random Forest classifier. Range [0, 1]; 0.50 = random; lower is better for the generator. |
| OOB | Out-of-bag. Each Random Forest tree is trained on a bootstrap sample. OOB samples are the ~37% not in that bootstrap. OOB predictions aggregate only trees that did not train on each sample. |
| kappa | Curvature. Computed as \|v x a\| / \|v\|^3 where v is velocity and a is acceleration. Units: 1/px. |
| dx, dy | Displacement between consecutive samples. In pixels. |
| dt | Time between consecutive samples. Typically 8ms (125 Hz USB polling). |
| px/s | Pixels per second. Speed unit. |
| CV | Coefficient of variation: standard deviation / mean. Dimensionless. |
| eta | DDPM stochastic sampling parameter. eta = 0 gives deterministic (DDIM) sampling; eta = 1 gives full DDPM stochastic sampling. |
| VQ-VAE | Vector-Quantized Variational Autoencoder. Learns a discrete codebook of motion tokens from continuous displacement data. |
| CFM | Conditional Flow Matching. Learns a velocity field that transports noise to data via an ODE. |
| DDPM | Denoising Diffusion Probabilistic Model. Learns to reverse a noise-adding process via iterative denoising. |
| NLL | Negative log-likelihood. Training loss for probabilistic models. |
| RF | Random Forest. Ensemble of decision trees trained on bootstrap samples. |

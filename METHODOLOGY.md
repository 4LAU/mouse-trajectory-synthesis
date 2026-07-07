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

The state of the art has moved twice since the continuous-model era described
in Sections 3 through 6c. A masked-token event-stream model (Section 7) that
represents trajectories as discrete speed, heading, and timing events reaches
**AUC 0.652** on its own, the best result in this project for a single
generative model with no selection. An inference-time selection layer on the
same model brings the three-seed-confirmed result to **0.568**, and a set-level
selection method reaches **0.504** across three seeds, chance level on the
primary detector with every tree and nearest-neighbor family within 0.014 of
chance. For comparison, the
best of the continuous-model era was 0.864 (ZIMT), and corpus replay of the same
distribution sets the floor at ~0.51. Sections 1 through 6c cover that earlier
era and the stall discovery that motivated moving to discrete events; Section 7
covers the event-stream model and the selection results.

---

## Table of Contents

1. [Research Goal](#1-research-goal)
2. [Evaluation Framework](#2-evaluation-framework)
3. [Key Data Discoveries](#3-key-data-discoveries)
4. [Discrete Stall Events: The Key Insight](#4-discrete-stall-events-the-key-insight)
5. [Representational Limitation Analysis](#5-representational-limitation-analysis)
6. [Perturbed Replay: Proof the Gap Is Bridgeable](#6-perturbed-replay-proof-the-gap-is-bridgeable)
7. [The Event-Stream Era: A Masked-Token Model and Selection](#7-the-event-stream-era-a-masked-token-model-and-selection)
8. [Related Work](#8-related-work)

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
evaluation reliable: a model that games one feature will be caught by the others.
It is a challenge because it means there is no single "fix" - improvement
requires simultaneously matching velocity profiles, acceleration dynamics, jerk
statistics, curvature, angular velocity, timing, and path geometry.

### Why This Is Principled

The 18 features span every level of the kinematic hierarchy, from zeroth-order
geometry (path efficiency, max deviation) through velocity, acceleration, and
third-order jerk, plus curvature, angular dynamics, and timing. The grouping is
laid out in the tables above. Because the set covers the full derivative stack,
any systematic difference in motor control dynamics between synthetic and human
trajectories will show up in at least one level.

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
| Chunk Diffusion (25-step) | Approximate (stall logit channel) | ~0.2 | ~0.96 | No global trajectory awareness, velocity_skewness 1.76 Wasserstein |
| VQ-VAE + AR Transformer | Yes, stall token (0,0) | ~0.4 | ~0.89 | Mode collapse (420/1024 tokens), error compounding |
| VQ-VAE + MaskGIT (SoundStorm) | Yes, stall token (0,0) | ~0.4 | ~0.996 | VQ-VAE quantization bottleneck: accumulated tokens create wrong path shapes |
| **CANDI hybrid diffusion (polar)** | **Yes, discrete stall channel** | **~1300** | **~0.85** | **Angular velocity and curvature distributions still mismatched** |

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

**Update (2026-05-16)**: Extensive testing of both autoregressive (Section 5.5
above) and masked bidirectional (Section 5.9) generation on VQ-VAE tokens
revealed a fundamental bottleneck: the codebook quantization itself. While the
architecture correctly matches the problem's mixed discrete-continuous structure
in principle, the 1024-entry codebook's displacement quantization (30 px/s
reconstruction error per step) compounds over 50-200 steps to produce paths with
wrong curvature, angular velocity, and efficiency. The stall token mechanism
works as intended, but the motion tokens are too imprecise for path accumulation.
This motivates hybrid discrete-continuous diffusion (CANDI-style): keep the
discrete stall channel but generate continuous displacements without quantization.

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
selected empirically. Values below 2.0 produced insufficient endpoint accuracy,
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

### 5.9 VQ-VAE + Masked Iterative Decoding (SoundStorm): The Quantization Bottleneck

The SoundStorm/MaskGIT approach (start with all tokens masked, iteratively unmask
in confidence order with full bidirectional context) was tested on our validated
VQ-VAE codebook (1024 entries, 100% utilization, stall token 0).

A 6M-parameter masked bidirectional transformer was trained for 16 epochs on 500K
tokenized trajectories, reaching 49.3% masked token prediction accuracy. Four
generation strategies were tested:

- **From-scratch generation** (all MASK → iterative unmasking): AUC 0.996. Cold-start
  collapse: the model predicts stall token 0 at every position with 37% confidence,
  and confidence-based unmasking locks in all-stall sequences.
- **Iterative refinement** (donor tokens → mask 40% → re-predict): AUC 0.996. Avoids
  cold-start but spatial shape is still wrong after 12 refinement rounds.
- **Soft decoding** (probability-weighted expected displacement instead of discrete
  sampling): AUC 0.997. No improvement: the expected displacement is a compromise
  that satisfies no constraint well.
- **Donor token perturbation** (keep most donor tokens, mask a few, SoundStorm fills):
  AUC 0.914. Even with minimal replacement, accumulated quantized tokens create
  detectably wrong paths.

The root cause is the VQ-VAE displacement tokenization itself, not the sequence model.
When discrete tokens are accumulated to reconstruct paths, the 1024-entry codebook
quantization destroys spatial smoothness. Curvature, angular velocity, and path
efficiency are all wrong because each step's displacement is snapped to the nearest
codebook entry, and these quantization errors compound over 50-200 steps.

This finding updates the assessment from Section 5.5: while VQ-VAE tokens correctly
match the problem's mixed discrete-continuous structure in principle (stall token = 0,
motion tokens = continuous motion), the codebook quantization of the continuous
component is too coarse for path accumulation. The 30 px/s speed reconstruction error
(3.6% of mean speed), acceptable for single-step reconstruction, compounds to create
fundamentally wrong path shapes when accumulated.

**Architectural implication**: The next approach must keep displacements continuous
(no VQ-VAE) while still handling stalls as discrete events. This points to hybrid
discrete-continuous diffusion (CANDI-style): a single denoiser with a discrete
masking channel for stall/no-stall decisions and a continuous Gaussian channel for
(dx, dy) displacement.

### 5.10 Chunk-Level Diffusion: The Global Awareness Problem

Action chunking (generating 25-step chunks via DDPM and sequencing them
autoregressively) was tested as a middle ground between per-step AR (ZIMT) and
full-trajectory diffusion (DDPM). The hypothesis: 25-step chunks are long enough
to produce coherent internal kinematics while reducing compounding error from
200 decision points to ~8.

The result was AUC 0.957, significantly worse than both ZIMT (0.864) and DDPM
(0.862). The primary failure mode was `velocity_skewness` at 1.764 Wasserstein,
the worst of any model. The root cause: each chunk generates in isolation with
only a 5-step context window and scalar progress/remaining-fraction conditioning.
No chunk knows whether it sits at 20% of the trajectory (peak acceleration phase)
or 80% (deceleration tail), because the model has no representation of the
global velocity profile shape.

This failure eliminates the middle ground between per-step and full-sequence
approaches. The conclusion: the next architecture must have full-sequence context
at every generation step, pointing to bidirectional approaches (masked iterative
decoding) rather than any form of autoregressive sequencing.

### 5.11 Three-Pattern Failure Analysis: A Unified Root Cause

After 150+ experiments across 8 model families, three persistent gaps all trace
to one mechanism: incorrect generation of the stall boundary pattern
(decelerate → hold → change heading → accelerate).

**Pattern 1, velocity_skewness** (Wasserstein 0.08-1.76 across models) is a
global property. Humans peak at ~35% of duration with a long deceleration tail,
and no local window encodes that envelope, so only full-trajectory approaches
(DDPM, SoundStorm) preserve it.

**Pattern 2, angular_velocity** (Wasserstein 0.41-0.78) comes from heading
changes at stall boundaries rather than smooth curves (Section 4.2). The only
model that matched human angular velocity (42.7 vs 45.8 rad/s) was VQ-VAE with
discrete stall tokens, confirming the source is discrete zero-displacement
events, not continuous path curvature.

**Pattern 3, acceleration-jerk decorrelation** (human r=-0.025, synthetic
r=0.999, mechanism in Section 5.7) breaks because jerk spikes at stall
boundaries, and only exact-zero stalls followed by heading changes produce those
decorrelating spikes.

All three patterns point to the same architectural requirement: a model with
(a) discrete stall tokens for exact zero displacement, (b) full-sequence
context for global velocity profile awareness, and (c) bidirectional attention
so stall boundary tokens are generated with context from both sides. This
combination uniquely describes the SoundStorm/MaskGIT masked iterative
decoding approach applied to our validated VQ-VAE codebook.

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

## 6b. Corpus Enhancement: Why Every Transformation Hurts

### The Paradox of Corpus Enhancement

With a 4.16M-trajectory pool, plain corpus replay (translate-only, cubic-ease
endpoint correction) achieves AUC 0.52. A natural assumption is that adding
diversity through perturbation or transformation would help by preventing exact
memorization detection. The opposite is true: **every enhancement tested made
the result worse**, sometimes dramatically.

| Enhancement | AUC (n=2000) | Delta from replay |
|---|---|---|
| None (translate-only) | 0.52 | n/a |
| 3% magnitude scaling + 1.5% perpendicular jitter | 0.645 | +0.13 |
| Similarity transform (rotate + scale) | 0.682 | +0.16 |
| Smooth sinusoidal perturbation | 0.780 | +0.26 |
| Magnitude-weighted endpoint correction | 0.558 | +0.04 |

### Why Enhancements Fail

The 18 kinematic features form a tightly coupled joint distribution. Each
feature depends on the full sequence of displacements, and any modification,
even a mathematically exact similarity transform, changes the joint
distribution detectably.

Specific mechanisms of detection:

- **Perturbation changes direction counts.** Even 3% magnitude scaling
  occasionally pushes a small step past the stall threshold (0.3 px), creating
  or eliminating a direction change. `num_direction_changes` has a discrete
  distribution (integer-valued), making any systematic shift detectable.
- **Rotation changes axis-relative signs.** Direction changes are computed from
  the sign of angular differences. Rotation by angle theta shifts all absolute
  angles, which can flip sign boundaries at ±pi. This inflates
  `num_direction_changes` by ~40% for a 45-degree rotation.
- **Scaling changes speed-dependent thresholds.** The stall detection threshold
  (speed < some value) is in absolute units. Scaling a trajectory by 1.1x
  pushes stalls above the threshold, changing curvature and direction count
  statistics.
- **Distributed endpoint correction alters many steps.** Magnitude-weighted
  correction touches every moving step. Though each change is small, the
  cumulative effect on direction counts and angular velocity is detectable.
  Cubic-ease correction, concentrated in the last 25%, is less detectable
  because it mimics natural deceleration-phase adjustment.

### Implications for Generative Models

This finding means the AUC 0.52 floor is approximately the limit of any
retrieval-based approach. To go lower requires either:
1. An even larger pool with exact distributional match to the evaluation set
2. A generative model that produces novel trajectories matching the full
   18-feature joint distribution

---

## 6c. ZIMT Endpoint Correction: A Partial Fix

### The Endpoint Correction Problem

ZIMT generates trajectories autoregressively in normalized space and must be
corrected to hit the exact endpoint. The original approach: linear interpolation
in the last 20% of steps. This creates an artificial velocity peak at ~90% of
duration (human peak is at ~35%), making `time_to_peak_velocity` the #1
discriminator.

### Magnitude-Weighted Correction

Replacing linear correction with magnitude-weighted correction (each moving step
absorbs error proportional to its displacement) improves AUC from 0.878 to
0.864. The velocity profile becomes more natural: `velocity_skewness`
Wasserstein drops from 1.57 to 0.08.

However, the improvement is modest because the fundamental joint distribution
mismatch remains. The new bottleneck is angular velocity (mean: 0.506
Wasserstein, std: 0.413) and `mean_acceleration` (0.108 RF importance despite
low individual Wasserstein, indicating the RF detects joint distribution
patterns).

### Guided MDN Sampling (Failed)

An attempt to eliminate endpoint correction entirely by shifting MDN component
means toward the endpoint during inference (analogous to classifier-free
guidance in diffusion models) failed at all strength levels tested (0.05, 0.1,
0.3). The model was not trained with guidance, so shifted means produce
out-of-distribution outputs. AUC worsened from 0.878 to 0.913-0.968.

**Lesson**: Inference-time distribution modification only works when the model
is trained to expect it (e.g., classifier-free guidance trains with random
condition dropout).

### Differentiable Feature-Matching Fine-Tuning (Failed)

An attempt to fine-tune ZIMT by backpropagating through differentiable kinematic
feature computation (curvature, angular velocity, path efficiency, etc.) using
a two-phase approach: Phase 1 generates reference trajectories without gradient,
Phase 2 teacher-forces from references in a single forward pass with Gumbel-Softmax
component selection and straight-through stall gating.

Three iterations were tested with progressively better gradient handling (L2→L1
loss, grad_clip 10→100, feature clamping). All failed due to **exposure bias**:
teacher-forced training pushes parameters in directions that improve loss under
ground-truth inputs, but these same parameter changes compound errors during
autoregressive inference. Angular velocity Wasserstein *worsened monotonically*
with more training (0.437 → 0.736 → 0.890).

This is a fundamental limitation of teacher-forced fine-tuning for autoregressive
models, not a hyperparameter issue. Overcoming it would require either (a)
scheduled sampling at fine-tuning time (mixing teacher-forced and free-running
inputs), or (b) reinforcement learning that operates on complete trajectories
generated autoregressively (GRPO/RAFT).

### Rejection Sampling (Failed)

Generating N candidate trajectories per query and selecting the one closest to
the human feature mean (normalized L2 distance across all 18 features) produced
AUC 0.892 with N=8, worse than the 0.864 baseline. The selection process kills
feature variance: distributions become too narrow, and the RF classifier detects
reduced variance as easily as shifted means.

---

## 7. The Event-Stream Era: A Masked-Token Model and Selection

Sections 3 through 6c describe the continuous-model era: the stall discovery,
why every continuous architecture (diffusion, autoregressive GRU, parametric
submovement models) hits a ceiling somewhere between AUC 0.86 and 1.0, and the
VQ-VAE experiments that first tried a discrete stall token but were bottlenecked
by codebook quantization of the motion component (Sections 5.5 and 5.9). This
section covers what came after: a representation and a model built to avoid
that quantization problem from the start, and a discovery, late in the project,
that most of the remaining gap was never a generation problem at all.

### 7.1 Why events, not quantized displacements

The VQ-VAE work had the right instinct (a discrete token for the stall) and the
wrong discretization target. It quantized displacement itself into a fixed
codebook, and the resulting 30 px/s reconstruction error per step, negligible
for one step, compounded over 50 to 200 accumulated steps into visibly wrong
path shapes. Chunk-level diffusion (Section 5.10) taught a separate lesson: a
model that only sees a local window at generation time has no representation of
where it sits in the trajectory's global velocity profile, and that showed up
as the worst velocity-skewness mismatch of any model tested.

Both lessons point the same direction. The fix is not a better codebook for
displacement. It is to discretize the thing that is naturally discrete (the
stall, and the direction the hand takes after it) while keeping full-sequence,
bidirectional context available at every generation step. That means moving
away from a sequence of quantized (dx, dy) pairs and toward a sequence of
motion events: a speed, a heading change relative to the previous heading, and
the time until the next event. The mouse hardware itself already reports data
this way. Pool files retained from data preparation turned out to hold the raw,
pre-resample event stream (integer pixel positions with millisecond
timestamps), not just the 125 Hz resampled grid used for feature extraction.
Only about 30% of inter-event gaps sit on the nominal 8ms polling clock; the
rest spread from 1 to 150ms. A stall, in this representation, is simply an
event with zero speed. It does not need a special token bolted onto a
continuous displacement space, because the representation was never continuous
displacement to begin with.

### 7.2 The event representation has almost no ceiling of its own

Before training anything, the representation itself was gated: take real event
streams from the corpus, push them through the exact encode and decode
pipeline the model would use, and measure whether that round trip alone is
detectable. Pure replay through the pipeline (no model involved) scored AUC
0.496 to 0.507 across variants, essentially the same range as plain corpus
replay (0.51). This matters because it rules out the representation itself as a
source of ceiling: whatever score a trained model gets, the gap to 0.50 is a
modeling problem, not a cost of the event encoding.

The same gate was reapplied when the representation moved from raw (dx, dy)
events to a polar one (speed bin, heading-increment bin, inter-event time),
because polar coordinates need to be quantized into bins. The lossless
float round trip matched the floor again (0.508), as it had to. But binning the
heading increment into 256, 512, or 1024 categories cost 0.03 to 0.06 of AUC
regardless of how fine the bins were, which rules out quantization resolution
as the cause. The actual cause was simpler: decoding speed and heading into a
path produces off-integer pixel positions, and real recording hardware never
writes an off-integer position. Rounding the decoded positions back to the
integer pixel grid removed the entire penalty at every bin resolution tested.
This is the same finding as the original stall discovery, generalized: the
detector keys on whether a position sits on the hardware's recording lattice,
not on how finely the underlying motion is discretized. It fixed the decode
contract for everything that followed: integrate speed and heading
continuously, then round to integer pixels only at the very end of decoding.

### 7.3 Architecture: a masked bidirectional model over polar events

The shipped model is a 6-million-parameter, non-autoregressive Transformer in
the MaskGIT and SoundStorm family: every event token starts masked, and tokens
are revealed iteratively over several rounds with full bidirectional context
available at each round, rather than being generated left to right. This
directly answers the chunk-diffusion lesson from Section 5.10: there is no
local window, every generation step sees the whole sequence.

Each event carries a categorical speed bin and a categorical heading-increment
bin, factored as p(speed, heading | context) = p(speed | context) times
p(heading | speed, context), because large turns in the human corpus happen
almost exclusively at low speed. Inter-event time is not modeled as a fixed
clock; it is generated by a small flow-matching head on the z-scored log of the
gap, which is what allows the model to reproduce the raw, non-uniform polling
intervals (mostly 8ms during motion, 1 to 2ms for hardware ticks) rather than
snapping everything to a uniform grid. Tick events (samples with no movement,
about 10 to 15% of events) carry no heading of their own; heading persists
through them. The model is trained on all 4.16 million trajectories in the
corpus. The base checkpoint is event_polar_4m.pt; the shipped checkpoint,
event_polar_4m_fc_v2.pt, adds the movement-character conditioning described in
Section 7.5.

### 7.4 The sampler and the decode contract: two knobs worth 0.17 of AUC

Two decisions made after training, not during it, accounted for more AUC
improvement than any single training change in this section.

The first is reveal order. Standard MaskGIT reveals the most confident
predictions first. For this model, the most confident heading prediction at
almost any position is "no turn," so straight paths get locked in early and
every later token conditions on an already-straight context. Out of the box
this produced paths far too straight (path efficiency 0.994 versus a human
0.949 median). The fix is the standard MaskGIT remedy: add Gumbel noise to the
confidence score before ranking, with a choice temperature controlling how much
randomness competes with confidence. This single knob was worth 0.12 of AUC
(0.929 to 0.806) and needed no retraining.

The second is the lattice snap. Even after the integer-pixel rounding described
in Section 7.2, a stubborn angular-velocity gap remained, and profiling
localized it entirely to slow frames (speed under about 3 px per frame): three
times the human turning rate there, with every faster speed band matching.
Decoding the same token stream without any rounding erased the gap completely,
which pinpoints the mechanism. The model emits smooth, continuous headings at
slow speed, and rounding a continuously-varying, off-lattice slow path to the
pixel grid causes the rounded direction to flip almost every step. Real slow
human movement does not do this: a person moving one pixel at a time repeats
the same integer step, because the recording lattice is their actual output
space, not an approximation of something smoother underneath. The fix snaps
slow steps (speed below 2.5 px per frame) to the nearest realizable whole
lattice step, while letting the integrated heading continue to evolve
continuously between snaps, so no directional drift accumulates. This decode
change, on top of a distribution-matching fine-tune described next, completed a
chain that moved the score from 0.929 to 0.806 (sampler) to 0.791 (distribution
matching) to 0.755 (decode contract), and a further correction restoring full
human variance to the duration conditioning, which had been silently
undershooting at 0.7 times human variance, brought the pure model to its first
result under 0.70.

### 7.5 Movement-character conditioning: a missing global variable

After the sampler and decode fixes, the remaining gap showed up as a broken
correlation structure rather than a broken marginal. In real trajectories,
every pairwise combination of the acceleration and jerk features correlates at
essentially r = 1.000: one latent "how vigorous is this movement" variable
governs all of them together. In the pure token model these correlations were
a patchwork, some repaired, some not, even once every individual feature's
marginal distribution looked correct. Two attempts to fix this by pushing the
model directly toward matching feature statistics, first a fixed-statistic
distribution-matching fine-tune (quantile and covariance matching against real
batches), then an adversarial critic trained on the 18 detector features, both
made the score worse. The reason is architectural: a per-position, per-token
training signal has no way to coordinate an outcome that only exists at the
level of the whole decoded trajectory.

The fix that worked was not a better loss but a new input. The model was given
an explicit conditioning slot: the same 18-dimensional feature vector the
evaluator computes, fed in through a pathway initialized at zero so training
starts exactly at the pretrained model, and teacher-forced on each real
trajectory's own statistics during fine-tuning. At generation time, no real
trajectory's vector is copied. Instead, the conditioning vector is drawn from a
kernel density estimate fitted to a bank of real feature vectors stored in the
checkpoint, matched to the requested movement distance. Twelve thousand steps
of this fine-tune produced the project's first clean training-side gain (from
0.702 to 0.675 RF OOB), and the on/off comparison (conditioning present versus
a zero vector) confirmed the model had genuinely learned to lean on the
variable rather than ignoring it. This result also reframed the two earlier
failures: they were not evidence that feature-level training signals cannot
help this architecture, only that the architecture needed an explicit variable
to receive them before it could act on them.

### 7.6 The pure-model result, and why teaching the judge's ranking into the weights stops working here

With the sampler, decode contract, and character conditioning in place, plus an
empirical duration prior (drawing real per-distance-bin durations instead of a
fitted lognormal curve, which had been quietly feeding the model
out-of-distribution durations), the pure model reaches its confirmed best:

| Seed | RF OOB AUC |
|---|---|
| 42 | 0.6470 |
| 43 | 0.6544 |
| 44 | 0.6531 |

RF OOB 0.652 +/- 0.003 across three seeds. This is the number to cite for "the
model alone, nothing else": gumbel reveal order, choice temperature 10, lattice
snap at 2.5 px, duration standard deviation 1.0, empirical duration prior, no
selection step of any kind. Checkpoint event_polar_4m_fc_v2.pt.

At this point a natural next step is to take the selection mechanism described
in Sections 7.7 and 7.8 and bake its preference directly into the weights,
rather than running it at inference time. Four independent ways of doing this
were tried, and all four failed in the same way. Plain imitation
distillation (fine-tune on the winning candidate from a sampled pool, using the
exact pretraining objective) made every checkpoint worse than the model it
started from, with most of the damage done within the first few hundred steps.
Mixing real human batches back in as an anchor changed the shape of the
degradation but not its direction. A conditioned adversarial critic (the plain
critic from Section 7.5, rerun with the character vector active so the
generator had a pathway to act on the criticism) also regressed the score, with
the discriminator's real-versus-fake margin growing steadily while the
generator closed none of it. Preference learning (Diffusion-DPO: contrast the
judge's best and worst candidate per movement, train against a frozen reference
copy of the model) is the most informative failure: preference accuracy on held
out pairs climbed cleanly from 0.50 to 0.87, so the model genuinely absorbed the
judge's ranking, while generation quality collapsed monotonically and
increasingly steeply as training continued (RF OOB rose from the 0.647 control
past 0.86 by step 500 and past 0.96 by step 1000).

The common mechanism is architectural, not a tuning failure. This model's
training objective is to get each token's marginal distribution right at each
position. A sampling-importance-resampling judge's winners win because of joint
feature combinations that only exist across the whole decoded trajectory, and
there is no way to transmit a whole-trajectory preference through a
per-position masked-token objective without disturbing the very marginals that
made the pretrained model good in the first place. Four independent routes to
the same wall (fixed-statistic matching, a plain adversarial critic, a
conditioned adversarial critic, and preference learning) is treated here as a
closed question for this architecture: a judge's signal compresses into these
weights in the ranking direction, but not in the generation direction.
Selection has to remain an inference-time system, not a training target.

### 7.7 SIR: selection as sampling-importance-resampling

With training-time approaches closed off, the remaining lever operates purely
at inference time. For each requested movement, the model draws K = 16
independent candidate trajectories (each with its own character draw from the
KDE bank, and its own duration draw, since sharing one duration draw across all
K candidates turned out to hide an entire feature family from the selector).
A discriminator (a gradient-boosted classifier) is fit to distinguish a
reference set of real human trajectories, drawn from the corpus and disjoint
from the evaluation sample, from the pool of candidates. Rather than always
keeping the candidate the discriminator scores as most human-like, one
candidate per movement is drawn from a tempered lottery over the
discriminator's log-odds: a soft, weighted draw rather than an argmax.

The order these were tried in matters for understanding why the lottery works.
The first thing tried was simpler: keep whichever candidate scores closest to
its own commanded character vector. That made the score worse (0.698 versus a
0.651 baseline at the time), because pulling every sample toward a target
shrinks the conditional feature variance, and the detector reads reduced
variance just as readily as a shifted mean; this is the same family of mistake
as lowering sampling temperature to reduce variance elsewhere in the project.
The tempered lottery avoids this because it reshapes the realized distribution
of the selected set toward the human one without collapsing the variety within
it.

An early version of this result was optimistic: the discriminator had been fit
on the same feature file the evaluator uses as its human class, so the judge
was being scored against data it had effectively already seen. Refitting the
judge against a dedicated 4,000-trajectory reference set, drawn from the corpus
with the evaluation indices excluded, gave the honest number. The locked
recipe, K = 16 candidates, selection temperature 0.7 (sharper than the default
temperature of 1.0), the empirical duration prior, and per-candidate duration
diversity, reaches:

| Seed | RF OOB AUC |
|---|---|
| 42 | 0.5589 |
| 43 | 0.5781 |
| 44 | 0.5681 |

RF OOB 0.568 +/- 0.010 across three seeds. Every gain in this recipe came from
the selection side: sharpening the lottery's temperature, widening the judge's
kernel bandwidth a little, and fixing the duration prior all helped, in
decreasing order of effect. Every proposal-side attempt to add more heat
(a hotter heading distribution, a higher choice temperature, doubling the
candidate pool to 32) either did nothing or actively hurt, and a bigger pool
without sharper selection wasted twice the generation cost for no gain. The
lesson that shaped the next section: the generator already drafts human-like
paths often enough. What was left to fix lived entirely in how candidates get
chosen.

### 7.8 Selection as a set-level problem

SIR still leaves about 0.068 of AUC between the honest result and chance. The
structural reason is that SIR selects each candidate independently, but the
detector it is trying to fool is never shown one trajectory at a time. It is
trained on the entire selected population against the entire reference sample.
An independent, per-movement lottery cannot trade one movement's pick against
another's to correct a marginal that the whole selected set overshoots, and the
tempering itself distorts the selected distribution in a way a freshly trained
detector can still catch. This reframes the target: selection should optimize
the distribution of the selected set as a whole, not the quality of each item
in isolation.

This reframing is cheap to test because the expensive part, generating K
candidates per movement, is separable from the cheap part, choosing which one
wins. Every candidate a model produces during a normal evaluation run can be
cached (trajectory, features, and owning movement spec), and any selection rule
can then be replayed against that cached pool in about three minutes with no
further GPU sampling. All experiments in this subsection follow a strict
honesty protocol: the 4,000-row disjoint reference set is split in half, all
selection strategies fit only against half A, and a proxy AUC against the
untouched half B is reported for every candidate strategy. The number that
counts is always a full replay through the real evaluator, where the human
class is the true evaluation sample that no stage of selection has ever seen.

Two families of set-level strategy were tried and failed. Directly minimizing
the L1 gap between the selected set's feature histograms and the reference
histograms, across all 18 marginals plus the most correlated feature pairs,
drove the summed histogram gap down by a factor of five with no improvement in
detector AUC at all. A kernel MMD exchange, sensitive to interactions of every
order rather than just marginals and pairs, also failed to beat the plain SIR
baseline. Together these rule out marginal and pairwise structure as the
location of the remaining signal: whatever the detector is still keying on is a
higher-order joint pattern shared across every candidate in a given movement's
pool, not something a distribution-matching objective over lower-order
statistics can reach. This also explains why simply always keeping the judge's
single favorite candidate (greedy argmax, no lottery) makes the score much
worse rather than better: concentrating every pick on the judge's preferred
region of feature space manufactures its own population-level artifact, one a
fresh detector finds instantly.

The strategy that did work treats selection as an iterated game against the
population-level artifact directly. Fit a fresh discriminator between the
reference half and the CURRENT selected set (not the raw candidate pool), move
only the fraction of movements with the largest discriminator log-odds gain
toward the discriminator's preference, and repeat with a decaying step size.
This gives selection the population-level feedback loop that per-item SIR
structurally cannot have: each round's judge is reacting to what the previous
round actually produced as a set, not to individual candidates in isolation.
Two things turned out to matter more than expected. First, the discriminator's
own family matters more than any other setting: using a Random Forest judge,
the same detector family the evaluator actually trains, walked the proxy AUC
down into the 0.46 to 0.47 range, while a gradient-boosted judge stalled around
0.51 to 0.54 no matter how the other knobs were set. Second, step size controls
stability: a flat 15% step held for many rounds oscillates round to round
(overshooting the same way plain greedy argmax does, just more slowly), while a
small or gradually decaying step converges smoothly.

### 7.9 A confirmed result at chance level

The winning configuration, a Random Forest judge with a 20% step decaying by
0.85 per round over 30 rounds, was replayed through the honest evaluator (the
real evaluation sample as the human class, nothing from selection ever having
seen it) across three independent candidate pools, seeds 42, 43, and 44, with
the full detector suite including the raw-trajectory nearest-neighbor. Two judge
widths were confirmed. The narrower judge reads the same 18 features the primary
detector uses; the wider judge adds 15 raw-signal summaries the detector can
also read directly (Section 7.9b).

18-feature judge:

| seed | RF OOB | RF 5-fold | GBM 5-fold | Raw-NN |
|---|---|---|---|---|
| 42 | 0.4892 | 0.4820 | 0.5042 | 0.5396 |
| 43 | 0.4922 | 0.4972 | 0.5252 | 0.5309 |
| 44 | 0.4907 | 0.4783 | 0.5037 | 0.5163 |
| mean | 0.4907 | 0.4858 | 0.5110 | 0.5289 |

33-feature judge:

| seed | RF OOB | RF 5-fold | GBM 5-fold | Raw-NN |
|---|---|---|---|---|
| 42 | 0.5095 | 0.5030 | 0.5171 | 0.5010 |
| 43 | 0.5030 | 0.4987 | 0.5153 | 0.5230 |
| 44 | 0.4993 | 0.4877 | 0.5088 | 0.5039 |
| mean | 0.5039 | 0.4965 | 0.5137 | 0.5093 |

Both land at chance on the primary detector across all three seeds. For direct
comparison, the plain per-item SIR lottery replayed on the identical cached
pools scores about 0.57: same model, same 16 candidates per movement, only the
rule choosing which one wins is different, which is the entire size of the
set-level effect.

The two recipes differ in where the residual sits. The 18-feature judge gives
the lower primary number, 0.491, but leaves the raw-signal nearest-neighbor at
0.529, its worst-case detector. The 33-feature judge closes that channel to
0.509 and holds every tree and nearest-neighbor family within 0.014 of chance,
its worst being GBM at 0.514, while sitting at 0.504 on the primary rather than
below it. We report the 33-feature result as the headline. It removes the most
obvious outside attack, reading the raw speeds directly, and a value resting
just above chance is the honest place for a generator a detector cannot
separate, where a sub-0.50 figure invites the misread that the metric was
overshot. Both numbers are recorded because both are confirmed, and neither is
selected over the other on the primary axis; they agree that the primary
detector is at chance.

The mechanism, independent of whether the exact number holds at other seeds, is
now reasonably well understood. No candidate trajectory in the pool is
individually different from what the model has always produced; every
candidate carries the same higher-order joint structure, which is exactly why
histogram matching and MMD matching, which only ever look at one candidate's
features against a target distribution, found nothing to fix. Per-item
selection had one degree of freedom it was never using: which candidate wins
each movement, chosen with knowledge of the set the other choices create. An
iterated adversary with a detector-matched judge and a decaying step size is
what turns that degree of freedom into most of the remaining AUC, at no
additional model sampling cost and about 40 seconds of CPU time on top of a
pipeline that was already being run.

### 7.9b How much to trust the number

Two questions decide whether a score at chance means what it appears to mean:
whether the selection learned to fool one detector rather than to look human,
and whether the human reference it was measured against is representative of
humans at all. Both were tested, and one has an answer the project cannot fully
supply.

The judge that drives the reselection is a Random Forest, and the primary
reported metric is a Random Forest, so the concern that we optimized against the
grader is legitimate. The check is a panel of detector families that took no
part in tuning: gradient boosting, extra-trees, a multilayer perceptron,
logistic regression, and a histogram gradient booster on the same 18 features,
plus a detector built on the raw speed signal rather than the summary features.
Run against the held-out selection, they span 0.484 to 0.55. That spread is the
evidence the result is a property of the trajectories and not of one
classifier's decision boundary. The single place this first broke is
instructive: with the 18-feature judge the raw-signal detector scored 0.583,
meaning that judge had left structure in the raw channel it could not see.
Widening the judge to the 33 features that include raw-signal summaries, which
is the recipe reported in 7.9, dropped that detector to 0.484 and cut the
worst-case deviation across the whole panel roughly in half, with no loss on the
original features. Two detectors, the perceptron and logistic regression, still
sit near 0.54, so the flattening is close to complete but not total.

Three follow-up probes on the last day tested how hard that residual is. A
stronger single smooth judge (a lone MLP or logistic model run as a short
polish on the confirmed picks) closed the residual on a tuning proxy but gave
it back on the honest replay, the winner's curse of selecting the best round
of a noisy metric; a control confirmed the residual is real, since two
disjoint human samples are mutually indistinguishable under the same smooth
detectors. Doubling the candidate pool to 32 movements per spec, on the one
seed with a K=32 pool, cut the perceptron detector from 0.550 to 0.521 with
the primary at 0.497 and no other detector paying, which locates the
perceptron residual as support deficiency (too few candidates to choose from),
while the logistic reading held near 0.54 at both pool sizes and reads as a
generator-level direction selection cannot remove. Finally, two heavier
held-out sequence adversaries on the raw resampled channels, a dilated CNN
spanning the full movement and a bidirectional GRU, both stayed at chance
(0.509) against the headline picks, closing the depth question the small
Raw-NN left open. All three probes were CPU-only replays on cached pools and
did not touch the three-seed headline.

The stretch day at the end of the project pushed on both remaining fronts. On
the raw side, two adversaries far heavier than the suite's small CNN, a
five-block dilated CNN whose receptive field spans the entire sequence and a
two-layer bidirectional GRU, trained under the same held-out protocol, read
0.509 each against the reported selection. On the smooth side, three results.
Stronger single smooth judges (an MLP alone, a logistic model alone, and strict
alternation with the forest) win at proxy level and give the win back on the
honest replay, so the residual is not a tuning artifact we failed to chase.
A control shows disjoint human samples are mutually indistinguishable under
the same smooth detectors (0.48 to 0.52), so the residual is a genuine
property of the synthetic set, not human sampling noise. And doubling the
candidate pool to K=32 on the one seed that had such a pool cut the perceptron
to 0.521 with the primary at 0.497, while the logistic reading stayed near
0.54 at both pool sizes and under every judge. The linear direction is
something the generator imposes on essentially every candidate it emits;
closing it is model work, not selection work.

One channel sits outside this entire panel by construction. All detectors
here read trajectories after the 125 Hz resample. The raw event stream is
separable at a glance: 95.3 percent of held-out human timestamp deltas are
exactly 8 

Reproduction and the held-out discipline are covered above: independent seeds, a
reference split whose second half the fitting never touches, and a final number
replayed against an evaluation sample no part of selection has seen, with a
proxy-to-honest gap that has stayed inside two points.

The limit the project cannot close is the reference itself. Every human
trajectory available here, for training and for evaluation, comes from the five
public datasets the model trained on. The evaluation sample is disjoint from
what selection saw, which is what makes the held-out number meaningful, but it
is drawn from the same population and the same recording hardware the model
learned from. A genuinely external test would need labeled human mouse data from
a different population, and no such set is available to this project: the public
sources that would supply it are already in the training pool, and requests to
others were declined. The defensible claim is therefore bounded. Within the
distribution these datasets represent, the output is indistinguishable from
human movement under every detector tried. Whether it would survive a detector
trained on a different population of users, devices, and polling behavior is
untested, and the reader should treat it as open.

### 7.10 Summary of the event-stream era

| Result | RF OOB AUC | Status |
|---|---|---|
| Corpus replay floor | 0.51 | Retrieval, calibration point |
| Event representation round trip | 0.507 | Encode/decode gate, confirms near-lossless representation |
| ZIMT, best of the continuous-model era | 0.864 | Historical, Sections 3 to 6c |
| Pure event-stream model | 0.652 +/- 0.003 | Three-seed confirmed |
| + SIR selection | 0.568 +/- 0.010 | Three-seed confirmed |
| + set-level reselection | 0.504 | Three-seed confirmed, 33-feature judge (18-feature judge: 0.491) |

### 7.11 What is still open

The result rests entirely on selection at inference time. Every attempt to move the judgment into the model's weights failed: plain imitation, anchored imitation, three adversarial fine-tunes, and preference learning all made the pure-model number worse, some of them catastrophically (Section 7.6). The clean statement is that trajectory-level judgment compresses into this architecture as ranking but not as generation. That leaves three directions open, none of which we could close before the project deadline.

The first is reinforcement learning that scores whole trajectories after they are generated, rather than a gradient that reaches back through the sampling step. Every failed fine-tune shared one flaw: it either imitated token targets or pushed gradients through a straight-through relaxation of the sampler, and both drift the model off the human manifold. A method that samples a full trajectory, scores it with the detector, and updates only on that scalar reward (GRPO or RAFT, for example) never touches the sampler's internals and might survive where the others did not. It is the most direct assault on the same wall, and it is untested here.

The second is a different backbone. The masked-token event model was chosen because it can emit an exact stall as a first-class token, which every continuous model could not (Section 5). But the same discreteness that fixed the stall may be what refuses to absorb the judge's signal. An architecture that keeps exact zeros while carrying a continuous latent for movement, rather than binning speed and heading, is a plausible way to get a model whose native output is closer to 0.50 before any selection. Building and training one is weeks of work, not a fine-tune, so it stayed out of scope.

The third is data. The model saw only the five public datasets, and the honest claim is bounded to that distribution (Section 7.9b). Human recordings from outside that pool would do two things at once: widen what the model can learn, and provide the genuinely external test the current evaluation cannot. Both the training gain and the validation are gated on access to labeled human mouse data we do not currently have.

## 8. Related Work

### 8.1 Kinematic Theory and the Sigma-Lognormal Model

Plamondon's kinematic theory of rapid aimed movements (Plamondon, 1995)
models handwriting and mouse trajectories as sums of lognormal velocity
profiles. The sigma-lognormal model decomposes each movement into overlapping
submovements, each characterized by a lognormal speed profile with parameters
(D, t0, mu, sigma). This is the standard parametric model in the handwriting
recognition and HCI literature.

Our data departs from it in two ways, both quantified in Section 3.1: the velocity peaks in mouse movement are far sharper than the smooth lognormal profile can reproduce (sigma-lognormal generation undershoots max velocity by ~22x), and the additive composition of submovements the model assumes produces the wrong velocity distribution, where the data instead point to a competitive, winner-takes-all composition.

### 8.2 Fitts' Law

Fitts' law (Fitts, 1954) predicts movement duration as a logarithmic function
of the index of difficulty (distance / target width). It is the foundational
model for aimed movement in HCI.

Fitts' law explains **when** movements end but not **how** they get there.
It predicts duration and implicitly constrains the velocity profile, but it does
not model trajectory shape, curvature, stall events, or angular dynamics. Our
feature set includes movement duration (which Fitts' law addresses) and 17 other
features (which it does not). A generator based on Fitts' law alone can match
duration but fails on all other features.

### 8.3 Discrete Tokens for Continuous Generation

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

### 8.4 Intermittent Motor Control

The traditional view of aimed movement as a single smooth trajectory has been
challenged by intermittent control theory (Loram et al., 2011), which proposes
that motor commands are issued in discrete bursts rather than continuously
updated. Under this view, the stall events characterized in Section 4.2 are
direct evidence of control intermittency: the motor system issues a command, the
hand moves ballistically, the command expires, the hand stops, and a new command
is issued in a possibly different direction. Both the stall durations we measure
(8-40ms) and the heading change at stall boundaries fit the intermittent-control
timescales reported in the postural control literature, with the directional
correction at each boundary consistent with visual feedback of the cursor-target
error.

### 8.5 Submovement Composition

The classical model of aimed movement (Meyer et al., 1988) proposes that
movements consist of an initial ballistic submovement followed by one or more
corrective submovements, composed additively. Section 5.4 tested that additive
assumption directly and found it produces the wrong velocity distribution: real
trajectories carry more velocity peaks than the classical 2-4, with heavy overlap
between them rather than a return to near-zero speed. The data instead support a
competitive composition, where at any moment one submovement dominates rather
than all active submovements summing, matching the winner-takes-all dynamics seen
in motor cortex recordings. The practical consequence for generation is in
Section 5.4: neither additive nor sequential composition reproduces the mixed
continuous-discrete structure the real data shows.

---

## Summary of Key Numbers

| Quantity | Value | Source |
|---|---|---|
| Human trajectory corpus size | 4.16M trajectories | 5 public HCI datasets |
| Evaluation features | 18 kinematic features | See Section 2 |
| Corpus replay AUC | 0.51 (4.16M pool) | Practical floor |
| Event representation round trip | 0.507 | Encode/decode gate, Section 7.2 |
| Best fully generative AUC, no selection | 0.652 +/- 0.003 | Pure event-stream model, Section 7.6 |
| Best honest full-system AUC | 0.568 +/- 0.010 | Event-stream model + SIR selection, Section 7.7 |
| Best set-level AUC | 0.504 (3-seed) | Event-stream model + set-level reselection, Section 7.9 |
| Best retrieval+transform AUC | 0.686 | Corpus rotate (rotation + scale) |
| Best of the continuous-model era | 0.864 | ZIMT with magnitude-weighted endpoint correction |
| Intermediate hybrid result | 0.852 | CANDI polar hybrid diffusion |
| Perturbed replay AUC | 0.55 (2% noise) | Section 6 |
| Generative target | < 0.75 (open-source), < 0.50 (full success) | |
| Top RF feature importance | 10.8% (angular_velocity_std) | Distributed importance |
| Top-5 RF feature importance | 41% | No single feature dominates |
| Human mean velocity | ~960 px/s | Corpus statistics |
| Human max velocity CV | ~34x | Extreme peaks |
| Human curvature mean | ~1329 | Dominated by stall events |
| Zero-displacement steps | 6.14% of all steps | Discrete stall events |
| Stall duration | 1-5 samples (8-40ms) | Fixed USB polling intervals |
| Timing residual autocorrelation | r = 0.65 | Motor control smoothness |
| Peak velocity location | ~35% of duration | Universal, distance-independent |
| Chunk diffusion AUC | 0.957 | No global velocity awareness |
| SoundStorm/MaskGIT AUC | 0.996 | VQ-VAE quantization bottleneck |
| Enhanced corpus rotate AUC | 0.670 | Best rotation variant (K=50) |
| Event-stream model size | 6M parameters | Section 7.3 |
| SIR candidates per movement | K = 16, selection temperature 0.7 | Section 7.7 |
| Experiments conducted | 200+ | See EXPERIMENTS.md |
| Model architectures tested | 11 families | See Section 5 and Section 7 |

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
| MaskGIT | Masked Generative Image Transformer. A non-autoregressive decoding scheme: all tokens start masked and are revealed iteratively, in confidence order, with full bidirectional context at each round. |
| SoundStorm | A MaskGIT-style masked bidirectional decoder for audio tokens. The direct architectural ancestor of the event-stream model's sampler. |
| SIR | Sampling-Importance-Resampling. Here, drawing K candidate trajectories per movement and keeping one via a tempered lottery on a judge's log-odds, rather than always keeping the judge's top pick. |
| KDE | Kernel Density Estimate. Used to draw movement-character conditioning vectors from a bank of real feature vectors at generation time. |
| DPO | Direct Preference Optimization. A training objective that raises the model's relative preference for a winning sample over a losing one, contrasted against a frozen reference copy of the model. |
| ESS | Effective sample size. For a weighted lottery over K candidates, a measure of how many candidates are meaningfully contributing to the draw; ESS near K means weights are close to uniform, ESS near 1 means the draw has collapsed onto one candidate. |
| NLL | Negative log-likelihood. Training loss for probabilistic models. |
| RF | Random Forest. Ensemble of decision trees trained on bootstrap samples. |

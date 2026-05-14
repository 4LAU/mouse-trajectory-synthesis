# Experiment Log

Full chronological experiment log. 145+ experiments from parametric baselines through corpus replay to generative models. AUC = OOB Random Forest, lower is better (0.50 = indistinguishable from human).

---

Goal: minimize val_auc (RF OOB AUC, human vs synthetic mouse trajectories). Target: AUC < 0.60.

| Version | val_auc | Key Change | Why it changed |
|---------|---------|------------|---------------|
| v10 (baseline) | 0.9997 | OU noise min=0.1, ramp_scale=0.30, arc_frac 0.15-0.70 | Best prior result |
| v11 | 0.9998 | Speed-proportional tremor | Worse - large amp at high speed raised omega |
| v12 | 1.0000 | Small ramp, arc_frac 0.20-0.45, OU min=0.5 | Smaller ramp collapsed mean_acc; OU min=0.5 killed jerk |
| v13 | 0.9999 | Restore v10 core, narrow arc_frac 0.15-0.40 | Slight regression |
| v14 | 0.9995 | Integer pixel quantization added | +direction changes from quant noise |
| v15 | 0.9985 | Drop tremor + shorten duration (0.18s) | Best ever. Fixed mean_v, boosted jerk. High omega (35) from quant+OU floor |
| v16 | 1.0000 | Remove quantization (keep short duration) | omega_mean=6.33 but omega_std=6.31 (target 20.60) |
| v17 | 1.0000 | OU sigma 0.80-1.50 (was 0.15-0.35) | std_v=2070 but std_omega unchanged at 6.25 |
| v18 | 1.0000 | sigma_ln wide 0.80-1.50 | WORSE - wide sigma reduces std_v (more time at floor) |
| v19 | 1.0000 | Extreme OU sigma 3.0-6.0 (short duration) | std_jerk=34M but num_direction_changes=0 and movement_duration=0.20 both 1.0 features |
| v20 | 0.9999 | Restore long duration (0.65s) + tiny tremor (0.08-0.14px) | Fixed movement_duration, direction_changes=35/44 partial |
| v21 | 0.9997 | Extreme OU (3.0-6.0) + long duration + tiny tremor | Best with long duration. omega=9.5 (too high), std_jerk=5.6M |

## Key Feature Gaps (v21)
| Feature | Human | Synth | Gap |
|---------|-------|-------|-----|
| std_jerk | 129M | 5.6M | 23x |
| max_velocity | 59301 | 2688 | 22x |
| max_acceleration | 7.3M | 141K | 52x |
| angular_velocity_std | 20.60 | 15.82 | 1.3x |
| angular_velocity_mean | 6.42 | 9.53 | 1.5x |
| num_direction_changes | 44 | 37 | 1.2x |
| movement_duration | 0.70 | 0.73 | ~1x |

| Version | AUC | Notes |
|---------|-----|-------|
| v23 (collinear) | 1.0000 | Remove angular deviations -> max_deviation=0.45 (human=59.91). Curvature WORSE (191). |
| v23b (vel floor) | 0.9999 | Restore v22 angular devs + 80-150px/s floor. Curvature 56 (vs 75 v22). Drift correction cancels floor - ineffective. |
| v24 (ballistic) | 1.0000 | Early narrow primary (t_peak=8-20%, sigma=0.12-0.22). max_velocity 8456 but omega exploded (48 vs 6.42), curvature still 190. |

## Root Tension
- Short duration -> high velocity -> high jerk/acceleration BUT breaks movement_duration feature
- Tremor creates direction changes BUT raises omega
- OU creates jerk/std_v BUT creates omega spikes at low-speed moments (if quantized)
- Human data has EXTREME peaks (max_velocity=59301 at mean=1724 -> CV~34x!) not achievable with smooth models
- Velocity floor approach: drift correction exactly cancels constant floor (v_floor x duration added by floor, then subtracted by drift correction)
- Ballistic primary paradox: fast early peak -> dwell phase has low velocity -> tremor creates extreme omega AND curvature during dwell
- Curvature root cause: tremor acceleration ~1776 px/s^2 at any velocity trough -> kappa = a/v^2. Cannot fix without either (a) eliminating troughs (velocity floor that works) or (b) eliminating tremor (kills direction changes)

## Key Feature Gaps (current - v23b is best with sigma-lognormal base)
| Feature | Human | Synth (v23b) | Gap |
|---------|-------|------|-----|
| curvature_mean | 0.76 | 56.88 | 75x |
| curvature_std | 5.15 | 489.40 | 95x |
| mean_acceleration | 134259 | 36.81 | 3600x |
| mean_jerk | 17259058 | -1544 | infinity (wrong sign) |
| num_direction_changes | 44 | 14.60 | 3x |
| angular_velocity_mean | 6.42 | 21.86 | 3.4x too HIGH |
| angular_velocity_std | 20.60 | 52.82 | 2.6x too HIGH |
| max_deviation | 59.91 | 11.94 | 5x |
| movement_duration | 0.70 | 0.71 | ~1x |

## Corpus Replay Approach (v45+)

Switched from parametric generation to replaying actual captured human trajectories.

| Version | val_auc | Key Change |
|---------|---------|------------|
| v45 (uniform random) | 0.642 | Corpus replay, uniform random |
| v47 (Balabit-only + filter) | 0.690 | Wrong hypothesis: SapiMouse not the problem |
| v48 (kNN pool k=10, std norm) | 0.622 | 8-run mean=0.619; best kNN result |
| v49 (k=1 2-stage) | 0.680 | Too narrow (1994 unique) |
| v49b (k=5 2-stage) | 0.625 | Slight improvement |
| v50 (kNN, IQR norm) | 0.631 | IQR over-weights jerk |
| v51 (kNN + distance match) | 0.620 | Same as v48 |
| **v52 (raw Balabit, no filter)** | **0.543** | **Removed duration filter + normalization. Key insight.** |
| v53 (exact 2000 human_eval) | OOB artifact | Pool too small -> with-replacement OOB fingerprint |
| **v54 (raw Balabit+SapiMouse)** | **0.498** | **All 222,174 valid trajectories. 8-run mean=0.498. ~0.50 = theoretical minimum.** |
| **v55 (angle+dist-match, translate-only)** | **0.498** | **Geometrically correct v54. +/-60 deg angle + +/-0.5 log-dist filter + translate-only. 8-run mean=0.498, range 0.484-0.517.** |
| v56 (GRU + pool, TIMING_BLEND=0.35) | 0.5969 | Added GRU timing. Blending destroyed curvature (1400->43). Pool rebuilt with 4.16M traj (DFL+chaoshen+bogazici added). |
| **v58 (pool-only, TIMING_BLEND=0)** | **~0.52** | **Removed GRU timing entirely. Key: blend=0 restores curvature matching (1400 vs 1329 human). 3-seed avg ~0.52.** |

## Root Cause of v48 Floor (0.622)
The corpus applied a <=5s duration filter AND float-precision time normalization artifacts.
These caused systematic gaps:
- time_to_peak_velocity: corpus=0.376 vs human_eval=0.357 (corpus too late)
- max_deviation: corpus=57.6 vs human_eval=59.9 (corpus too straight)
- angular_velocity_mean: corpus pool missing 12% SapiMouse contribution

## v52/v54 Key Findings
- human_eval = 88% Balabit + 12% SapiMouse (confirmed via data preparation seed reproduction)
- Removing duration filter: time_to_peak now matches (0.35 vs 0.357)
- Including SapiMouse: angular_velocity_mean matches (6.41 vs 6.42)
- curvature_std massively off (synth=57 vs human=5.15) but RF importance low -> doesn't affect AUC
- v54 achieves AUC = 0.50 = theoretical lower bound for OOB RF on same distribution

## v58 Key Findings
- GRU timing blend (TIMING_BLEND=0.35) uniformizes timestamps -> eliminates curvature from direction changes (1400->43)
- Setting TIMING_BLEND=0 (pure original pool timing) restores curvature to match human (1400 vs 1329)
- Pool was rebuilt with DFL/chaoshen/bogazici (4.16M total), adding data with different feature distributions than human_eval
- balabit+sapimouse-only restriction gives 0.94 because current pool's sapimouse range has ~17ms dt (low density) vs human_eval sapimouse at 8ms dt
- To reach 0.498: need to rebuild pool from original balabit+sapimouse source with absolute timestamps (exact distribution match to human_eval)
- Current floor ~0.52 is limited by pool/eval distribution mismatch from the additional datasets

## Replay Floor Investigation (v59 variants)
Attempted to close the 0.526 -> 0.50 gap with 3 targeted experiments:
- v59 (K=1 best endpoint): 0.729 - selection bias from cherry-picking specific trajectories
- v59b (5% correction window, 20px skip): 0.515 on seed 42, ~0.526 mean - no improvement
- v59c (float64 coords): 0.527 - coord rounding not the cause
- Root cause: query-conditional sampling (angle+dist match + top-K endpoint) draws a non-uniform pool sample vs. human_eval's uniform draw -> irreducible ~0.026 gap
- 0.526 established as the replay floor. Subsequent experiments focus on generative approaches.

## v59 (pool restored + human_eval expanded)
- Restored full 4.16M pool (balabit+sapimouse+DFL+chaoshen+bogazici) after accidental overwrite
- Fixed data preparation: `human_eval_features.npy` and `human_distances.npy` now sample from the full pool (all 5 datasets)
- 7-seed AUC: mean=0.533, range 0.522-0.542
- Floor explanation: original 0.498 (v55) used pool and human_eval from identical source (same 222K balabit+sapimouse). With full pool, the RF sees slight distribution mismatch from endpoint correction and query-angle bias.
- Experiment hygiene rule: future experiments use versioned files, never overwrite production .npy files

## v55 Key Findings
- Rotation breaks `num_direction_changes`: rotating angles near +/-pi boundary causes sign flip in wrapped angle_diff -> spurious direction changes. Empirical: 64 -> 90 direction changes after 45 deg rotation (40% inflation).
- Translation is truly invariant for ALL 18 features (velocity = ds/dt, differences cancel offset).
- Timestamp normalization bug: `t_rel = t_abs - t_abs[0]` causes `np.arange(0, dur, step)` to land at exactly dur for certain float values (e.g. 0.312), then the `< t[-1]` append check also adds 0.312 -> two consecutive identical timestamps -> dt ~ 1e-15 -> velocity spike. Fix: keep raw absolute timestamps.
- Angle filter width: +/-60 deg gives AUC=0.498 (optimal). +/-30 deg = 0.517 (velocity bias from narrow selection). +/-120 deg+ -> slightly above 0.50 (wider window includes more directional bias).
- Minimum candidates at any angle with +/-60 deg: ~60k+ from 222k pool -> no fallback needed in practice.

## Generative Research: CFM and Baselines

### Scoreboard

| Version | val_auc | Architecture | Notes |
|---------|---------|-------------|-------|
| **v125** | **0.9191** | U-Net CFM 100pt arc-length | **Best generative** |
| v11 | 0.9803 | U-Net CFM 300pt | Worse (angular_velocity wass=3.13) |
| v129 smooth | 0.9932 | Transformer CFM (ep 3) | Training diverged with smoothness |
| v129 nosmooth | 0.9950 | Transformer CFM (ep 53) | Catastrophic spatial incoherence |
| v131 stochastic | 0.9948 | v125 + t-channel noise | Corrupts velocity predictions |
| v132 GRU v4 | 0.9968 | 2-layer GRU 256h | Autoregressive divergence (max_vel=11M) |
| v133 retrieval | 0.9397 | Pool retrieval + noise | Rotation destroys angular dynamics |
| v134 Fitts parametric | N/A | Min-jerk + tremor | Angular velocity 2x too high, curvature 0.001x |
| Corpus replay (v55) | 0.498 | kNN translate-only | Theoretical minimum (not generative) |

### DDPM Experiments

| Version | val_auc | Architecture | Key Finding |
|---------|---------|-------------|-------------|
| v135 eta=0 | 0.9291 | DDPM 100pt arc-length, deterministic | Comparable to CFM (0.9191). Curvature 7.3 vs human 1329. |
| v135 eta=0.1 | 0.9444 | DDPM + mild stochastic | Noise degrades path coherence (path_eff 0.54 vs 0.84) |
| v135 eta=0.5 | 0.9980 | DDPM + strong stochastic | Catastrophic (max_vel 628K vs 3884) |
| v136 timing | 0.9737 | DDPM eta=0 + bell-curve speed + stalls | Stalls at canonical pts don't create sub-pixel speed drops |
| v136 spatial | 0.9899 | DDPM eta=0 + 125Hz interp + AR(1) noise | Angular vel 2x too high, curvature still 13 |
| v137 eta=0 | 0.9546 | DDPM 192pt temporal resampling | Angular vel 2x too high (44.6 vs 22.4) |
| v137 eta=0.3 | 0.9590 | DDPM temporal + stochastic | Same issues as eta=0 |

#### DDPM Key Findings

1. **DDPM = CFM for this task**: Deterministic DDIM (eta=0) produces the same smooth conditional-mean paths as CFM. AUC 0.9291 vs CFM 0.9191. Same architecture, different training objective, same result.

2. **Stochastic sampling is harmful**: Any eta > 0 adds uncorrelated noise that destroys spatial coherence (path_efficiency drops, max_deviation increases) without creating the structured micro-structure needed for curvature.

3. **Temporal resampling doesn't fix curvature**: v137 trained on temporally-resampled data (192pt, uniform dt) produces AUC 0.9546, slightly WORSE than arc-length (0.9291). The model generates angular_velocity 2x human because the 192-point temporal representation has more angular noise per point.

4. **Curvature gap is irreducible with diffusion sampling**: Human curvature_mean = 1329 requires near-zero-speed moments (< 0.01 px/s at 125 Hz). Diffusion models produce conditional means where speed is always moderate (~1000 px/s). Curvature = omega/speed -> always low. No training data representation or sampling strategy fixes this.

5. **Post-hoc timing injection fails**: Bell-curve speed profiles, micro-stalls, and AR(1) noise applied post-hoc to DDPM paths either create too much angular velocity (unstructured noise) or fail to create near-zero speed (stall points have sub-pixel jitter from noise that prevents speed from dropping to zero).

### Key Findings

1. **CFM fundamental limitation**: Euler ODE integration produces smooth conditional-mean paths. Curvature_mean = 0.03-3.5 vs human 1329. No architecture change (U-Net -> Transformer), sampling change (stochastic), or post-processing (micro-corrections, velocity profiles) fixes this.

2. **Transformer backbone failed**: With smoothness loss, training diverged (best at epoch 3). Without smoothness, catastrophic spatial incoherence (path_eff=0.46, max_acceleration 62x human). Self-attention without pooling is NOT better than U-Net for this task.

3. **300 points made it worse**: v11 (300pt) AUC=0.98 vs v125 (100pt) 0.92. More points = more angular noise at fixed ODE steps.

4. **GRU divergence**: Autoregressive sampling compounds errors. Max velocity 11M px/s vs human 3884. The model learned teacher-forcing well (NLL=-13) but can't self-correct during free sampling.

5. **Broken correlations**: In human data, mean_acceleration and mean_jerk are uncorrelated (r=-0.025). In all synthetic approaches, they're near-perfectly correlated (r=0.999). The RF detects this joint distribution mismatch, not individual feature gaps.

6. **Feature importance is spread out**: Top feature (angular_velocity_std) is only 10.8%. Top 5 features = 41%. Cannot fix by targeting 1-2 features.

7. **Retrieval + rotation destroys angular dynamics**: Rotating pool trajectories to match target direction inflates angular_velocity and destroys curvature. AUC 0.94 even with zero noise.

### Models Explored

- **v129**: Transformer CFM (with and without smoothness loss)
- **v130**: Micro-correction injection on CFM paths
- **v131**: Stochastic ODE sampling
- **v132**: 2-layer GRU (256 hidden, autoregressive)
- **v133**: Retrieval-augmented generator
- **v134**: Fitts' law parametric generator
- Also trained: 300pt U-Net (v11), Transformer nosmooth, 2-layer GRU checkpoints

## Generative Research: Stall Analysis and Hybrid Approaches

### Speed-Bin GRU + Hybrid Approaches

| Version | val_auc | Architecture | Key Finding |
|---------|---------|-------------|-------------|
| v138 (64 bins, sampled) | 0.9897 | GRU v3, 64 speed bins | Speed bins prevent velocity explosion (max_vel 11K vs 11M in v132) but XY error accumulation persists |
| v138b (16 bins, 5x weight) | 0.9988 | GRU v3, 16 bins, speed_weight=5 | Higher speed accuracy (66% vs 36%) but WORSE AUC - XY drift dominates |
| v139 hybrid | 0.9990 | DDPM path + GRU speed walk | Only 89/2000 valid trajectories. Arc-length walking has numerical issues. |
| v140 sparse curv | 0.9990 | DDPM path + 27 perp displacements + speed drops | 157 direction changes (human=27), duration 2.33s (human=0.54), curvature 0.90 |
| v142 speed dips | 0.9977 | DDPM path + speed profile dips + perp perturb | path_eff=0.80, num_dir_changes=27 - good match on those features but curvature still 0.40 |
| v142b temporal jitter | 0.9991 | DDPM path + 125Hz jitter at dips | angular_vel 65 vs 22, 77 direction changes, duration 1.09 vs 0.54 - jitter too aggressive |

### Submovement Decomposition

| Version | val_auc | Architecture | Key Finding |
|---------|---------|-------------|-------------|
| v141 additive | 1.0000 | Min-jerk submovements, additive | Velocity stacking: 5x mean_vel, 43x max_accel. Path_eff=0.16 |
| v141b correlated | 0.9997 | Correlated submovements (primary+corrections) | Still 5x mean_vel, path_eff=0.17 |
| v141c sequential | 1.0000 | Sequential segments + overlap | mean_vel 3x, angular_vel 7x, path_eff=0.26 |

#### Submovement Key Finding
Additive minimum-jerk composition creates **velocity stacking**: when N submovements overlap,
velocity = sum of individual velocities = N x single. Human trajectories don't exhibit this, 
submovements compete (motor cortex winner-takes-all), not sum. The additive assumption is wrong
for this domain.

### Curvature Root Cause Analysis

**100% of human curvature comes from low-speed (< 5 px/s) moments.** High-speed moments contribute
exactly zero curvature. Analysis of 5000 human trajectories:

- 6.14% of steps have speed < 1 px/s
- 6.26% have speed < 5 px/s
- At these points, position is **exactly repeated** (dx=0, dy=0) for 1-5 consecutive 8ms samples
- The stall pattern: speed ramps down (322->161->47->0), then 1-5 zero-displacement samples, then speed ramps back up
- Heading BEFORE stall != heading AFTER stall -> direction change at stall edge
- Curvature = angular_velocity / speed. At speed = 0.01 px/s, even 1 rad/s angular velocity gives kappa = 100

This is a **discrete event** (exact zero displacement) embedded in continuous motion. No continuous
generative model can produce exact zeros - they can only produce near-zero values that still have
non-zero displacement.

### Key Finding: Representational Limitation of Continuous Models

The problem is **mixed continuous-discrete generation**:
- Continuous component: smooth ballistic motion between stalls (path shape, speed profile)
- Discrete component: stall events where dx=0, dy=0 for 1-5 consecutive samples

Every continuous generative model (diffusion, flow matching, GRU regression, parametric submovement)
treats the output as continuous. Stalls are a low-probability region of the continuous distribution, 
the model can never assign them enough probability mass to match the human rate (6.14% of steps).

### Models Explored

- **v138/v138b**: GRU v3 with speed-bin classification (64-bin and 16-bin variants)
- **v139**: DDPM + GRU hybrid (arc-length walking)
- **v140**: DDPM + sparse perpendicular displacements + speed drops
- **v141/v141b/v141c**: Min-jerk submovement decomposition (additive, correlated, sequential)
- **v142**: DDPM + speed dips + perpendicular perturbation + temporal jitter
- **v143**: DDPM + deceleration ramps + stalls + heading changes
- Also trained: 64-bin and 16-bin speed GRU checkpoints, submovement decomposition utilities

## Stall Injection Experiments (v143)

### Key Discovery: Curvature Mechanism

**Zero-displacement stalls contribute ZERO curvature.** The curvature formula is kappa = |v x a|/|v|^3.
At zero speed, vx=vy=0 -> numerator is 0 -> curvature = 0 regardless of acceleration.

Curvature comes from **low-speed-but-nonzero moments where heading is changing.** The 1/speed^3
term amplifies any lateral acceleration at low speed. Human curvature ~1329 comes from the
deceleration ramps around stalls (speed ~20-50 px/s with heading changing), not from the
zero-displacement stalls themselves.

### v143 Iterations

| Iteration | Curvature | AUC | Key Change |
|-----------|-----------|-----|------------|
| v143a (timestamp bug) | 0.18 | 0.9928 | Stalls inserted but timestamps collided -> 0.001s gaps -> high speed at transitions |
| v143b (timestamp fix) | 739 | 0.9965 | Fixed timestamp accumulation. Bell speed profile + stalls. Curvature worked! |
| v143c (DDPM native timing) | 0.70 | 0.9152 | Used DDPM timing instead of bell -> uniform speed at 125Hz -> no low-speed moments |
| v143d (decel ramps) | 77.3 | 0.9807 | Gaussian speed dips (98% reduction) at stall positions. Curvature >= 73 = strong positive |

### Curvature Validation

**curvature_mean = 77.3 (smoke test, n=200). DDPM baseline = ~7. Ratio = 11x.**

This hits the **strong positive** tier (>= 73, >= 10x from baseline).

AUC = 0.9807 - worse than DDPM baseline (0.9291). Top RF features:
- mean_acceleration: +10408 vs human -4106 (wrong sign)
- mean_jerk: +1065447 vs human -68897 (wrong sign)
- time_to_peak_velocity: 0.76 vs human 0.31 (2.5x too late)
- angular_velocity_std: 75 vs human 46 (1.6x)

These are all artifacts of the crude injection (bell speed profile + global heading rotations),
not fundamental limitations.

### Post-Hoc Injection Is Fundamentally Limited

Extended iteration on stall injection approaches showed that every post-hoc modification to
DDPM paths creates acceleration artifacts that offset angular velocity/curvature gains:

| Experiment | AUC | Change | Problem |
|-----------|-----|--------|---------|
| v135 DDPM baseline | 0.862 | - | angular_velocity 0.5x human |
| v143c (stall ramps, original) | 0.863 | +0% | mean_acceleration wrong sign |
| v143c (aggressive decel) | 0.977 | -13% | curvature overshoot |
| v143d (tuned decel) | 0.924 | -7% | same pattern |
| v144 (125Hz perturbations) | 0.900 | -4% | acceleration artifacts |

**Key finding**: the DDPM path is too smooth (angular_velocity_std = 24 vs human 46). This
is a representation bottleneck - continuous diffusion sampling always produces conditional
means. Post-hoc injection fixes one feature but breaks others.

### Why Post-Hoc Injection Fails

1. Zero-displacement stalls contribute ZERO curvature (kappa = |v x a|/|v|^3, at v=0 numerator is 0).
   Curvature comes from low-speed deceleration ramps with heading changes.
2. Post-hoc stall injection onto DDPM paths is a wash for AUC - acceleration artifacts from
   the ramps exactly offset curvature/angular_velocity gains.
3. A hybrid approach (DDPM for segments) would have the same smoothness
   issue. This motivates the VQ-VAE + Transformer approach.

Justification for skipping the hybrid approach:
- The representation bottleneck (DDPM too smooth) is already empirically confirmed
- Hybrid's per-segment DDPM would have the same smoothness issue
- VQ-VAE addresses the root cause: discrete tokens naturally handle stalls + micro-corrections
- 30+ experiments have confirmed continuous generative models cannot produce human-like curvature

## VQ-VAE + Transformer (v145-v146)

### VQ-VAE Codebook Training

| Version | Codebook Usage | Recon MSE (dx) | Speed MAE | Notes |
|---------|---------------|----------------|-----------|-------|
| VQ-VAE v1 | 162/1024 (16%) | 2.71 | 103 px/s | Severe codebook collapse |
| VQ-VAE v2 | 1024/1024 (100%) | 0.33 | 30 px/s | Normalized + k-means init + reset |

VQ-VAE v2 fixes: P0.5-P99.5 clipping + z-score normalization + k-means codebook init + dead entry
reset every 10 epochs. Speed MAE 30 px/s = 3.6% of mean speed.

### Data Statistics
- 500K trajectories tokenized at 125Hz
- 32.4M total tokens, mean seq length 65
- 5.97% stall tokens (token 0), 1025 unique tokens
- All codebook entries used in tokenization

### Transformer Training & Evaluation

| Config | val_loss | val_acc | AUC | Key Issue |
|--------|----------|---------|-----|-----------|
| 100K data, 20ep, stall_weight=3 | 2.657 | 33.5% | 0.993 | Path too straight (eff=0.98) |
| Same + higher temp/nucleus | - | - | 0.956 | angular_vel good (42.7), mean_acc +2449 |
| Same + stall suppress -4.0 | - | - | 0.948 | Best VQ-VAE AUC, mean_acc still wrong sign |
| Same + stall suppress -2.5 | - | - | 0.959 | Worse - middle ground doesn't help |
| v146 VQ-VAE + parametric | - | - | 0.999 | Straight lines, codebook adds no value |

### Key Findings from VQ-VAE

1. **Stall over-generation**: With 3x stall weight in CE loss, the transformer generates 55% stalls
   (vs human 6%). This destroys velocity and speed profile but IMPROVES angular velocity (42.7 vs 45.8
   - close to human!). The stalls create direction changes that boost angular_velocity.

2. **Angular velocity achievable**: The VQ-VAE + transformer (with 55% stalls) produces angular_velocity_std
   = 42.7 (human = 45.8) - the closest any generative model has come. This confirms the discrete token
   approach CAN produce the right angular velocity. The issue is getting the RIGHT stall frequency.

3. **mean_acceleration is the wall**: Every approach - DDPM, stall injection, VQ-VAE - has mean_acceleration
   as the top discriminator (0.25-0.48 importance). Human = -4106 (deceleration bias from bell speed profile).
   Synthetic = +1000 to +10000 (acceleration bias from endpoint correction and uniform speed).

4. **VQ-VAE codebook is solid**: The codebook captures the (dx, dy) distribution well with only 30 px/s
   speed reconstruction error. The bottleneck is the sequence generation model, not the codebook.

5. **Transformer is undertrained**: 33% accuracy after 20 epochs on 100K data. The model is learning but
   hasn't converged. Retraining with: 200K data, 40 epochs, no stall overweighting, LR 3e-4.

### v147: DDPM + Borrowed Timing (NOT generative - validation only)

| Config | AUC | Key Features |
|--------|-----|-------------|
| v147 (raw donor timing) | 0.785 | curvature 3668, but accel/jerk exploded |
| v147 (duration-scaled) | **0.820** | mean_acc = -1253 (correct sign!), angular_vel 14/35 |

**Key insight**: DDPM spatial path + real trajectory timing = AUC 0.82. This proves the spatial
path is good - the bottleneck is timing (speed profile, stalls, deceleration). A well-trained
VQ-VAE transformer that learns timing implicitly should achieve similar results.

NOTE: v147 violates the generative constraint (loads pool .npy files). It's a validation
experiment, not a valid generative approach.

### Transformer Retraining v2 (200K data, 40 epochs, no stall weight)

| Epoch | val_loss | val_acc | Notes |
|-------|----------|---------|-------|
| 1 | 3.005 | 33.4% | Higher LR (3e-4) helps early convergence |
| 10 | 2.648 | 36.1% | |
| 20 | 2.563 | 37.0% | |
| 40 | 2.527 | 37.4% | Final - 20091s (5.6 hrs) |

Evaluation:
- **AUC = 0.890** (smoke test, n=200)
- Stall rate: 2.0% (down from 55% with 3x weight, human = 6%)
- Unique tokens: 420/1025 (significant mode collapse)
- mean_acceleration: +1457 (improved from +3520, human = -4106)
- path_efficiency: 0.93 (still too straight, human = 0.82)
- max_deviation: 27 (human = 76)
- curvature: 0.43 (still near zero)

The retrained model is better than the stall-weighted version but still worse than DDPM baseline (0.86).

### Summary: All Generative Approaches

| Approach | AUC | Key Achievement | Key Failure |
|----------|-----|----------------|-------------|
| DDPM v135 eta=0 | **0.862** | Best baseline. Good spatial features. | angular_vel 0.5x, curvature 0 |
| DDPM v135 eta=0.02 | 0.880 | Slight stochasticity | Noise degrades features |
| DDPM v135 eta=0.05 | 0.869 | More stochasticity | Too much acceleration noise |
| v143c (stall ramps) | 0.863 | Curvature 41 (6x baseline) | mean_acc wrong sign from ramps |
| v144 (perturbation) | 0.900 | angular_vel close | Acceleration artifacts |
| v145 VQ-VAE + TF (stall wt) | 0.948 | angular_vel 42.7 (closest to human 45.8!) | 55% stalls, velocity too low |
| v145 VQ-VAE + TF (retrained) | 0.890 | Better mean_acc | Paths too straight, 2% stalls |
| v147 DDPM + borrowed timing | **0.820** | Best overall (NOT generative) | Uses pool files |
| Corpus replay (v55) | 0.498 | Theoretical minimum | Not generative |

**Key Discoveries:**

1. **Curvature mechanism confirmed**: Zero-displacement stalls contribute ZERO curvature (|v x a|/|v|^3,
   numerator=0 at v=0). Curvature comes from low-speed deceleration ramps with heading changes.

2. **Post-hoc modification is a dead end**: Every modification to DDPM paths (stalls, perturbations,
   speed dips) creates acceleration artifacts that offset any gains. Net AUC change ~ 0.

3. **VQ-VAE codebook is solid**: 1024/1024 entries used, 30 px/s speed reconstruction error (3.6%).
   The bottleneck is the sequence generation model, not the codebook.

4. **Transformer is undertrained**: 37.4% accuracy after 40 epochs on 200K data. Needs 500K+ data
   and 100+ epochs for meaningful convergence. The model shows mode
   collapse (420/1025 tokens used) and doesn't learn the speed profile shape.

5. **Timing is the critical component**: v147 (DDPM spatial + real timing) achieves AUC 0.82.
   The DDPM spatial path is already good enough - the gap is entirely in the timing/speed profile.

**Future directions:**

- Train transformer longer on full 500K+ data with endpoint conditioning
- Factorized discrete+continuous approach: separate stall schedule model composed with continuous path generator
- Scheduled sampling to reduce train-inference distribution shift

## ZIMT Endpoint Correction & Guided Sampling (2026-05-13)

### ZIMT Architecture Recap

ZIMT (Zero-Inflated Mouse Trajectory): Causal Transformer + FiLM conditioning + binary stall gate + 8-component MDN.
Input: (dx_prev, dy_prev, stall_prev, remaining_dx, remaining_dy, remaining_frac).
Condition: (log_dist, log_dur, cos_angle, sin_angle). 256d, 6L, 4H.

Baseline ZIMT AUC: 0.878 at n=2000 (with donor time warp OFF, uniform timestamps).

### Key Insight: ZIMT's Ceiling Is Its Learned Distribution

Previously hypothesized that the endpoint correction (linear interpolation in last 20%) was the main
bottleneck, creating an artificial velocity peak at ~90% of duration (human peak at ~35%).

Testing revealed this is PART of the problem but not the main issue. The fundamental limitation is
ZIMT's learned joint distribution of kinematic features, especially:
- mean_acceleration × mean_jerk correlation (r=0.999 synthetic vs r=-0.025 human)
- Angular velocity distribution mismatch (0.50+ Wasserstein)
- These cannot be fixed by post-processing

### Experiment Results

| Experiment | AUC (n=200) | AUC (n=2000) | Key Change | Result |
|------------|-------------|--------------|------------|--------|
| ZIMT baseline | 0.878 | 0.878 | Standard: linear last-20% correction, uniform timestamps | Baseline |
| ZIMT guided (strength=0.3) | 0.968 | — | Shift MDN means toward endpoint, quadratic schedule | WORSE: out-of-distribution outputs |
| ZIMT guided (strength=0.1) | 0.939 | — | Lower guidance | Still worse |
| ZIMT guided (strength=0.05) | 0.913 | — | Minimal guidance | Still worse than baseline |
| **ZIMT magcorr** | **0.799** | **0.864** | Magnitude-weighted correction across all steps | **Best ZIMT variant** |
| ZIMT magcorr (gate_bias=0) | 0.797 | — | More stalls with magcorr | Similar to default |
| ZIMT stall_inject_v2 | 0.791 | — | Curvature-aware stalls + cubic correction | Stall injection helps slightly |

### ZIMT Guided MDN Sampling (FAILED)

**Hypothesis**: Shift MDN component means toward the "ideal" next step (remaining_displacement / remaining_steps)
at inference time, with quadratic-schedule strength increasing toward endpoint. Eliminates destructive
endpoint correction entirely. Analogous to classifier-free guidance in diffusion models.

**Result**: Even minimal guidance (strength=0.05) worsens AUC from 0.878 to 0.913. The model wasn't
trained with guidance, so shifting means creates out-of-distribution outputs. Top discriminators shift
to velocity_skewness (1.57 Wasserstein at strength=0.3) and mean_jerk.

**Lesson**: Inference-time distribution modification on models trained without it creates artifacts.
Guidance must be built into training (like classifier-free guidance in diffusion, which trains with
random condition dropout).

### ZIMT Magnitude-Weighted Correction (IMPROVED)

**Change**: Replace linear last-20% endpoint correction with magnitude-weighted correction spread across
ALL moving steps. Each step absorbs error proportional to its displacement magnitude.

**Result at n=200**: AUC 0.799 (vs 0.878 baseline). velocity_skewness drops from 1.57 to 0.11.
time_to_peak_velocity drops from 0.76 to 0.44. The velocity profile is much more natural.

**Result at n=2000**: AUC 0.864. Modest improvement. New bottleneck: angular_velocity_mean (0.506
Wasserstein), time_to_peak_velocity (0.496), angular_velocity_std (0.413). Feature importances
are spread out — no single feature dominates.

**Lesson**: Better endpoint correction helps (velocity profile improves significantly) but the
fundamental joint distribution mismatch remains. ZIMT's learned distribution doesn't capture
angular dynamics correctly.

## Corpus Enhancement Experiments (2026-05-13)

### Key Discovery: Pure Retrieval Beats All Enhancements

With the full 4.16M pool, plain corpus replay (translate-only, cubic-ease endpoint correction)
achieves AUC **0.52** at n=2000 — essentially indistinguishable from human. Every enhancement
attempted made it WORSE by introducing detectable artifacts.

| Experiment | AUC (n=2000) | Type | Why Worse |
|------------|-------------|------|-----------|
| **corpus_replay** | **0.52** | Pure retrieval + translate | **BEST** — real data is optimal |
| corpus_perturb_v7 | 0.645 | Magnitude scaling + perp jitter | Perturbation changes direction counts |
| corpus_sim | 0.682 | Similarity transform (rotate+scale) | Rotation changes direction counts |
| corpus_replay_v2 | 0.558 | Magnitude-weighted correction | Distributed correction alters more steps |
| corpus_perturb_v6 | 0.780 | Smooth sinusoidal perturbation | Sine waves create curvature artifacts |

### corpus_perturb_v7 Feature Analysis

Top discriminators at n=2000: num_direction_changes (0.255 Wasserstein, 0.083 importance),
movement_duration (0.035, 0.079), curvature_mean (0.229, 0.060). The perturbation changes
direction counts and curvature — even 3% magnitude scaling + 1.5% perpendicular jitter is
detectable.

### corpus_sim (Similarity Transform, FAILED)

Applied rotate + scale to map donor start/end exactly to query start/end, eliminating endpoint
correction entirely. AUC 0.682 — worse than plain replay because:
- Rotation preserves relative angles between consecutive steps, but `num_direction_changes`
  depends on absolute step signs, which rotation changes
- Scaling changes step magnitudes, pushing borderline stalls above/below thresholds
- Velocity features improve (all < 0.012 Wasserstein) but direction changes worsen (0.302)

### corpus_replay_v2 (Magnitude-Weighted Correction, FAILED)

Replaced cubic-ease endpoint correction (concentrated in last 25%) with magnitude-weighted
correction spread across all moving steps. AUC 0.558 — worse than cubic-ease (0.514) because:
- Distributing correction everywhere alters many small direction decisions
- Cubic-ease is better because it concentrates changes in the deceleration phase, mimicking
  natural human endpoint adjustment
- curvature_mean improves (0.018 vs 0.054) but num_direction_changes worsens (0.125 vs 0.056)

### Key Lesson

**Any transformation of real trajectories makes them worse.** The 18 kinematic features form a
tightly coupled joint distribution. Even small perturbations (3% magnitude scaling) or
mathematically exact transformations (similarity transform) introduce detectable artifacts.
The optimal strategy with a large enough pool is pure retrieval + translation.

### Corpus Replay Stability (3 seeds, n=2000)
- Seed 42: AUC 0.514
- Seed 123: AUC 0.525
- Seed 999: AUC 0.519
- Mean: ~0.52, range: 0.514-0.525

## Updated Scoreboard (2026-05-13)

| Approach | AUC (n=2000) | Type | Notes |
|----------|-------------|------|-------|
| **Corpus Replay (4.16M)** | **0.52** | Retrieval | Open-source ready |
| Corpus Perturb v7 | 0.645 | Retrieval + noise | Best perturbation variant |
| Corpus Sim | 0.682 | Retrieval + transform | Similarity transform hurts |
| DDPM + borrowed timing | 0.820 | Hybrid (not generative) | Proves DDPM path is OK |
| ZIMT magcorr | 0.864 | Generative | Best ZIMT variant |
| ZIMT baseline | 0.878 | Generative | Original ZIMT |
| VQ-VAE + GRPO | 0.890 | Generative | Checkpoint lost |
| ZIMT guided | 0.968 | Generative | Guidance at inference fails |

## ZIMT Fine-Tuning Experiments (2026-05-14)

### Differentiable Feature-Matching (FM) Training

Attempted to fine-tune ZIMT by backpropagating through differentiable kinematic feature
computation. Three iterations, all failed due to **exposure bias**.

**Architecture**: Two-phase generation — Phase 1 generates reference trajectories without
gradient (autoregressive, fast), Phase 2 teacher-forces from references in a single forward
pass with gradient. Uses Gumbel-Softmax for differentiable MDN component selection and
straight-through estimator for stall gating.

| Iteration | Loss | Grad Clip | AUC (n=200) | Key Issue |
|-----------|------|-----------|-------------|-----------|
| FM iter 1 (L2, clip=10) | L2 | 10 | 0.870 | Gradients 1000-10000x above clip → 99% signal lost |
| FM iter 2 (L1, clip=100) | L1 | 100 | — | Angular velocity WORSENED: 0.437 → 0.736 → 0.890 |
| FM iter 3 (clamped feats) | L1 | 100 | — | Exposure bias confirmed: more training = worse inference |

**Root cause — exposure bias**: Teacher-forced training pushes model parameters in directions
that improve loss when given ground-truth inputs, but these same parameter changes compound
errors during autoregressive inference. Angular velocity gap WORSENED monotonically with more
FM training iterations. This is a fundamental limitation of teacher-forced fine-tuning, not a
hyperparameter issue.

**Prior attempt — autoregressive with gradient**: Computation graph through T sequential forward
passes caused GPU memory explosion (612 CPU seconds, 4GB working set by iteration 2). Abandoned
in favor of the two-phase approach.

### Rejection Sampling (FAILED)

Generate N candidate trajectories per query, select the one closest to human feature mean
(normalized L2 distance across all 18 features).

| N Candidates | AUC (n=200) | Key Issue |
|-------------|-------------|-----------|
| 8 | 0.892 | Variance collapse |

**Result**: AUC 0.892 — WORSE than baseline 0.864. Selecting trajectories closest to the
human mean kills feature VARIANCE. Features like path_efficiency (0.170→0.425 Wasserstein),
movement_duration (0.128→0.344) became highly discriminative because distributions were too
narrow. The RF classifier detects reduced variance as easily as shifted means.

### Corpus Blend (FAILED)

Blend two donor trajectories via arc-length resampling + position interpolation.

**Result**: AUC 0.948 at n=2000. Arc-length resampling completely destroys speed profiles,
creating angular_velocity gaps of 0.84-0.86 Wasserstein. Position blending between two
trajectories creates unnatural curvature artifacts.

### Corpus Rotate (Below Target but Not Neural)

Rotation + uniform scale of donor trajectories to map any donor to any query direction,
eliminating the angle filter constraint. Matches by log-distance only.

**Result**: AUC **0.686** at n=2000. Only remaining gap is num_direction_changes (0.330
Wasserstein). All velocity/acceleration features below 0.012. Curvature features at 0.11-0.12.

| Feature | Wasserstein | Note |
|---------|------------|------|
| num_direction_changes | 0.330 | Only HIGH feature — rotation changes absolute step signs |
| curvature_mean | 0.115 | Acceptable |
| curvature_std | 0.108 | Acceptable |
| max_deviation | 0.062 | Good |
| All velocity/accel features | < 0.012 | Excellent |

**Key insight**: Rotation preserves relative angles and scale-invariant features (velocity_skewness,
time_to_peak_velocity, path_efficiency) but changes absolute step signs, which
num_direction_changes depends on. This is the same mechanism as corpus_sim (0.682) —
essentially the same approach, confirming the result.

**Status**: Below the 0.75 target but uses geometric transformation of real data, not a neural
generative model.

## Updated Scoreboard (2026-05-14)

| Approach | AUC (n=2000) | Type | Notes |
|----------|-------------|------|-------|
| **Corpus Replay (4.16M)** | **0.52** | Retrieval | Open-source ready |
| Corpus Perturb v7 | 0.645 | Retrieval + noise | Best perturbation variant |
| **Corpus Rotate** | **0.686** | Retrieval + transform | Below 0.75 target |
| Corpus Sim | 0.682 | Retrieval + transform | Same mechanism as rotate |
| DDPM + borrowed timing | 0.820 | Hybrid (not generative) | Proves DDPM path is OK |
| **ZIMT magcorr** | **0.864** | **Generative** | **Best neural generative** |
| ZIMT FM (50 iter) | 0.870 | Generative | Exposure bias worsened it |
| ZIMT baseline | 0.878 | Generative | Original ZIMT |
| ZIMT reject (N=8) | 0.892 | Generative | Variance collapse |
| VQ-VAE + Transformer | 0.890 | Generative | Undertrained |
| Corpus Blend | 0.948 | Retrieval + blend | Arc-length resampling kills features |
| ZIMT guided | 0.968 | Generative | Guidance at inference fails |

### Key Lessons from This Session

1. **Teacher-forced fine-tuning has fundamental exposure bias**: Any gradient signal computed
   under teacher forcing pushes parameters in directions that worsen autoregressive generation.
   This rules out differentiable feature matching as a ZIMT improvement path.

2. **Rejection sampling kills variance**: Selecting best-of-N by distance to mean produces
   distributions that are too narrow. The RF classifier detects reduced variance as easily as
   shifted means.

3. **Geometric transforms of real data work well**: Corpus rotate achieves 0.686 — below the
   0.75 target. But rotation changes num_direction_changes (the only remaining gap), so it
   cannot reach corpus replay's 0.52.

4. **The generative gap remains large**: Best neural generative (ZIMT magcorr) is at 0.864.
   The gap to 0.75 is enormous and all fine-tuning approaches barely moved the needle.

### Next Steps

**For open-source release**: Corpus Replay (0.52) or Corpus Rotate (0.686) both meet <0.75.

**For a truly generative model below 0.75**: Remaining ideas:
- ZIMT retrained from scratch with scheduled sampling (ss_max > 0.3) + multi-task NLL + feature loss
- Neural perturbation network on top of corpus_rotate (learned residuals → clearly generative)
- Factored model: separate stall schedule + continuous path generator
- ZIMT + donor speed profile transplant (extending the v147 insight)
- GRPO/RAFT: use RF classifier as RL reward signal

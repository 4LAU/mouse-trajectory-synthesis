# Experiment Log

Full chronological experiment log. 155+ experiments from parametric baselines through corpus replay to generative models. AUC = OOB Random Forest, lower is better (0.50 = indistinguishable from human).

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

## Proposed Generative Architectures (2026-05-15)

Literature review across handwriting synthesis, speech/audio generation, robotics, and motion
capture identified 9 novel approaches not yet attempted. Ranked by how directly they address
our two known root causes: (1) the stall representation gap (6% of human timesteps are exact
zero displacement — no continuous model produces this), and (2) exposure bias in autoregressive
training (teacher-forced fine-tuning worsens free-running inference).

### Tier 1 — Directly addresses a known root cause

**1. Hybrid Discrete-Continuous Diffusion (CANDI-style)**

Ref: CANDI (arXiv 2510.22510), CDTD (arXiv 2312.10431), MissHDD (arXiv 2511.14543)

Runs two coupled processes in a single denoiser: discrete masking for stall/no-stall decisions,
continuous Gaussian diffusion for (dx, dy) displacement. The discrete channel assigns real
probability mass to exact-zero events; the continuous channel handles smooth kinematics. One
shared neural network learns the coupling between stall decisions and surrounding motion
(deceleration ramps, heading changes at stall boundaries).

- **Why novel vs. what we tried**: Our DDPM/CFM kept everything continuous (can't produce exact
  zeros). Our VQ-VAE quantized everything (loses continuous precision). CANDI quantizes only the
  stall decision and keeps displacement continuous.
- **Additional opportunity**: The RF classifier could serve as a guidance signal during sampling
  (classifier-based guidance), steering trajectories toward human-like feature distributions
  without RL/GRPO.
- **Risk**: Implementation complexity. Need to handle temporal dissonance between discrete and
  continuous corruption schedules.

**2. Action Chunking / Chunk-Level Diffusion**

Ref: ACT (Zhao et al., RSS 2023), Diffusion Policy (Chi et al., RSS 2023)

Generate coherent 25-step (~200ms) chunks via diffusion, then sequence chunks autoregressively.
Each chunk is produced holistically (natural kinematic profiles internally), conditioned on the
tail of the previous chunk. Compounding error drops from 200 decision points (per-step AR) to
~8 (per-chunk AR).

- **Why novel vs. what we tried**: ZIMT generates one step at a time (exposure bias). Our
  DDPM generates the full 200-step trajectory at once (smooth conditional means). Chunking is
  the middle ground — holistic within-chunk generation avoids per-step compounding, while
  chunk-level sequencing avoids global smoothing.
- **Can reuse**: Existing DDPM U-Net backbone, adapted for chunk-length inputs with overlap
  conditioning.
- **Risk**: Chunk boundaries may create artifacts. Chunk size is a key hyperparameter (too
  small = exposure bias returns, too large = smoothing returns).

**3. SoundStorm/MaskGIT Masked Iterative Decoding on VQ-VAE Codebook**

Ref: SoundStorm (Borsos et al., 2023), MaskGIT (Chang et al., CVPR 2022)

Uses our existing VQ-VAE codebook (validated: 30 px/s error, 1024/1024 entries used) with a
new generation paradigm. Instead of left-to-right autoregressive (which caused mode collapse —
420/1024 tokens used), start with all tokens masked and iteratively unmask in confidence order.
A bidirectional Transformer sees the full sequence context including both endpoints.

- **Why novel vs. what we tried**: Our VQ-VAE + autoregressive Transformer generated
  left-to-right (mode collapse, error compounding). This is bidirectional — the model sees
  both start and end when deciding each token. Eliminates endpoint correction entirely.
  Easy tokens (mid-ballistic) fill first; hard tokens (stall transitions) refine with full
  context.
- **Can reuse**: Existing VQ-VAE codebook and tokenization pipeline.
- **Risk**: Masked generation may not capture temporal causality as well as autoregressive.
  Confidence-based ordering may not align with trajectory structure.

### Tier 2 — Strong theoretical fit, moderate risk

**4. Decoupled Shape + Speed Generation (Two-Thirds Power Law)**

Ref: Two-thirds power law (Lacquaniti et al., 1983), motor control literature

Generate trajectory in two stages: (a) spatial path shape (sequence of heading changes or
curvature values), (b) speed profile along that path conditioned on curvature. The two-thirds
power law (v = gamma * kappa^(-1/3)) provides a physics prior: humans slow down at curves.

- **Why novel**: All our models generate (x, y, t) jointly. This factored approach isolates
  curvature (our hardest feature) into a dedicated shape model, with speed constrained by
  physics. The v147 experiment (DDPM spatial + real timing = AUC 0.82) validates the
  factored premise — the spatial path is already good enough if timing is right.
- **Risk**: Reintegrating shape + speed back to (x,y,t) may introduce artifacts. Stall
  events need separate handling (speed = 0 violates the power law).

**5. ProDMP (Probabilistic Dynamic Movement Primitives)**

Ref: ProDMP (Li et al., CoRL 2023, arXiv 2210.01531)

Represent each trajectory as ~30 basis function weights rather than 200 timesteps. The DMP
differential equation guarantees smooth velocity profiles with goal convergence. A neural
network (VAE or diffusion) generates the basis function weights conditioned on (start, end,
distance).

- **Why novel**: Our parametric models (min-jerk, sigma-lognormal) used fixed functional
  forms. ProDMP's basis functions are learned from data and capture the full covariance
  structure. Reduces the generation problem from 200-D sequence to ~30-D vector.
- **Risk**: DMP produces smooth trajectories by construction — stalls need augmentation via
  a separate discrete model. Integration of smooth DMP + stall injection is the same
  hybrid challenge we've hit before.

**6. Autoregressive Normalizing Flow (MoGlow-style)**

Ref: MoGlow (Alexanderson et al., SIGGRAPH Asia 2020)

Invertible neural network that maps noise to data, trained with exact maximum likelihood.
Autoregressive (frame-by-frame) with LSTM temporal context. No mode collapse risk (exact
likelihood), no adversarial training instability, naturally handles variable-length sequences.

- **Why novel**: Architecturally distinct from everything tried. Not diffusion, not an MDN,
  not a VAE. Exact likelihood avoids the ELBO gap. Invertibility means every training example
  maps to a unique latent code — no information loss.
- **Demonstrated**: Controllable motion generation conditioned on desired direction, directly
  analogous to our (start, end) conditioning.
- **Risk**: Invertibility constraint limits expressiveness of each layer. Training can be slow.

### Tier 3 — Interesting, higher risk or incremental

**7. Neural SDE with Signal-Dependent Noise**

Ref: Stable Neural SDEs (ICLR 2024, arXiv 2402.14989), Harris & Wolpert 1998

Models trajectory as a stochastic dynamical system: dx = f(x,t)dt + g(x,t)dW. The noise
coefficient g scales with speed (matching known neuroscience: motor noise is signal-dependent).
Produces structured variability where high-speed segments have proportional noise and low-speed
segments have directional uncertainty.

- **Why novel**: Our DDPM uses diffusion as a generative mechanism (noise→data). Neural SDE
  uses diffusion as a dynamics model (the trajectory evolves as a stochastic process). The
  noise is structured by the learned g(x,t), not isotropic Gaussian.
- **Risk**: Standard SDE can't produce exact-zero stalls. Would need a Jump-SDE variant
  (Poisson events trigger stall mode). Complex to implement and train.

**8. Energy-Guided Diffusion (EnergyMoGen-style)**

Ref: EnergyMoGen (CVPR 2025, arXiv 2412.14706), Energy Matching (NeurIPS 2025)

Define energy functions for feature groups (velocity stats, curvature, angular velocity).
During diffusion sampling, energy gradients steer each denoising step toward human-like
feature distributions. Multiple energies compose via learned fusion weights.

- **Why novel**: Our feature-matching fine-tuning failed from exposure bias in the
  autoregressive loop. Energy guidance in diffusion has no autoregressive loop, so no
  exposure bias. Our rejection sampling operated post-hoc and killed variance. Energy
  guidance steers the distribution during generation, preserving variance.
- **Risk**: Requires differentiable approximations of the 18 kinematic features (curvature
  involves division by |v|^3 — numerically unstable near stalls). Conflicting energy
  gradients may cause oscillation.

**9. Mamba/S4 State-Space Backbone (Drop-in for ZIMT)**

Ref: Mamba (Gu & Dao, 2023), MamTra (arXiv 2603.12342)

Replace ZIMT's 6-layer Transformer with a selective state-space model. Linear-time inference,
selective state propagation (learn what to remember/forget). The state space mechanism is
a learned linear dynamical system — natural fit for trajectory dynamics.

- **Why novel**: Different inductive bias than attention. Selective state propagation may
  better capture velocity/acceleration momentum than position-based attention.
- **Risk**: Backbone swap is low-effort but likely incremental. Same MDN output head, same
  training procedure — unlikely to close the 0.864→0.75 gap alone.

### Implementation Priority

For a generative model below AUC 0.75, recommended order:
1. ~~Action Chunking (#2) — simplest, reuses DDPM backbone, directly solves exposure bias~~ **TRIED, FAILED (AUC 0.957)**
2. ~~SoundStorm on VQ-VAE (#3) — reuses existing codebook, novel generation paradigm~~ **TRIED, FAILED (AUC 0.996)**
3. CANDI hybrid diffusion (#1) — most theoretically sound, addresses stalls without VQ-VAE quantization bottleneck
4. Decoupled Shape + Speed (#4) — partially validated by v147 (DDPM spatial + real timing = 0.82)
5. MoGlow (#6) — exact likelihood, no mode collapse, but still autoregressive

## Chunk Diffusion Experiment (2026-05-15)

### Architecture

Action chunking / chunk-level diffusion, inspired by ACT (Zhao et al., RSS 2023) and
Diffusion Policy (Chi et al., RSS 2023). Generates 25-step (~200ms) chunks via DDPM
with DDIM sampling, sequenced autoregressively.

- **Model**: 1D U-Net (48→96→192 channels), ~1.17M params
- **Input**: Noisy chunk (B, 25, 3) — dx, dy, stall_logit
- **Conditioning**: Global (log_dist, log_dur, cos_a, sin_a) + Local (rem_dx, rem_dy, rem_frac, progress, cum_dx, cum_dy) + Context encoder (5-step tail of previous chunk)
- **Diffusion**: 200 steps cosine schedule, x_0 prediction, DDIM sampling (50 steps, eta=0.3)
- **Stall modeling**: Joint 3rd channel (stall logit +3.0/-3.0), sigmoid threshold at inference
- **Training data**: 1,740,707 overlapping chunks (stride 20) from 499,955 trajectories

### Training

80 epochs, AdamW lr=3e-4, CosineAnnealing, batch_size=256, AMP, grad clip 1.0.
Best checkpoint: epoch 67, val_loss 0.148.

### Results

**AUC 0.957 at n=2000 — FAILED.**

| Feature | Wasserstein | Note |
|---------|------------|------|
| velocity_skewness | 1.764 | *** Catastrophic — no global velocity awareness |
| angular_velocity_mean | 0.775 | *** HIGH |
| angular_velocity_std | 0.655 | *** HIGH |
| time_to_peak_velocity | 0.537 | *** HIGH |
| path_efficiency | 0.258 | Moderate |
| num_direction_changes | 0.254 | Moderate |
| All velocity/accel features | < 0.063 | Good |

RF 5-fold CV: 0.957, GBM 5-fold CV: 0.964. Feature importance spread across
velocity_skewness (0.149), angular_velocity_mean (0.138), mean_jerk (0.082),
angular_velocity_std (0.080).

### Why It Failed

**No global trajectory awareness.** Each 25-step chunk generates in isolation — the
5-step context and local conditioning (remaining_frac, progress) are insufficient to
convey where in the global velocity profile the chunk sits. Chunk at 20% progress
(peak acceleration) looks identical to chunk at 80% (deceleration tail) because the
model has no representation of the overall trajectory shape.

velocity_skewness = 1.764 Wasserstein is the smoking gun: this feature measures
the asymmetry of the full-trajectory speed distribution, which requires knowing the
complete velocity envelope. No individual chunk has enough context for this.

**Lesson**: The exposure-bias / global-awareness tradeoff has no middle ground.
Per-step AR (ZIMT) has exposure bias but preserves global velocity profile shape.
Full-trajectory diffusion (DDPM) preserves global shape but over-smooths.
Per-chunk AR is the worst of both: no global awareness AND chunk-boundary artifacts.

This confirms that the next architecture must have full-sequence context at every
generation step — pointing to bidirectional approaches (SoundStorm/MaskGIT).

## Frontier Research Synthesis (2026-05-15)

Deep literature survey across masked generative models, robotics action generation,
handwriting/speech synthesis, and our own failure mode analysis. Key findings below.

### Three-Pattern Failure Analysis

All three persistent gaps in our generative models trace to a single mechanism:
incorrect generation of the decelerate → hold → change heading → accelerate pattern
at stall boundaries.

**Pattern 1 — velocity_skewness**: Requires global trajectory awareness. The velocity
profile is asymmetric (peak at 35%, long decel tail). Full-trajectory models preserve
this; chunk and per-step models destroy it because no local window encodes the global
velocity envelope.

**Pattern 2 — angular_velocity**: Comes from heading changes at stall boundaries
(decel→hold→heading change→reaccel), NOT from smooth curves. The only model that
matched human angular velocity (42.7 vs 45.8) was VQ-VAE with discrete stall tokens
at 55% stall rate. The stalls were wrong (55% vs 6%) but the angular velocity
mechanism was correct.

**Pattern 3 — accel-jerk decorrelation**: Human r=-0.025, all synthetic r=0.999. Jerk
spikes at stall transitions (abrupt acceleration changes between smooth ballistic
segments) break the derivative relationship. Only exact-zero stalls followed by
heading changes produce these spikes. Continuous models always have smooth
acceleration→jerk relationships.

### Key Papers from Frontier Research

| Paper | Venue | Relevance |
|-------|-------|-----------|
| **MoMask** (Guo et al.) | CVPR 2024 | MaskGIT on VQ motion tokens → SOTA human motion generation. Closest analogue to our problem. |
| **MAR** (Li et al.) | NeurIPS 2024 | Masked autoregressive with per-token diffusion loss. Continuous tokens without VQ. |
| **MaskGCT** | ICLR 2025 | Masked generative codec transformer for speech. Non-autoregressive, full bidirectional context. |
| **CARP** (Desai et al.) | Dec 2024 | Autoregression across SCALE not TIME — generate coarse shape, then refine. Matches diffusion quality at AR speed. |
| **JointDiff** | ICLR 2026 | Joint continuous+discrete diffusion for trajectories with discrete events. |
| **IMPACT** | CVPR 2025 | Iterative masked prediction for action chunking. Extends MaskGIT to robotics. |
| **MDG** | 2025 | Masked discrete generation for molecular conformations. Same paradigm, different domain. |
| **CosyVoice 2** | 2025 | Finite scalar quantization + masked generation for speech. Shows FSQ can replace VQ-VAE. |
| **DiffInk** | 2025 | Diffusion for handwriting with variable-length sequences. Similar kinematic requirements. |
| **PALLE** | 2025 | Predictive autoregressive latent language for motion. Multi-scale latent codes. |

### Why SoundStorm/MaskGIT Is the Recommended Next Step

The synthesis across all research fronts points to masked iterative decoding on
our existing VQ-VAE codebook as the highest-probability path:

1. **Full bidirectional context**: Unlike left-to-right AR (ZIMT, VQ-VAE transformer),
   the model sees the entire sequence including both endpoints at every decoding step.
   Preserves global velocity profile shape (Pattern 1).

2. **Discrete stall tokens**: Reuses our validated VQ-VAE codebook (1024 entries,
   100% utilization, stall token 0). Only model family that matched human angular
   velocity was discrete tokens (Pattern 2).

3. **Confidence-ordered unmasking**: Easy tokens (mid-ballistic) fill first; stall
   boundary tokens fill last with maximum context. Hardest decisions get the most
   information (Pattern 3).

4. **No endpoint correction**: Start and end tokens always unmasked. Model learns to
   fill the middle such that cumulative displacements naturally sum to the correct
   endpoint.

5. **Existing assets**: VQ-VAE codebook and 32.4M tokenized sequences already
   validated. Only the sequence model needs to be built.

**Fallback**: CARP coarse-to-fine (generate 10 keypoints, then refine to full
resolution). Addresses global-vs-local tension from a different angle.

## Updated Scoreboard (2026-05-15)

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
| VQ-VAE + Transformer | 0.890 | Generative | Undertrained, mode collapse |
| ZIMT reject (N=8) | 0.892 | Generative | Variance collapse |
| Corpus Blend | 0.948 | Retrieval + blend | Arc-length resampling kills features |
| **Chunk Diffusion** | **0.957** | **Generative** | **No global awareness** |
| ZIMT guided | 0.968 | Generative | Guidance at inference fails |

## SoundStorm / MaskGIT on VQ-VAE (2026-05-16)

### SoundStorm Training

Trained a masked bidirectional transformer (SoundStormTransformer) on our existing VQ-VAE
token sequences (32.4M tokens, 1025 vocab including stall token 0).

| Config | Value |
|--------|-------|
| Architecture | 6-layer Transformer, 256d, 8 heads, 1024 FFN |
| Parameters | 6,088,705 |
| Training data | 500K tokenized trajectories (VQ-VAE codebook) |
| Conditioning | FiLM on (log_dist, log_dur, cos_angle, sin_angle) |
| Training | 16 epochs, AdamW lr=3e-4, cosine schedule, batch 256 |
| Best checkpoint | Epoch 16, val_loss 2.4722, val_acc 49.3% |

### SoundStorm Evaluation

| Method | AUC (n=2000) | Key Issue |
|--------|-------------|-----------|
| From-scratch generation | 0.996 | Cold-start collapse: all-MASK → model predicts stall (37% confidence) → locks in all-stall |
| generate_refine (donor init) | 0.996 | Spatial shape wrong despite iterative refinement |
| Soft decoding (expected displacement) | 0.997 | Probability-weighted displacement doesn't help |
| Donor token perturbation | 0.914 | VQ-VAE quantization bottleneck — accumulated tokens create wrong path shapes |

### Root Cause: VQ-VAE Quantization Bottleneck

The fundamental failure is the VQ-VAE displacement tokenization, not the SoundStorm architecture.
When discrete displacement tokens are accumulated to reconstruct paths, the 1024-entry codebook
quantization destroys spatial smoothness. Key diagnostics:

- Curvature: wrong (quantized steps create artificial curvature patterns)
- Angular velocity: wrong (quantized directions don't match continuous)
- Path efficiency: wrong (quantized path length differs from continuous)
- Donor token perturbation (AUC 0.914): even starting from real donor tokens and only replacing
  a few with SoundStorm predictions gives 0.914 — the quantization is the bottleneck, not the
  sequence model

**Lesson**: VQ-VAE-based generation is a dead end for this task. The codebook is validated for
reconstruction (30 px/s speed MAE) but accumulating discrete tokens into paths creates
fundamentally wrong kinematic profiles. Future approaches must keep displacements continuous.

### Cold-Start Collapse Analysis

When generating from all-MASK tokens, the model's first prediction at every position is stall
token 0 with 37% confidence. Confidence-based unmasking then reveals all positions simultaneously
as stall tokens, producing zero-displacement trajectories. Attempts to fix:

- Random-order unmasking for first 3 rounds: didn't break the stall attractor
- Higher temperature (2.5) + top-p (0.95): more diverse tokens but still wrong spatial shape
- CFG (scale 2.0): amplified endpoint conditioning but spatial coherence still wrong

### atan2(0,0) Discovery

During investigation of corpus rotate's `num_direction_changes` mismatch (Wasserstein 0.32),
discovered the root cause: `atan2(0,0)=0` during stall periods. Rotation shifts movement
angles but stall angles always return 0, changing sign patterns at stall→movement transitions.
This adds ~8 direction changes on average. Confirmed:
- Direction changes correlate with trajectory length (r=0.586), NOT stall fraction (r=0.019)
- Pool and evaluation set have nearly identical direction change distributions (Wasserstein 0.059)
- The mismatch comes from the rotation transform itself, not sampling bias
- This is inherent to the rotation approach and cannot be fixed without changing the transform

### Enhanced Corpus Rotate Parameter Sweep

After SoundStorm failed, reverted experiments/soundstorm.py to enhanced corpus rotate with
raw donor coordinates (no VQ-VAE). Systematic parameter sweep:

| Config | AUC (n=2000) | Notes |
|--------|-------------|-------|
| K=5 (distance match only) | 0.696 | Baseline corpus rotate |
| K=20 | 0.682 | More diversity helps |
| K=50 | **0.670** | **Best** |
| K=100 | 0.675 | Diminishing returns |
| K=unlimited | 0.678 | Too much diversity |
| K=50 + length matching (ratio 0.3) | 0.717 | Length matching HURTS |
| K=50 + stall jitter (directional) | 0.763 | Curvature artifacts |
| K=50 + stall jitter (random) | 0.799 | Worse curvature artifacts |
| K=50 + Gaussian smoothing | — | Destroyed angular velocity and curvature |
| K=50 + pixel rounding | — | angular_velocity 0.55 Wass, curvature 0.33 Wass |

Best enhanced corpus rotate: **AUC 0.670** at K=50 with distance matching only.
Still not a neural generative approach.

## Updated Scoreboard (2026-05-16)

| Approach | AUC (n=2000) | Type | Notes |
|----------|-------------|------|-------|
| **Corpus Replay (4.16M)** | **0.52** | Retrieval | Translate-only, floor |
| Corpus Perturb v7 | 0.645 | Retrieval + noise | Best perturbation variant |
| **Enhanced Corpus Rotate (K=50)** | **0.670** | Retrieval + transform | Best rotation variant |
| Corpus Rotate (K=5) | 0.686 | Retrieval + transform | Less donor diversity |
| Corpus Sim | 0.682 | Retrieval + transform | Same mechanism as rotate |
| DDPM + borrowed timing | 0.820 | Hybrid (not generative) | Proves DDPM path is OK |
| **ZIMT magcorr** | **0.864** | **Generative** | **Best neural generative** |
| ZIMT FM (50 iter) | 0.870 | Generative | Exposure bias worsened it |
| ZIMT baseline | 0.878 | Generative | Original ZIMT |
| VQ-VAE + Transformer | 0.890 | Generative | Undertrained, mode collapse |
| ZIMT reject (N=8) | 0.892 | Generative | Variance collapse |
| SoundStorm donor perturb | 0.914 | Generative | VQ-VAE quantization bottleneck |
| Corpus Blend | 0.948 | Retrieval + blend | Arc-length resampling kills features |
| **Chunk Diffusion** | **0.957** | **Generative** | **No global awareness** |
| ZIMT guided | 0.968 | Generative | Guidance at inference fails |
| SoundStorm from-scratch | 0.996 | Generative | Cold-start stall collapse |
| SoundStorm generate_refine | 0.996 | Generative | Spatial shape wrong |
| SoundStorm soft decode | 0.997 | Generative | No improvement |

### Architectures Tried: 3 of 9 — All Failed

| # | Architecture | Status | AUC | Failure Mode |
|---|-------------|--------|-----|-------------|
| 2 | Action Chunking | FAILED | 0.957 | No global velocity awareness |
| 3 | SoundStorm/MaskGIT | FAILED | 0.996 | VQ-VAE quantization bottleneck |
| 1 | CANDI hybrid diffusion | **NEXT** | — | Addresses stalls without VQ-VAE |
| 4 | Decoupled Shape+Speed | Untried | — | Partially validated by v147 |
| 5 | ProDMP | Untried | — | Smooth by construction, needs stall augmentation |
| 6 | MoGlow | Untried | — | Exact likelihood, but still AR |
| 7 | Neural SDE | Untried | — | Can't produce exact stalls |
| 8 | Energy-Guided Diffusion | Untried | — | Numerically unstable near stalls |
| 9 | Mamba | Untried | — | Likely incremental over ZIMT |

## CANDI Hybrid Discrete-Continuous Diffusion (2026-05-26)

### Architecture

Joint Gaussian diffusion on continuous channels + absorbing-state masking on binary stall labels,
in a single Transformer denoiser. Two output heads share a backbone that learns the coupling
between stall decisions and displacement dynamics.

| Config | Value |
|--------|-------|
| Architecture | 6-layer Transformer, 256d, 4 heads, 1024 FFN, FiLM conditioning |
| Parameters | 5,762,051 |
| Input channels | 4: (ch0, ch1) continuous + stall_state + mask_flag |
| Output | 2-channel continuous head + 1-channel stall logit |
| Diffusion | 1000-step cosine schedule, DDIM sampling (50 steps) |
| Discrete masking | Absorbing-state, mask_prob = 1 - sqrt(alpha_bar_t) |
| Conditioning | FiLM on (log_dist, log_dur, cos_angle, sin_angle) with 10% dropout for CFG |
| Max seq len | 128 (covers 89% of trajectories, 4x less attention compute vs 256) |

### Training: Cartesian (dx, dy) Representation

First trained on distance-normalized (dx,dy) displacements.

| Config | Value |
|--------|-------|
| Data | 200K trajectories (from 500K), 90/10 train/val split |
| Training | 20 epochs, AdamW lr=3e-4, CosineAnnealing, batch 128, AMP |
| data_std | 0.055, data_scale = 18.1 |
| Best checkpoint | Epoch 20, val_cont=0.264, val_disc=0.112 |
| Time per epoch | ~108s on RTX 4070 |

### Cartesian Evaluation

| Config | AUC (n=200) | AUC (n=2000) | Key Issue |
|--------|-------------|-------------|-----------|
| CFG=2.0, eta=0.0 | 0.863 | 0.950 | angular_velocity_std 0.34 Wass |
| CFG=1.0, eta=0.0 | 0.841 | 0.948 | Similar |
| CFG=0.0, eta=0.0 | 0.856 | — | — |
| CFG=2.0, eta=1.0 | 0.889 | — | Stochastic adds noise |

**200-sample AUC is unreliable** — Cartesian showed 0.86 at n=200 but 0.95 at n=2000.
The RF needs enough data to detect consistent subtle artifacts.

**Key discriminators at n=2000**: angular_velocity (0.34 Wasserstein), max_deviation (0.28),
path_efficiency (0.25), curvature (0.19/0.24). All path-shape features — the model gets
velocity/acceleration magnitudes right but direction sequences wrong.

### Root Cause: Cartesian (dx,dy) Can't Capture Angular Dynamics

Diffusion on raw (dx,dy) produces displacements that are individually reasonable but whose
direction sequence doesn't match human biomechanics. Angular velocity (rate of direction change)
and curvature are structural properties of the direction sequence that Gaussian diffusion
in Cartesian space can't capture.

Post-processing attempts all failed:
- Uniform filter smoothing: AUC 0.97-0.99 (destroys velocity structure)
- Cubic spline resampling: AUC 0.94-0.99 (same issue)
- DH_SCALE adjustment: no improvement at n=2000

### Training: Polar (speed, delta_heading) Representation

Switched to polar representation where angular velocity is a direct model output.

| Config | Value |
|--------|-------|
| Representation | (speed, delta_heading) instead of (dx, dy) |
| Data conversion | Inline in DataLoader: atan2 → heading → diff → wrap to [-π,π] |
| Stall handling | Carry-forward heading during stalls (avoids atan2(0,0)=0 artifact) |
| Speed std | 0.074 |
| Delta heading std | 0.428 |
| Separate scaling | Each channel scaled to unit variance independently |

Two training runs:

| Run | Epochs | Best val_cont | Best val_disc | Notes |
|-----|--------|--------------|--------------|-------|
| 20 epochs | 20 | 0.458 (ep 18) | 0.109 | First polar run |
| 30 epochs | 30 | 0.423 (ep 24) | 0.106 | Slower LR schedule helps |

### Polar Evaluation

| Config | AUC (n=200) | AUC (n=2000) | Key Observation |
|--------|-------------|-------------|-----------------|
| 20ep, CFG=1.0 | 0.810 | 0.919 | Beats Cartesian (0.950) |
| 30ep, CFG=1.0 | 0.881 | 0.917 | Barely better |
| **30ep, CFG=0.0** | **0.720** | **0.852** | **New best — no guidance** |
| 30ep, CFG=2.0 | 0.815 | 0.922 | CFG hurts |
| 30ep, CFG=0.0, eta=0.7 | — | 0.862 | Stochastic adds noise |

### Critical Bug Fix: Endpoint Correction in Polar Mode

The first polar evaluation code computed endpoint correction weights from wrongly-scaled raw
model output (dividing both channels by speed_scale, making delta_heading contribute incorrectly
to magnitude weights). The fix: compute step magnitudes from the reconstructed (cum_x, cum_y)
positions instead.

This bug fix improved AUC from **0.917 → 0.852** — a larger improvement than any hyperparameter
change. The distorted correction weights were systematically warping path shapes.

### Key Findings

1. **Polar representation matters**: Encoding (speed, delta_heading) instead of (dx,dy) improved
   AUC from 0.950 to 0.852 (when combined with correct endpoint correction). The model directly
   learns angular velocity distributions.

2. **CFG hurts naturalness**: Classifier-free guidance amplifies conditioning signal, distorting
   fine dynamics. CFG=0.0 is best (0.852). Also 2x faster (no second forward pass).

3. **Endpoint correction bugs are devastating**: A subtle scaling error in the correction weights
   caused a 0.065 AUC regression. Path-shape features (angular velocity, curvature, efficiency)
   are extremely sensitive to how endpoint error is distributed.

4. **30 epochs > 20 epochs**: Slower LR schedule allows continued improvement. Val loss dropped
   from 0.458 to 0.423.

5. **200-sample AUC is unreliable**: Varied by 0.06-0.10 between runs. Always verify at n=2000.

### Remaining Gaps (n=2000, CFG=0.0)

| Feature | Wasserstein | Note |
|---------|------------|------|
| angular_velocity_std | 0.304 | Still largest gap |
| angular_velocity_mean | 0.280 | |
| path_efficiency | 0.262 | Path straightness |
| curvature_std | 0.237 | Persistent across all models |
| curvature_mean | 0.188 | Persistent across all models |
| num_direction_changes | 0.179 | |
| velocity_skewness | 0.148 | Much improved from Cartesian |
| movement_duration | 0.151 | |
| All velocity/accel features | < 0.05 | Excellent |

RF feature importances spread out — no single feature dominates (top: mean_jerk 0.110,
movement_duration 0.080, mean_acceleration 0.072).

## Updated Scoreboard (2026-05-27)

| Approach | AUC (n=2000) | Type | Notes |
|----------|-------------|------|-------|
| **Corpus Replay (4.16M)** | **0.52** | Retrieval | Translate-only, floor |
| Corpus Perturb v7 | 0.645 | Retrieval + noise | Best perturbation variant |
| **Enhanced Corpus Rotate (K=50)** | **0.670** | Retrieval + transform | Best rotation variant |
| Corpus Rotate (K=5) | 0.686 | Retrieval + transform | Less donor diversity |
| Corpus Sim | 0.682 | Retrieval + transform | Same mechanism as rotate |
| DDPM + borrowed timing | 0.820 | Hybrid (not generative) | Proves DDPM path is OK |
| **CANDI polar flow (21ep)** | **0.854** | **Generative** | **Best generative. Flow matching.** |
| CANDI polar DDIM (30ep, CFG=0) | 0.852 | Generative | DDIM baseline |
| ZIMT magcorr | 0.864 | Generative | Previous best generative |
| ZIMT FM (50 iter) | 0.870 | Generative | Exposure bias worsened it |
| ZIMT baseline | 0.878 | Generative | Original ZIMT |
| VQ-VAE + Transformer | 0.890 | Generative | Undertrained, mode collapse |
| ZIMT reject (N=8) | 0.892 | Generative | Variance collapse |
| SoundStorm donor perturb | 0.914 | Generative | VQ-VAE quantization bottleneck |
| Corpus Blend | 0.948 | Retrieval + blend | Arc-length resampling kills features |
| CANDI cartesian (20ep) | 0.950 | Generative | (dx,dy) can't capture angular dynamics |
| **Chunk Diffusion** | **0.957** | **Generative** | **No global awareness** |
| ZIMT guided | 0.968 | Generative | Guidance at inference fails |
| SoundStorm from-scratch | 0.996 | Generative | Cold-start stall collapse |

### Architectures Tried: 4 of 9

| # | Architecture | Status | AUC | Failure Mode |
|---|-------------|--------|-----|-------------|
| 1 | **CANDI hybrid diffusion** | **BEST** | **0.854** | **Best generative. Angular velocity + curvature remain.** |
| 2 | Action Chunking | FAILED | 0.957 | No global velocity awareness |
| 3 | SoundStorm/MaskGIT | FAILED | 0.996 | VQ-VAE quantization bottleneck |
| 4 | Decoupled Shape+Speed | Untried | — | Partially validated by v147 |
| 5 | ProDMP | Untried | — | Smooth by construction, needs stall augmentation |
| 6 | MoGlow | Untried | — | Exact likelihood, but still AR |
| 7 | Neural SDE | Untried | — | Can't produce exact stalls |
| 8 | Energy-Guided Diffusion | Untried | — | Numerically unstable near stalls |
| 9 | Mamba | Untried | — | Likely incremental over ZIMT |

## Flow Matching + Post-Processing Optimization (2026-05-30)

### Training Objective: DDIM → Flow Matching

Replaced 1000-step cosine-schedule DDIM with flow matching. Linear interpolation
x_t = (1-t)·x_0 + t·ε, 200-step Euler ODE sampling. Same Transformer backbone (5,794,819 params).
Training on 500K trajectories, polar (speed, Δheading), heading-flip augmentation.

Training ongoing (epoch 23/80). Best val_cont = 1.220 at epoch 21.

### Post-Processing Sweep (30+ experiments on epoch 14 checkpoint)

All 18 kinematic features derive from the same trajectory. Improving one feature via
post-processing systematically degrades others. This is the fundamental ceiling of
inference-time corrections.

**Working parameters (small consistent benefit):**

| Parameter | Effect | Optimal |
|-----------|--------|---------|
| SPEED_SKEW | Time-warps speed profile: t^(1/(1+k)) | Shifts with training (0.3 → 0.1) |
| PERP_SCALE | Compresses perpendicular displacement | 0.7 |
| GUIDE | Directional guidance during ODE sampling | 0.3 |
| CORRECT=rotate | Endpoint correction via rotation+scaling | Always > additive |

**Dead-end parameters (all net negative at every tested value):**
DH_AMP, PERP_HP, FLOW_NOISE, SPEED_RAMP, JITTER, DH_OU, SPEED_SKEW_SCALE, RESIDUAL_VEL.
Each improves its target metric but degrades others through feature coupling.

### Parameter Shift: Post-Processing Invalidated by Model Improvement

As the model trains longer, it learns speed asymmetry internally. Post-processing
parameters optimized for epoch 14 become harmful on epoch 19:

| Config | AUC (n=100) | Notes |
|--------|-------------|-------|
| Epoch 14, skew=0.3+perp=0.7 | 0.533 | Optimal at epoch 14 |
| Epoch 19, skew=0.3+perp=0.7 | 0.734 | Same config over-corrects |
| Epoch 19, raw (no post-proc, no guide) | 0.527 | Model improved enough to need less help |
| Epoch 19, skew=0.3 only | 0.526 | Mild skew still helps |

Post-processing must be re-swept at each training milestone. Combined corrections
(skew+perp) over-correct on improved checkpoints.

**N=100 AUC is systematically optimistic.** The RF needs ~2000 samples to detect subtle
artifacts. Cartesian showed 0.86 at n=200 → 0.95 at n=2000. Use n=100 only for
relative ranking between configs.

### Definitive Evaluation (n=2000, epoch 19)

| Config | AUC (n=2000) | Notes |
|--------|-------------|-------|
| skew=0.3+perp=0.7, guide=0.3 | 0.854 | Old config, over-corrected |

Re-optimized config N=2000 evaluation pending.

### Remaining Gaps (n=2000, epoch 19)

| Feature | Wasserstein | RF Importance |
|---------|------------|---------------|
| angular_velocity_std | 0.379 | 0.066 |
| angular_velocity_mean | 0.332 | 0.062 |
| path_efficiency | 0.291 | — |
| curvature_std | 0.232 | 0.067 |
| max_deviation | 0.231 | — |
| curvature_mean | 0.186 | 0.070 |
| time_to_peak_velocity | 0.171 | 0.058 |
| mean_jerk | 0.029 | **0.131** |
| mean_acceleration | 0.030 | 0.055 |

mean_jerk is the #1 RF discriminator despite low Wasserstein distance. The correlation
structure is broken: human mean_acc × mean_jerk = +1.000, synthetic = -0.742 (gap 1.74).

**Root cause:** mean_acceleration = mean(Δspeed/Δt) telescopes to (speed_end - speed_start)/T
for constant dt. Rest-to-rest trajectories → mean_acc ≈ 0 regardless of speed scale. Human
data includes partial movements where mean_acc ≠ 0, creating speed-dependent signed
acceleration that the model cannot replicate.

### N=100 vs N=2000 Reliability Problem

The N=100 sweep showed AUCs of 0.526-0.527 for top configs, suggesting the model was near
0.50. The N=2000 eval revealed the true AUC is **0.854** — a 0.33 discrepancy. The RF at
N=100 simply lacks enough data to detect subtle systematic artifacts.

This invalidates all N=100-based optimization done to date. A reliable proxy metric is needed
before further optimization — one that's stable at small N and predicts N=2000 AUC.

### Next Steps

**1. Proxy metric (highest priority).** Build a deterministic feature-matching score
(sum of Wasserstein distances + correlation matrix Frobenius norm) that's stable at N=100.
Script created (`proxy_metric_validation.py`), needs to complete its 2000-trajectory
generation run. Once validated, this replaces RF AUC for rapid iteration.

**2. Training augmentations.** Current training uses only heading-flip. Key untapped
augmentations:
- `--heading-scale 0.15`: augments heading magnitude range. Directly targets angular velocity
  gap (Wasserstein 0.38). Training started from epoch 23 checkpoint.
- `--heading-noise`: OU-process on delta_heading. Could improve curvature.
- `--path-weight`: auxiliary loss matching path_efficiency distribution.

**3. Longer training.** Model improved steadily through epoch 23 (val_cont 1.26 → 1.22).
Continue to epoch 80. Re-evaluate with proxy at each milestone.

**4. Structural fix for correlation gap.** The mean_acc telescoping issue requires either:
partial-movement conditioning (non-zero start/end speeds), data augmentation with partial
movements, or accepting the correlation gap and focusing on other features.

**5. Feature-aware training loss.** Add auxiliary losses during training that match
batch-level angular velocity and curvature statistics against human targets. The path_loss
infrastructure already reconstructs trajectories from polar predictions — extend to
angular velocity matching.

---

## 2026-07-01: Evaluation Overhaul and Baseline Re-Measurement

### Batched generation: n=2000 eval now takes 4 minutes instead of 100

The definitive eval on May 30 spent 5703s generating 2000 trajectories one at a time,
200 sampler steps each. Added `generate_paths` to `experiments/candi.py`: requests are
grouped by sequence length and run through the sampler in batches of up to 128. Same
sampler code, same per-trajectory decode and post-processing. Generation now takes 188s.
The proxy-metric plan from the previous entry is dead: full n=2000 evals are cheap enough
to use for every decision.

### Baseline re-measured: 0.829, not 0.854 (n=2000, 3 seeds)

CANDI polar flow (candi_polar_flow_best.pt), skew=0.3 perp=0.7 guide=0.3 correct=rotate
steps=200 CFG=0, batched generation:

| Seed | RF OOB | GBM 5-fold CV |
|------|--------|---------------|
| 42 | 0.8315 | 0.8521 |
| 43 | 0.8296 | 0.8501 |
| 44 | 0.8261 | 0.8559 |

Run-to-run spread is about 0.003, so n=2000 comparisons are trustworthy down to roughly
0.005 differences. The May 30 sequential run recorded 0.8537 with nominally the same
config. The gap is under investigation (parity check pending); the full environment of
the old run was not recorded, so 0.829 +/- 0.003 from the current committed code is now
the reference baseline.

### New held-out detector: raw-trajectory CNN (never tuned against)

`detector_raw.py`: a small 1D CNN on resampled, distance-normalized (dx, dy) sequences,
human side drawn from the 208K held-out test split. 3-fold CV, fixed hyperparameters,
runs automatically inside evaluate.py. Sanity check human-vs-human gives 0.48.

First reading on the baseline model: **raw-NN AUC 0.600** while the feature RF sees 0.83.
The raw local structure of generated trajectories is much harder to detect than the
18 aggregate features. The remaining gap is concentrated in feature-level statistics,
not in local waveform texture.

### The May 30 fine-tunes all hurt (n=2000, same eval config as baseline)

The speedaug and hscale fine-tune runs died mid-training on May 30 and were never judged
at full sample size. Verdict:

| Checkpoint | RF OOB | GBM CV | Raw-NN | vs baseline 0.829 |
|-----------|--------|--------|--------|-------------------|
| speedaug_best (DDIM, ep 34) | 0.9243 | 0.9374 | 0.591 | much worse |
| speedaug_gentle (DDIM) | 0.8883 | 0.9078 | 0.597 | worse |
| flow_hscale (ep 24) | 0.8389 | 0.8644 | 0.601 | worse |

Every fine-tune launched from N=100 sweep evidence made things worse at n=2000. Do not
resume any of them. Baseline stays candi_polar_flow_best.pt (epoch 21).

Sequential vs batched parity: same 300 endpoint specs through both paths, per-feature
Wasserstein all at or below 0.10, which is n=300 sampling noise. No evidence the batched
path changes the distribution. Note the best checkpoint on disk is epoch 21; training
continued past the epoch the old docs called best, overwriting the file, and the tuned
post-processing knobs are stale for it.

### Training speed: 780s to about 280s per epoch (2.8x)

Profiling result: the per-sample polar conversion loop in the DataLoader was costing
roughly 500s per epoch. Precomputed it once for all 500K trajectories
(training/prepare_polar.py, 10 seconds, verified bit-exact against _to_polar) and taught
train_candi.py to load the arrays when present. Dataloader cost drops to under 10s per
epoch. The remaining floor is GPU compute: 72.5 ms per step at batch 128, seq 128, which
is 266s per epoch on the 4070. Larger batches do not help (159 ms at 256) and batch 512
peaks at 6.6 GB VRAM, too close to the 8 GB limit. torch.compile was skipped since
Triton support on Windows is unreliable. Fine-tune epochs now cost about 5 minutes.

### WS4 diagnostic, human side

On 3000 held-out human trajectories: displacements after the standard 125Hz resample are
not integer (mean distance to nearest integer 0.148 px, uniform would be 0.25), 35% of
steps move less than 1 px, 4.8-6.0% are exact zeros. An integer-displacement output
channel loses its premise (the eval field is not integer), but the sub-pixel step
fraction is a concrete target: measure the same fraction on generated output next.

### Post-processing re-sweep at n=2000: the old knobs were hurting badly

Coordinate descent over the three post-processing knobs, every point a full n=2000 eval,
epoch 21 checkpoint, CANDI_STEPS=200 CFG=0 CORRECT=rotate throughout. The incumbent
values (skew 0.3, perp 0.7, guide 0.3) were tuned at N=100 on an older checkpoint.

| skew | perp | guide | RF OOB | GBM CV |
|------|------|-------|--------|--------|
| 0.3 | 0.7 | 0.3 | 0.829 | 0.853 |
| 0.2 | 0.7 | 0.3 | 0.8126 | 0.8422 |
| 0.1 | 0.7 | 0.3 | 0.8162 | 0.8416 |
| 0.0 | 0.7 | 0.3 | 0.7650 | 0.7981 |
| 0.0 | 0.85 | 0.3 | 0.7613 | 0.7838 |
| 0.0 | 0.85 | 0.15 | 0.7470 | 0.7710 |
| 0.0 | 0.85 | 0.0 | 0.7690 | 0.7868 |
| 0.0 | 1.0 | 0.0 (bare model) | 0.7584 | 0.7830 |

Findings:

1. Speed skew was the big lie. Turning it off alone is worth 0.064 AUC. The knob was
   compensating for an epoch 14 weakness that epoch 21 no longer has, and at 0.3 it was
   actively injecting a detectable artifact.
2. The completely bare model (no skew, no perp compression, no feature guidance, only
   the endpoint rotate correction) scores 0.7584. That is better than everything we had
   ever measured, and it is essentially pure T3 model output.
3. Descent winner: skew 0.0, perp 0.85, guide 0.15 at 0.7470. Held-out GBM moved in the
   same direction at every point, so this is not RF gaming.
4. This confirms standing rule 5 (post-processing params drift per checkpoint) in the
   strongest possible way: every knob tuned at N=100 on the old checkpoint was either
   useless or harmful at n=2000 on the current one.

New best: RF OOB 0.7470 at n=2000 (pending 3-seed confirmation and raw-NN check).
Previous best was 0.829. One sweep point (perp 1.0, guide 0.3) failed to parse and is
being rerun alongside the untested (perp 1.0, guide 0.15) combo.

### Follow-up: holes filled, winner confirmed across 3 seeds

The two missing points: (skew 0, perp 1.0, guide 0.15) scored 0.7627 and the retried
(skew 0, perp 1.0, guide 0.3) scored 0.7557. Neither beats the descent winner.

Winner (skew 0, perp 0.85, guide 0.15) across seeds 42/43/44: RF OOB 0.7470, 0.7503,
0.7591. Mean 0.752, spread 0.005. Held-outs on the confirmation seeds: GBM 0.78, raw-NN
0.58 and 0.55 (baseline was 0.600). Every detector improved, none was tuned against
except the RF.

Summary of bests:
- Best overall: skew 0, perp 0.85, guide 0.15, rotate correction. RF OOB 0.752 +/- 0.005.
- Best pure T3 (no post-processing except rotate): bare model at 0.7584 single seed.
  The 0.006 gap between them is near noise, so the model itself carries the result.

### WS4 diagnostic, synth side (best config, n=1000, after standard resample)

| Measure | Synth | Human |
|---------|-------|-------|
| Sub-pixel steps (<1 px) | 0.298 | 0.35 |
| Exact-zero steps | 0.025 | 0.048 to 0.060 |
| Mean distance to integer | 0.227 | 0.148 (0.25 = no grid structure) |

Reading: the sub-pixel fraction is close, so the slow-tail premise is mostly satisfied
already. Two real gaps remain. The model stalls half as often as humans, and its
displacements are nearly grid-free (0.227, close to the 0.25 uniform value) while human
data keeps visible pixel-grid structure even after resampling. The stall deficit is the
more promising target since stalls are known to drive the angular and jerk features.

### WS3 distribution-matching fine-tune: negative result, gate fired

Built a distribution-matching fine-tune (training/train_candi_dm.py): the model generates
short trajectories by unrolling the flow ODE with gradients on the last few steps, computes
13 differentiable kinematic features on the batch, and minimizes a multi-bandwidth RBF MMD
against real human batches. The standard flow-matching loss stays on as an anchor. This is
the first time any model here trained directly on the quantity the detector measures.

Ran 600 steps from candi_polar_flow_best.pt at lr 2e-5, batch 48, 8-step unroll. The MMD
fell from 0.78 to about 0.16 in the first 40 steps then plateaued and never went lower.

Result at n=2000 (candi_dm_v1.pt):

| Config | RF OOB | GBM CV | Raw-NN | vs pre-tune |
|--------|--------|--------|--------|-------------|
| bare (skew 0, perp 1.0, guide 0) | 0.7726 | 0.8042 | 0.5765 | worse by 0.014 |
| tuned (skew 0, perp 0.85, guide 0.15) | 0.7688 | 0.7975 | 0.5752 | worse by 0.017 |

Both configs came out worse than the pre-fine-tune baseline (bare 0.7584, tuned 0.752),
by more than the 0.005 noise floor. The matching loss plateaued well above zero and the
detector kept its edge. This is one of the two pre-defined failure signatures: the model
cannot be pushed onto the human feature distribution without the anchor loss fighting it.

Reading: the CANDI flow backbone is at or near its ceiling for this metric. Short-unroll
gradient matching does not move it. This is evidence, not proof (one short run at one lr),
so a fresh session may confirm with one or two more configs (higher mmd weight, weaker
anchor, longer unroll) before committing. But the weight of evidence points to a new
architecture rather than more tuning of this one.

### WS7 feasibility: raw event streams exist and the representation is near-lossless

The pool files pool_flat_i16.npy and pool_t_rel_f32.npy turn out to hold the raw
pre-resample event data: integer pixel positions with millisecond timestamps. Only 30%
of inter-event gaps sit on the 8ms clock, the rest spread from 1 to 150ms. This was the
single biggest risk for the event-stream model and it is retired. The model can train on
real event streams directly, no deconvolution needed.

Event statistics from a 20k-trajectory sample: median 43 events per trajectory (p99 389),
displacement vocabulary of +/-63 covers 99.1% of x steps and 99.7% of y steps, 14.8% of
events are pure time ticks with no movement, 6.3% of steps carry duplicate or out-of-order
timestamps.

Upper-bound test (experiments/event_replay.py): real event streams pushed through the
exact (dt, dx, dy) encode-decode the model would use, evaluated at n=2000.

| Variant | RF OOB | GBM CV | Raw-NN |
|---------|--------|--------|--------|
| pure replay (pipeline check) | 0.4960 | 0.4871 | 0.4868 |
| merge duplicate timestamps only | 0.5069 | 0.5071 | 0.4863 |
| merge + clip jumps to +/-63 | 0.5247 | 0.5498 | 0.4878 |
| merge + split jumps (no clipping), 3 seeds | 0.5018 / 0.5199 / 0.5186 | 0.49 to 0.51 | 0.49 to 0.51 |

Clipping oversized jumps is detectable, but splitting them into collinear sub-events
reconstructs the same path and costs nothing. Verdict: the representation ceiling is
about 0.51, a quarter-point below the current best generative result of 0.752. All
three feasibility gates pass. Building the model is justified.

### WS7 result: trained event-stream model does NOT beat CANDI (0.945 vs 0.752)

Model: 5.9M-param non-autoregressive Transformer, categorical dx/dy heads (128 classes
with PAD) trained by absorbing-state masking, flow matching on z-scored log(dt). Trained
22 epochs on a 1M subsample (val flow 0.2175, val ce 0.9908, monotone). Checkpoints and
code in models/event_stream.py, training/train_events.py, experiments/event_stream.py.

Sampler note: the first confidence-reveal sampler used argmax at every position, which
collapses each token to its per-position mode and telescopes net displacement to ~7% of
requested. Replaced with a MaskGIT-style sampler: reveal count follows the training mask
schedule sqrt_ab[t], tokens are sampled (not argmaxed) from the softmax, order by
sampled-token confidence. After the fix an all-MASK probe shows E[sum dx,dy] tracks the
conditioning within ~15%, angle error 2.4 deg, event count median 45 vs human 43.

| Config (n=2000) | RF OOB | RF CV | GBM CV | Raw-NN |
|-----------------|--------|-------|--------|--------|
| epoch 22, temp 1.0 | 0.9451 | 0.9463 | 0.9515 | 0.6293 |
| epoch 22, temp 0.7 | 0.9641 | 0.9647 | 0.9671 | 0.6619 |
| epoch 22, temp 0.5 | 0.9695 | 0.9702 | 0.9712 | 0.6350 |

Failure is entirely in path SHAPE, not signal texture. The worst per-feature Wasserstein
gaps are angular_velocity_mean (0.72), path_efficiency (0.57), num_direction_changes
(0.57) - the generated paths zigzag far more than human arcs. Every velocity/acceleration/
jerk feature matches to within 0.06, and the raw-NN detector on the low-level signal only
reaches 0.63. So the per-event texture we built this approach to capture is essentially
right; the model fails to produce directionally coherent (smooth) paths.

Lower temperature makes it WORSE (0.945 -> 0.964 -> 0.970 as temp 1.0 -> 0.7 -> 0.5),
which rules out sampling variance as the cause. The zigzag is structural: dx and dy are
predicted by independent heads and revealed largely independently per position, so nothing
enforces the strong direction autocorrelation of a human arc. Real events replayed through
the same representation score 0.51 (above), so the representation is not the problem - the
model is. Fixing this needs an architecture/training change (joint dx,dy or velocity-with-
smooth-prior; couple adjacent steps), not a decode knob. Go/no-go: this model as built is
a NO. CANDI at 0.752 remains the best result.

### WS7b gate: polar representation passes only with integer-pixel decode

The WS7b speed+heading representation was gated before training by replaying real
event streams through it (experiments/event_replay_polar.py, n=2000 each). The
lossless float round-trip matches the WS7 split-replay floor, as it must. The
surprise is what quantization costs and where it comes from:

| Variant | RF OOB | GBM CV | Raw-NN |
|---------|--------|--------|--------|
| exact float round-trip (sanity) | 0.5084 | 0.5033 | 0.4861 |
| dtheta 256 bins | 0.5383 | 0.5666 | 0.4853 |
| dtheta 512 bins | 0.5473 | 0.5701 | 0.4849 |
| dtheta 1024 bins | 0.5677 | 0.5818 | 0.4868 |
| dtheta 256 + speed 128 log bins | 0.5640 | 0.5803 | 0.4868 |
| dtheta 1024 + speed 512 log bins | 0.5662 | 0.5848 | 0.4871 |
| dtheta 256 + speed 128, positions ROUNDED to integer px | 0.5073 | 0.5141 | 0.4859 |
| dtheta 256 only, positions ROUNDED | 0.5121 | 0.5200 | 0.4858 |

Two findings. First, the penalty does not shrink with finer bins (256 to 1024 bins
all land around 0.54 to 0.57), so it is not quantization drift. Second, rounding
decoded positions back to integer pixels removes the penalty entirely at any bin
resolution tested. The feature detectors are keying on off-grid positions, not on
angular resolution: any decode that leaves positions off the integer lattice costs
about 0.05 AUC on otherwise-real data, invisible to the raw-NN (0.486 throughout)
but visible to RF and GBM. This confirms the pixel grid is a load-bearing part of
the signal (the original WS7 thesis) and fixes the WS7b decode contract: integrate
speed and heading continuously, then round positions to integer pixels inside the
decode. With that contract the full planned model quantization (256 dtheta bins,
128 log-speed bins) sits at the representation floor: 0.5073, within noise of pure
replay.

Human dtheta statistics that shaped the head design (500-trajectory sample): 32%
of heading increments are exactly 0, 46% sit on the 45-degree lattice of 1px
moves, median |dtheta| is 5.5 degrees but p90 is 45 degrees. Large turns happen
almost only at low speed, so the dtheta head is categorical (256 bins, spikes get
exact bins) and conditioned on the speed class at the same position:
p(s, th | ctx) = p(s | ctx) p(th | s, ctx). Ticks (10% of events after merge)
carry no heading; heading persists through them. PAD lives on the speed head only,
so decode truncation has a single owner. The first motion event's dtheta is
relative to the conditioning angle, so the decoder needs no side information.

### WS7b result: sampler reveal order was hiding 0.12 of AUC, model lands at 0.806

The 22-epoch overnight run (1M subsample, models/event_stream_polar.py) finished
cleanly on July 3 with all three losses still slowly improving (val flow 0.224,
speed CE 1.301, dtheta CE 1.886). Out of the box at n=2000 it scored RF OOB 0.929,
barely better than WS7's 0.945, and the failure looked inverted: paths were far too
straight (path_efficiency median 0.994 vs human 0.949, max_deviation 5.9px vs
16.6px, curvature_std 15x too small).

The cause was the sampler, not the model. MaskGIT confidence-order reveal always
unmasks the most confident positions first, and the most confident dtheta is
always "no turn", so straightness gets locked in early and every later token
conditions on a straight context. Pure random reveal order overshoots the other
way (max_deviation 55px, wandering paths). The standard MaskGIT fix, Gumbel noise
on the confidence with a linearly annealed choice temperature, interpolates:

| Sampler config (ep22 checkpoint) | RF OOB | GBM CV | Raw-NN |
|----------------------------------|--------|--------|--------|
| confidence order (as built)      | 0.9289 | 0.9310 | 0.6396 |
| gumbel choice_temp 3             | 0.8390 | 0.8494 | 0.6100 |
| gumbel choice_temp 4             | 0.8060 | 0.8223 | 0.6086 |
| ct 4 + tick merge + 200 steps    | 0.8080 | 0.8315 | 0.6036 |

One knob, 0.12 of AUC. Event-level statistics under ct=4 match humans closely:
dtheta sign persistence 0.358 vs 0.342, |dtheta| by speed band within a few
degrees at every band, dt-by-speed medians identical (8ms motion cadence, 1-2ms
ticks), speed lag-1 autocorrelation 0.66 vs 0.59, and both starts and endings
ramp like human trajectories. PAD placement is perfectly contiguous.

Two artifacts survive the sampler fix. First, the model emits ticks at 2.3x the
human rate (22% vs 9.4% of events) and scatters them mid-flight where humans
almost never put them (worst paths alternate tick/motion at 1ms/7ms, which is
what blows up the angular-velocity features). Merging ticks into the following
event at decode improves the Wasserstein gaps but not the AUC. Second, turning
per unit time stays about 1.8x human even with ticks merged, and the detector's
remaining edge sits in correlation structure (mean_acceleration vs everything,
gaps above 1.0) plus angular_velocity_mean (0.41) and path_efficiency (0.36).

Go/no-go: WS7b as trained does not beat CANDI polar flow (0.806 vs 0.752). The
remaining gaps are exactly the quantities a distribution-matching fine-tune
trains on directly, the event backbone is not at a representation ceiling
(replay floor 0.507), and WS3 already proved the MMD machinery runs. Next step
is the planned stage 2: MMD fine-tune of the s/dtheta heads on the event
backbone, dt head frozen, partial no-grad MaskGIT reveal matching the eval
sampler followed by a straight-through Gumbel pass for gradients.

### Stage 2, first pass: distribution matching moves the score for the first time

Built training/train_events_polar_dm.py, the WS3 idea rebuilt for the event
backbone. Per step: keep a real batch's dt and length (timing stays untouched,
dt head frozen), partially sample s/dtheta with the exact eval sampler (MaskGIT,
Gumbel choice order) to a random reveal fraction, complete the rest in one
straight-through Gumbel-softmax pass, and match statistics of the completed
streams against an independent real batch. The pretraining losses stay on as an
anchor.

What the first three versions taught:

- v1 matched 15 event-level features with RBF MMD. The MMD fell 0.22 to 0.17,
  but the real-vs-real MMD floor at that batch size turned out to be 0.15: the
  pretrained model was ALREADY matched on event-level statistics, so there was
  nothing to learn. AUC went 0.806 -> 0.834. Lesson: measure the estimator
  floor before reading a plateau as convergence.
- The detector's edge lives after the 125Hz resample, so v2 rebuilt the feature
  layer as a differentiable copy of features.py: linear-interpolation resample
  of integer-rounded (straight-through) positions, then all 18 detector
  features, plus per-feature quantile matching (the eval's Wasserstein table)
  and covariance matching (the eval's correlation table). v2 diverged: both
  match and anchor rose monotonically from step 1 (0.891 at eval). Cause:
  atan2 gradients scale as one over segment length squared and curvature
  divides by speed cubed, so sub-pixel resample frames flooded every update
  with noise.
- v3 stabilized it (curvature speed clamp raised to 30 px/s, angle gradients
  detached at slow frames, match weight cut to 1, lr 1e-5). Match losses fell
  toward their floors with the anchor flat, and the eval improved: RF OOB
  0.7907, GBM 0.8060, raw-NN 0.6125. First AUC gain from distribution matching
  in this project (WS3 on CANDI only ever made things worse).

0.791 is still short of CANDI's 0.752, but the mechanism finally works and has
obvious headroom: the v3 slow-frame gradient detach cut gradient flow exactly
where the top remaining gap (angular_velocity_mean, W 0.50) lives. v4 replaces
the detach with a scale-invariant atan2 on clamped-length-normalized segments
(identical values, bounded gradients) and a bigger matching batch.

### Lattice snap: the angular-velocity gap was manufactured by the decode

The stubborn angular_velocity_mean gap (W 0.4 to 0.5 across every config) turned
out not to be in the model at all. Frame-level profiling localized the entire
excess to slow frames (0.5 to 3.2 px per frame): 3x the human turning rate
there, all other speed bands matching. Decoding the same token streams without
integer rounding erased it completely (slow-band median |omega| 24.6 -> 0.0),
which pins the mechanism: the model emits smooth off-lattice headings, and
error-carrying rounding of a slow off-lattice path alternates lattice
directions nearly every step. Real slow movement does not do this: a person
moving 1px at a time emits repeated identical integer steps with occasional
direction changes, because the recording lattice IS their output space.

Fix (decode contract, same category as the mandatory integer rounding): emit
slow steps (s < 2.5 px) as whole lattice steps, rounding the step vector at
the continuous heading to the nearest realizable integer displacement. The
integrated heading stays continuous, so no drift accumulates in direction.
EVENT_SNAP=2.5 in experiments/event_stream_polar.py. Frame-level profile
after: slow-band median 5.5 vs human 7.8, overall mean |omega| 19.4 vs 21.2.

Result at n=2000 on the DM v3 checkpoint (gumbel ct=4, snap 2.5):

| Config | RF OOB | GBM CV | Raw-NN |
|--------|--------|--------|--------|
| dm_v3 + snap 2.5 | 0.7550 | 0.7822 | 0.6078 |

Statistically tied with the CANDI polar flow best (0.752 +/- 0.005), reached
in pure T3 with no post-processing, and the day's chain was 0.929 -> 0.806
(sampler) -> 0.791 (distribution matching) -> 0.755 (decode contract).
angular_velocity_mean and _std left the top-gap table entirely. What remains
is the heavy tail of messy human paths: 10% of human trajectories have
path_efficiency below 0.48 (overshoot loops, hesitation squiggles) and the
model generates almost none; human curvature_std p90 is 30x synth. Next:
retrain the DM stage with the snap inside the differentiable decode
(train/decode consistency) and check whether the quantile term can pull the
messy tail in; sweep choice_temp under the new decode.

### New project best: 0.702 +/- 0.007, event model, pure T3

Three more findings stacked on the lattice snap during the same day:

1. The choice-temperature optimum moved once the snap removed the slow-frame
   jitter: extra reveal randomness now converts into macro shape variety
   (which the detector rewards) instead of wiggle (which it punishes).
   Sweep at n=2000, dm_v3 + snap 2.5: ct4 0.7550, ct5 0.7380, ct6 0.7211,
   ct8 0.7191, ct10 0.7339. Flat bottom at 6-8.
2. Retraining the DM stage with the snap inside the differentiable decode
   (v5, from base, ct7 reveal) did NOT beat the v3 checkpoint: 0.726-0.730.
   The DM axis is saturated around 0.72 under this harness.
3. The duration sampler had been running at 0.7x human variance the whole
   time (DurationModel std_mult default, inherited from the CANDI eval
   configs). At EVENT_DUR_STD=1.0 the model is finally asked for slow,
   hesitant movements, which is where the messy-path tail lives:
   path_efficiency gap fell from 0.21 to 0.13 and the score dropped 0.023.
   ct7 and snap 3.5 rechecks under the new conditioning both lost, so the
   optimum stands.

Best config, 3-seed confirmed at n=2000 (seeds 42/43/44):

    EVENT_CKPT=event_polar_dm_v3.pt EVENT_ORDER=gumbel EVENT_CHOICE_TEMP=8
    EVENT_SNAP=2.5 EVENT_DUR_STD=1.0
    python evaluate.py --experiment experiments.event_stream_polar --n-synthetic 2000

| Seed | RF OOB | GBM CV | Raw-NN |
|------|--------|--------|--------|
| 42   | 0.6960 | 0.7167 | 0.5582 |
| 43   | 0.7114 | 0.7288 | 0.5565 |
| 44   | 0.6976 | 0.7231 | 0.5728 |

RF OOB 0.702 +/- 0.007, the first result below 0.75 and the largest one-day
move in the project (0.929 to 0.696 on seed 42). All of it is the same 6M
parameter model trained overnight; the gains came from the sampler reveal
order, distribution-matching fine-tune, two decode-contract fixes, and
restoring full duration variance in the conditioning.

Remaining top gaps at the best config: angular_velocity_mean 0.34 (the
ct=8 randomness re-inflates some wiggle; tension with shape variety),
curvature_std 0.23, curvature_mean 0.18. Candidate next levers, in rough
order of expected value: longer pretraining (22 epochs on 1M was still
improving; 4M is available), an adversarial feature-space critic to replace
the saturated fixed-statistic matching, and a curvature-aware term in the
DM feature set.

### 4M pretrain and the ~0.70 ceiling (July 4-5)

Full-data pretrain: 12 epochs scheduled over all 4,028,855 trajectories,
stopped after epoch 11 (the final epoch's learning rate was near zero and
the machine had bluescreened three times during the run; hardware, not
code). Checkpoint event_polar_4m.pt. Result at n=2000, ct6 snap 2.5
dur 1.0, seeds 42/43/44: 0.6901 / 0.7030 / 0.7062, mean 0.700 +/- 0.007.
A statistical tie with the fine-tuned 1M best, reached with no fine-tune
at all: four times the data absorbed everything the DM stage used to add.

Both fine-tuning axes then FAILED on the 4M base, in the same way:

| Fine-tune | RF OOB (seed 42) | vs base 0.6951 |
|-----------|------------------|----------------|
| Fixed-statistic DM (quant+cov+MMD) | 0.7102 | worse |
| Adversarial critic (hinge GAN, 18 detector features, 800 steps) | 0.7112 | worse |

The critic run is the informative one: the discriminator's real-vs-fake
margin grew monotonically from 0.03 to 1.80 over all 800 steps and the
generator never closed any of it, while the pretraining anchor loss stayed
flat. The critic finds real, learnable differences; pushing the generator
against them only degrades it. Combined with the correlation diagnosis
(human per-trajectory features co-vary, mean_acc vs std_acc r = +1.0;
synth r = -0.5), the conclusion is that the remaining ~0.20 of AUC lives
in global per-trajectory coherence that a masked token model with
per-position heads cannot represent, no matter the training signal.
The pure-generation (level 3) line therefore stands at 0.70 +/- 0.01.

### Masked-infill frontier (level 2): purity vs detectability

experiments/event_infill_polar.py: borrow a real trajectory of matching
distance (same selection as the replay gate), keep its timing, mask a
fraction of the speed/heading tokens, let the MaskGIT sampler regenerate
them, decode under the full contract, rotate to the requested direction.
INFILL_FRAC=0 is the replay floor (0.507); FRAC=1 with real timing lands
at pure generation (0.695). Seed 42, n=2000, 4M base, ct6 snap 2.5:

| Fraction regenerated | random positions | contiguous span |
|----------------------|------------------|-----------------|
| 10% | 0.5731 | |
| 15% | 0.5812 | |
| 25% | 0.5819 | 0.6102 |
| 50% | 0.6159 | 0.6681 |
| 75% | 0.6950 | 0.7683 |

Two lessons. First, the very first synthetic tokens are the expensive
ones: 10% regeneration already costs 0.066 of the 0.188 total, then the
curve is flat to 25% before climbing again. There is no infill fraction
that stays near the replay floor. Second, span masking loses to scattered
masking everywhere, and at 75% is WORSE than pure generation: a long
contiguous synthetic stretch is a small synthetic trajectory with the full
coherence gap plus two seams, while scattered tokens each sit inside real
local context that carries them. The frontier's usable summary: 90% real
buys 0.573, 75% real buys 0.582, half real buys 0.616.

### Movement-character conditioning (feature-conditioned fine-tune)

The first lever aimed at the diagnosed coherence gap itself. The model gets
a new conditioning input: the trajectory's own 18 detector features
(z-scored), projected through a zero-initialized pathway so training starts
exactly at the pretrained model. Teacher-forced on real data (each
trajectory conditioned on its own statistics), dt head frozen, 4,000 steps.
At sampling time the character vector is drawn from a KDE over a bank of
real feature vectors stored in the checkpoint, matched to the requested
distance (models/event_stream_polar.py feat_dim,
training/train_events_polar_featcond.py, EVENT_FEAT* knobs in the
experiment). Seed 42, n=2000, ct6 snap 2.5 dur 1.0:

| Config | RF OOB |
|--------|--------|
| 4M base | 0.6951 |
| fc_v1, conditioning ON | 0.6929 |
| fc_v1, conditioning OFF (zero vector) | 0.7247 |

The ON/OFF split shows the pathway is real: the model learned to lean on
the character vector (its unconditional mode degraded 0.03), and unlike
both fine-tunes before it, conditioning did not regress the score. The
cross-feature correlation check (400 trajectories) shows partial repair:
the jerk pairings moved from negative to positive (max_acc x mean_jerk
+0.12 -> +0.52, mean_jerk x std_jerk -0.21 -> +0.48), but
mean_acceleration's pairings got worse. Human derivative features
correlate at r = +1.000 for EVERY pair, i.e. one latent scale variable;
synthetic correlations remain a patchwork. Event-level endings are NOT the
cause (last-motion speed p50 is 1 px for both, no tick tails): the human
lockstep lives in the resampled-space heavy tail. v2 with 12,000 steps is
the scale-up test.

### New project best: 0.675 +/- 0.002, movement-character conditioning

Scaling the feature-conditioned fine-tune from 4,000 to 12,000 steps
(fc_v2) turned the tie into the project's first fine-tune GAIN, and the
first movement of the ~0.70 wall:

| Seed | RF OOB | GBM CV | Raw-NN |
|------|--------|--------|--------|
| 42   | 0.6750 | 0.6936 | 0.5526 |
| 43   | 0.6773 | 0.6943 | 0.5542 |
| 44   | 0.6725 | 0.6860 | 0.5398 |

RF OOB 0.675 +/- 0.002 (previous best 0.702 +/- 0.007; every secondary
detector also improved). Config: EVENT_CKPT=event_polar_4m_fc_v2.pt
EVENT_ORDER=gumbel EVENT_CHOICE_TEMP=6 EVENT_SNAP=2.5 EVENT_DUR_STD=1.0,
experiments.event_stream_polar. Insensitive to KDE bandwidth (0.1 vs 0.25:
0.6762 vs 0.6750 at seed 42). Pure level 3: the character vector is drawn
from a kernel density over real feature vectors, matched to the requested
distance, not copied from any trajectory.

Why it matters beyond the number: two fine-tunes that PUSHED the model
toward feature statistics (fixed matching, adversarial) both regressed,
while giving the model a global variable and teacher-forcing it on real
data gained 0.025. That is direct evidence the wall was the missing
per-trajectory coherence variable, and that it is trainable. The 4k-step
version had already repaired part of the correlation structure; 12k
strengthened adherence. Open levers, in order: even longer conditioning
training (loss was still noisy-flat, adherence may still be growing),
choice-temp sweep under conditioning (the ct=6 optimum was tuned for the
unconditional model), bandwidth/window shaping of the character draw, and
a conditioning-aware DM/critic pass now that the model has the variable
those signals needed.

Post-confirmation choice-temp sweep on fc_v2 (seed 42 only): ct4 0.7072,
ct6 0.6750, ct8 0.6505. The optimum moved UP under conditioning.

## Choice-temp 8 confirmed: new best 0.658 +/- 0.007 (July 5)

Three-seed confirmation of ct=8 on fc_v2 (EVENT_ORDER=gumbel, SNAP=2.5,
DUR_STD=1.0):

| seed | RF OOB | GBM 5-fold | Raw-NN |
|------|--------|------------|--------|
| 42   | 0.6505 | -          | -      |
| 43   | 0.6678 | 0.6765     | 0.5527 |
| 44   | 0.6569 | 0.6703     | 0.5477 |

Mean 0.658 +/- 0.007. Beats the ct=6 best (0.675 +/- 0.002) by 0.017,
outside combined noise. Higher seed variance than ct=6 (0.007 vs 0.002)
but the worst ct=8 seed (0.6678) still beats the best ct=6 seed.

ct=10 at seed 42 gave 0.6564 (GBM 0.6588, raw-NN 0.5366): a plateau, not
a further gain. The choice-temp axis is exhausted at ct=8; the remaining
levers are training-side (longer conditioning training, conditioning-aware
critic), not sampler-side.

## fc_v3 (24k total steps): tie, step axis saturated (July 5)

Continued fc_v2 for 12k more steps (constant lr 2e-5, so equivalent to a
fresh 24k run). Loss flat (3.39 to 3.36). Three seeds at ct=8:
0.6618 / 0.6650 / 0.6534, mean 0.660 +/- 0.005. Statistical tie with
fc_v2 at 0.658 +/- 0.007. The v1-to-v2 gain came from 4k-to-12k steps;
12k-to-24k buys nothing. Conditioning adherence from plain teacher-forced
training is saturated. fc_v2 stays the best checkpoint. Next lever:
conditioning-aware critic (adversarial pass with the character vector
active, so the generator can act on the criticism).

## Conditioning-aware critic: third adversarial strike (July 5)

training/train_events_polar_advfc.py reruns the hinge-GAN critic on fc_v2
with the movement-character vector active on the generation pass, the
anchor pass, and (implicitly) the saved eval bank. Hypothesis was that the
plain critic failed because the generator lacked the global pathway the
criticism referred to. Result: D gap grew 0.02 to 2.20 by step 400 and
plateaued there; the generator never closed any of it. Eval at seed 42:
0.6788 vs fc_v2's 0.6505, a regression consistent with the gap signal.

Verdict: the adversarial/distribution-matching axis is dead on this
architecture, three independent ways (fixed-stat DM, plain critic,
conditioned critic). The failure is not missing representation, it is the
gradient path: per-position token heads cannot coordinate global outcomes
through the ST-Gumbel chain regardless of what the model is told. The
detector's top features have also shifted: with derivative coherence
repaired, the RF now leans on path shape (curvature_std, path_efficiency,
curvature_mean, movement_duration). Remaining gap = the messy-path tail.
Next levers are selection-side, not gradient-side: best-of-N candidate
selection toward the commanded character at sampling time.

## Selection-side levers: best-of-N fails, SIR is the breakthrough (July 5)

Both use the same machinery (EVENT_BESTOF / EVENT_SIR in
experiments/event_stream_polar.py): sample K candidates per spec, keep one.
The difference is the keep rule, and it decided everything.

Best-of-N (K=4, keep the candidate closest to its commanded character):
0.6975 at seed 42 vs 0.6505 baseline. WORSE. Pulling every sample toward a
target shrinks the conditional feature variance, and the detector reads
the shrinkage. Same family of mistake as temperature-lowering in WS7.

SIR, sampling-importance-resampling (K=4, independent character draws,
keep one per spec by Gumbel draw on discriminator log-odds; discriminator
is a fresh GBM fitted human-vs-candidates on the 18 eval-pipeline
features, human refs data/human_eval_features.npy): NEW PROJECT BEST,
3-seed confirmed:

| seed | RF OOB | GBM 5-fold | Raw-NN |
|------|--------|------------|--------|
| 42   | 0.5904 | 0.5931     | 0.5495 |
| 43   | 0.5843 | 0.5750     | 0.5317 |
| 44   | 0.6059 | 0.5924     | 0.5355 |

Mean 0.594 +/- 0.009, previous best 0.658 +/- 0.007: a 0.064 jump, the
largest single gain in the project. All three detector families dropped
together. The mechanism matters: the weighted lottery shifts the realized
feature DISTRIBUTION toward the human one without collapsing variety,
which is exactly what every gradient-side method (DM, critic, conditioned
critic) and the argmin selector failed to do. Purity note: outputs remain
100% generated (Level 3); real data enters only as reference statistics
for the selection weights, the same role it already plays in the KDE
character bank.

Open knobs: K (4 -> 8), EVENT_SIR_TEMP (weight sharpness), and combining
SIR with a ct re-sweep (selection may prefer a different diversity level).

## SIR leakage audit and the clean result (July 5 evening)

The first SIR runs fitted the discriminator on data/human_eval_features.npy,
which turned out to BE the eval's human class (evaluate.py loads the same
file): the judge was studying the answer key. All SIR numbers above are
therefore optimistic. Fix: a 4000-trajectory reference drawn from the 4.16M
pool with the eval's 2000 seed-42 indices excluded
(data/human_ref_features_sir.npy, EVENT_SIR_REF knob, now the default).

Seed-42 sweep before the fix (leaky, for the record): ct8/K4 0.5904,
ct8/K8 0.5847, ct10/K8 0.5724, ct12/K8 0.5771; leaky ct10/K8 3-seed
0.583 +/- 0.008. Diversity helps selection up to ct10, then turns.

Clean confirmed result, ct10/K8 with the disjoint reference:

| seed | RF OOB | GBM 5-fold | Raw-NN |
|------|--------|------------|--------|
| 42   | 0.6001 | 0.5788     | 0.5200 |
| 43   | 0.5890 | 0.5987     | 0.5157 |
| 44   | 0.6000 | 0.5895     | 0.5062 |

NEW PROJECT BEST (honest): 0.596 +/- 0.005. Leakage accounted for roughly
0.01-0.03 of the apparent gain; the rest is real distribution matching
that transfers to unseen humans. Config: EVENT_SIR=8 EVENT_CHOICE_TEMP=10
EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5
EVENT_DUR_STD=1.0. Remaining knobs: K=16, EVENT_SIR_TEMP, DUR_STD under
selection, stronger discriminator.

## Boundary-speed hypothesis tested and refuted; K/trees probes (July 5 night)

Two knob probes at seed 42, K=16 ct=10 on the clean disjoint reference:
plain 0.5892 RF / 0.5871 GBM, judge with 600 trees 0.5881 / 0.5758. K=16
edges out K=8 (0.6001 s42) and judge strength is not the bottleneck.

First-principles check of the detector's remaining signal. The scary
correlation table (human mean_acc x everything at +1.000) is an artifact:
Spearman is -0.16 and Pearson drops to -0.11 once the top 1% of |mean_acc|
is excluded. A handful of short segments with enormous cut speeds pin
every Pearson entry to 1.0. Do not chase correlation-gap tables built on
these features again without rank statistics.

The mid-flight hypothesis behind WS5 is dead for this model family.
Human pool segments do start mid-flight (median first-step speed 250 px/s
after the 125Hz resample, 90% above 10 px/s), but the event model, trained
on those same segments, reproduces this: synthetic median 279 px/s, 89%
above 10 px/s, last-step 101 vs 83 px/s. mean_acc quantiles line up
through p95. No boundary conditioning or warm-start needed; measuring
before building saved a workstream.

One residue found: synthetic mean_acc shows a point mass at exactly zero
(p75 = -4e-12), which happens when first and last resampled steps are both
stationary. Humans may or may not share the atom; diag_macc_atom.py
measures it. If synth-only and sizable, an RF split at |mean_acc| < eps is
free AUC for the detector; fix would be decode-side tick trimming.

Overnight queue (run_overnight_sir.sh, all seed 42, K=16 ct=10 base):
SIR_TEMP=0.7, TH_TEMP=1.15 (hotter heading proposal for the curvature
tail, SIR filters), FEAT_BW=0.5 (wider character proposal), K=32, ct=12,
DUR_STD=1.25. Also added: [sir] ESS instrumentation per eval (median
effective sample size of the lottery, prints in every log) and a
DUR_EMPIRICAL=1 knob on DurationModel that resamples actual per-bin
log-durations instead of a Gaussian fit (conditional skew kept, same
existing conditioning prior, T3-legal, probe queued for July 6).

Atom follow-up (diag_macc_atom.py, n=400): humans share the exact-zero
mean_acc atom (synth 4.5% vs human 2.5%, within sampling noise; mean_jerk
atoms match too). Not an exploitable detector split. Both boundary leads
are now closed; the remaining gap is curvature-tail and duration shaped.

## Overnight sweep, first result: sharper SIR weights win (July 5, 22:20)

sir16_stemp07 (EVENT_SIR=16, EVENT_SIR_TEMP=0.7, seed 42): RF OOB 0.5649,
RF 5-fold 0.5740, GBM 0.5628, Raw-NN 0.5520. Same-seed comparison points:
sir16 at temp 1.0 was 0.5892, sir8 baseline 0.596. Sharpening the
selection weights (dividing log-odds by 0.7 before the Gumbel lottery) is
worth about -0.024 on its own, the largest single-knob gain since SIR
itself. ESS check says selection is not collapsing: median per-spec ESS
10.9 of 16 (p10 6.3), so the lottery still spreads mass over most
candidates. There should be room to push temp lower before variety
collapses; a stemp=0.5 probe belongs in the July 6 queue.

Wasserstein table after selection: curvature_std and curvature_mean still
top (0.13-0.15 range), then max_deviation, movement_duration. RF
importances have flattened further (top importance 0.064, movement_duration).
The queue died after this probe: the still-running bash re-read the
script file after it had been edited mid-run and hit a phantom syntax
error. Relaunched at 22:21; skip-if-log-exists resumed at probe 2.
Lesson repeated: never edit a shell script while a bash instance is
executing it; copy-then-edit instead.

## Overnight sweep, probe 2: theta temperature is a dead knob (July 5, 23:05)

sir16_tht115 (EVENT_SIR=16, EVENT_TH_TEMP=1.15, seed 42): RF OOB 0.5887,
RF 5-fold 0.5928, GBM 0.5844, Raw-NN 0.5550. Statistically identical to
the same-seed sir16 run at default theta temperature (0.5892), so heating
the heading distribution buys nothing even with SIR available to filter
the extra variety. Worse, the curvature Wasserstein gaps widened
(curvature_std 0.207, curvature_mean 0.167 versus roughly 0.13-0.15
without the knob): the hotter heading sampling produces wigglier paths
than humans and SIR cannot fully select its way back. ESS median 12.5 of
16, higher than the stemp07 run as expected since weights here use
temp 1.0.

Verdict: leave EVENT_TH_TEMP alone. The win so far is sharper selection
(stemp 0.7), not hotter proposals. Next probes: bw05 (judge bandwidth),
sir32, ct12, dur125, duremp.

## Overnight sweep, probe 3: wider judge bandwidth helps a little (July 5, 23:45)

sir16_bw05 (EVENT_SIR=16, EVENT_FEAT_BW=0.5, seed 42): RF OOB 0.5761,
RF 5-fold 0.5962, GBM 0.5860, Raw-NN 0.5262. About -0.013 versus the
same-seed temp-1.0 baseline (0.5892) on the headline number, though the
cross-validated RF and GBM barely moved, so the gain is softer than it
looks. The Raw-NN drop to 0.526 is the best raw-detector number seen at
K=16. ESS median 13.2 of 16: widening the judge's KDE bandwidth smooths
the log-odds, making weights less peaky, which is the same direction as
raising selection temperature. That makes bw05 partially redundant with
stemp, not orthogonal.

Standing question for the July 6 combo probe: does bw05 add anything on
top of stemp07, or are they two handles on the same smoothness dial?
Given stemp07 alone (0.5649) beats bw05 alone (0.5761), sharpness of
selection is the stronger lever and the combo test should hold stemp=0.7
fixed and vary bandwidth around the default.

## Overnight sweep, probe 4: K=32 without sharpening buys nothing (July 6, 01:00)

sir32_ct10 (EVENT_SIR=32, selection temp 1.0, seed 42): RF OOB 0.5962,
RF 5-fold 0.6040, GBM 0.5948, Raw-NN 0.5304. Slightly worse than sir16 at
the same temperature (0.5892), within single-seed noise, at double the
generation cost (77 minutes versus 39). ESS median 25.9 of 32, which is
81 percent, essentially the same relative spread as K=16 temp 1.0 (78
percent). That is the tell: at temperature 1.0 the Gumbel lottery keeps
mass spread across the pool, so doubling the pool mostly adds more
mediocre candidates to the raffle rather than concentrating on the best
ones. More candidates only pay off if selection is sharp enough to
exploit them.

Direct implication for the July 6 queue: the interesting untested cell is
K=32 with stemp 0.7 or 0.5. If sharp selection plus a bigger pool
compounds (stemp07 alone is 0.5649), that combination is the natural
recipe for the distillation corpus, where per-sample cost matters less
than corpus quality.

## Overnight sweep, probe 5: choice temp 12 is a dead knob (July 6, 01:40)

sir16_ct12 (EVENT_SIR=16, EVENT_CHOICE_TEMP=12, seed 42): RF OOB 0.5948,
RF 5-fold 0.6017, GBM 0.5905, Raw-NN 0.5416. No improvement over ct=10
(0.5892 same seed); if anything a hair worse, within noise. Same story as
th_temp: heating the proposal distribution does not help when the judge
already has enough variety to choose from at ct=10. EVENT_CHOICE_TEMP
stays at 10.

Proposal-side knobs are now 0 for 3 tonight (th_temp, K=32-soft, ct=12).
Selection-side knobs are 2 for 2 (stemp, bandwidth). The sweep is telling
one consistent story about where the remaining signal lives.

## Overnight sweep, probe 6: extra duration jitter hurts (July 6, 02:15)

sir16_dur125 (EVENT_SIR=16, EVENT_DUR_STD=1.25, seed 42): RF OOB 0.6132,
clearly worse than the 0.5892 same-seed baseline at DUR_STD=1.0. Widening
the duration prior pushes movement_duration off the human marginal faster
than the judge can select it back. Proposal-side knobs are now 0 for 4
tonight. DUR_STD stays at 1.0.

## Overnight sweep, probe 7 and wrap-up (July 6, 02:55)

sir16_duremp (EVENT_SIR=16, DUR_EMPIRICAL=1, seed 42): RF OOB 0.5813,
RF 5-fold 0.5923, GBM 0.5739, Raw-NN 0.5312. A small gain over 0.5892,
and the mechanism is visible: movement_duration Wasserstein fell to
0.040 from the 0.055 to 0.074 range of every other run tonight. The
empirical duration prior fixes the duration marginal directly instead of
asking the judge to fix it by selection. Technically this is the one
proposal-side change that helped, but it is a prior correction toward the
human distribution, not added heat, which is exactly why it fits the
pattern rather than breaking it.

Full sweep, seed 42, baseline sir16 temp 1.0 = 0.5892:

| probe            | knob                    | RF OOB | verdict |
|------------------|-------------------------|--------|---------|
| sir16_stemp07    | selection temp 0.7      | 0.5649 | win, big |
| sir16_bw05       | judge bandwidth 0.5     | 0.5761 | small win |
| sir16_duremp     | empirical duration prior| 0.5813 | small win |
| sir16_tht115     | heading temp 1.15       | 0.5887 | dead |
| sir16_ct12       | choice temp 12          | 0.5948 | dead |
| sir32_ct10       | K=32, soft weights      | 0.5962 | dead, 2x cost |
| sir16_dur125     | duration jitter 1.25    | 0.6132 | hurts |

One story: proposal heat fails, selection sharpness and prior fixes win.
The generator already drafts human-like paths often enough; remaining
gains come from choosing among drafts more decisively and from aligning
the conditioning priors with the human marginals.

Combo queue launched at 03:00 (run_combo_sir.sh): stemp05 (sharpness
curve), stemp07+duremp (stack the winners), sir32+stemp07 (pool size now
that selection can exploit it), stemp07+iter2 (iterated SIR). Results by
about 06:30, then the recipe gets locked for the distillation corpus.

## Combo queue, probe 1: sharpness curve bottoms near 0.7 (July 6, 03:40)

sir16_stemp05 (selection temp 0.5, seed 42): RF OOB 0.5704, RF 5-fold
0.5814, GBM 0.5788, Raw-NN 0.5473. Slightly worse than stemp 0.7
(0.5649), and ESS confirms why: median per-spec ESS fell to 8.1 of 16
with p10 at 3.3, meaning for a tenth of the specs the lottery is
effectively choosing among three candidates. Push sharpness too far and
selection starts cloning the judge's favorite mode, which the detector
can price. The sharpness curve is shallow between 0.5 and 0.7 but 0.7 is
the operating point: nearly all the gain, twice the effective variety.

EVENT_SIR_TEMP locked at 0.7. Remaining combos test what stacks on top.

## Combo queue, probe 2: the winners stack, new single-seed best (July 6, 04:20)

sir16_stemp07_duremp (selection temp 0.7 + empirical duration prior,
seed 42): RF OOB 0.5607, RF 5-fold 0.5728, GBM 0.5702, Raw-NN 0.5192.
Best single-seed number of the project so far, edging stemp07 alone
(0.5649). The gains compose because they act on different parts of the
pipeline: duremp fixes the duration marginal at the prior
(movement_duration drops out of the top Wasserstein list entirely), which
frees the judge's selection budget for shape. Raw-NN at 0.5192 is the
closest any raw-trajectory detector has come to chance. ESS median 10.8
of 16, unchanged from stemp07 alone, so no variety cost.

Curvature_std (0.19) and curvature_mean (0.15) remain the last big
marginal gaps. Two probes left: sir32+stemp07 and iter2.

## Combo queue, probe 3: pool size is saturated at 16 (July 6, 05:35)

sir32_stemp07 (K=32 + selection temp 0.7, seed 42): RF OOB 0.5705,
RF 5-fold 0.5941, GBM 0.5617, Raw-NN 0.5257. No better than K=16 at the
same temperature (0.5649), at double the generation cost. The hypothesis
from the overnight sweep, that a bigger pool would pay once selection was
sharp enough to exploit it, is refuted: even with sharp weights the extra
16 candidates add nothing the judge wants. The generator's 16 draws
already cover the region of feature space the judge can distinguish;
past that, candidate 17 through 32 are redundant samples from the same
distribution. K stays at 16, which also keeps the distillation corpus
cost at half.

## Combo queue wrap-up: recipe locked (July 6, 06:15)

sir16_stemp07_iter2 (two SIR rounds at temp 0.7, seed 42): RF OOB 0.5755,
worse than one round (0.5649). The log-odds tell the story: round two
lifted mean candidate log-odds from -2.36 to -1.78, so the survivors do
score better under the judge, yet the detector AUC got worse. Optimizing
harder against a fixed judge past one round trades real variety for
judge-pleasing, and the eval detector (a fresh RF the judge has never
seen) prices the difference. Same lesson as stemp 0.5 from the other
side: there is an optimum amount of selection pressure and one tempered
round at 0.7 is it.

Combo results, seed 42:

| probe                | RF OOB | verdict |
|----------------------|--------|---------|
| stemp07 + duremp     | 0.5607 | LOCKED RECIPE |
| stemp07 (reference)  | 0.5649 | |
| stemp05              | 0.5704 | too sharp |
| sir32 + stemp07      | 0.5705 | pool saturated |
| stemp07 + iter2      | 0.5755 | judge-overfit |

Locked SIR recipe for the distillation corpus: EVENT_SIR=16
EVENT_SIR_TEMP=0.7 DUR_EMPIRICAL=1 on event_polar_4m_fc_v2.pt with
EVENT_ORDER=gumbel EVENT_CHOICE_TEMP=10 EVENT_SNAP=2.5 EVENT_DUR_STD=1.0.
Single-seed 0.5607; multi-seed confirmation deferred to the protected
final confirmation window so the GPU can start on the distillation
corpus now.

## Distillation build: corpus generator and trainer (July 6, 06:50)

The locked SIR recipe is an inference-time system: draw 16, let a judge
pick. Distillation bakes the picking into the weights so a single draft
lands where the selected draft used to. Two new pieces:

training/make_distill_corpus.py generates the corpus with the locked
recipe (K=16, selection temp 0.7, empirical duration prior, fc_v2). The
design decision that matters: the corpus stores the winning candidate's
TOKENS, not its decoded pixels. The judge scores the fully decoded
trajectory, snap and round included, because that is what a detector
sees; but training happens in token space with the exact pretraining
objective, and snap/round stay serving-time decode steps. Re-encoding
decoded pixels back to tokens would bake the decode artifacts into the
data and then apply them again at serving. 20,000 specs in ten
2000-spec blocks, each block judged by a fresh GBM against the disjoint
4000-row human reference (never the eval humans), one crash-safe shard
per block. Smoke test: 64/64 specs selected, ESS median 9.3 of 16,
1.17 s per spec on the 4070, so the full corpus takes about 6.5 hours.
Batch 512 was tried and changed nothing; the GPU is saturated at 256.

training/train_events_polar_distill.py fine-tunes fc_v2 on the corpus:
same losses as pretraining, dt head frozen (timing already matches
humans), feature vector teacher-forced from each corpus trajectory's own
tokens through the same differentiable pipeline that built the bank, and
the checkpoint's real-human feature bank passes through untouched, so
sampling still draws characters from real data. Low learning rate 1e-5,
default 3000 steps, snapshots every 500 so the eval can pick the best
point on the gain-versus-drift curve. A --real-frac flag can mix real
batches back in as an anchor if pure self-corpus training drifts.

## Distillation verdict: fails at every depth, and the control finds a new pure best (July 6, 13:20)

The corpus came out healthy: all 20,000 specs selected, log-odds and ESS
stats flat across all ten blocks (logw mean -2.35, ESS median ~10.7 of
16), 2300 s per block. Fine-tuning ran 3000 steps in 17 minutes, loss
easing 3.54 to 3.22 with no instability. Then the snapshot evals, pure
model, no SIR, seed 42, all with the locked serving env (gumbel, choice
temp 10, snap 2.5, dur std 1.0, empirical duration prior):

| checkpoint | RF OOB AUC |
|---|---|
| fc_v2 control (no distillation) | 0.6470 |
| distill s500 | 0.6802 |
| distill s1000 | 0.6965 |
| distill s2000 | 0.6936 |
| distill s3000 | 0.6932 |

Every distilled checkpoint is worse than the model it started from, and
the damage is nearly done by step 500. Distillation on this
architecture does not transfer the judge's preference; it just moves
the model off the real-data manifold it was pretrained on. The likely
mechanism is the same one that killed the adversarial and DM fine-tunes:
the masked-token objective teaches marginal token distributions, but SIR
winners win on joint feature combinations of the whole decoded
trajectory. Training on winners nudges marginals the sampler then
recombines freely, losing exactly the joint structure that made them
winners, while the model drifts away from real-token statistics it was
anchored to.

The control run is the real news. Pure fc_v2 with DUR_EMPIRICAL=1
scores 0.6470 at seed 42, against 0.675 +/- 0.002 without it. The
empirical duration prior had only ever been tested inside the SIR
system, where it was one of the three selection-side wins. It turns out
it helps the bare model just as much: the fitted lognormal duration
prior was feeding the model out-of-distribution durations, and fixing
the prior is a T3-legal conditioning correction, no picker involved.
New best pure single-net result, pending multi-seed confirmation.

Two follow-ups queued: seeds 43 and 44 of the pure control, and one
distillation rerun with --real-frac 0.5 mixing real human batches back
in as an anchor. If the anchored version still lands above the control,
CE-based distillation is dead on this architecture, not just drifted.

## Pure model + empirical duration prior: confirmed at 3 seeds (July 6, 13:25)

Seeds 42/43/44 of pure fc_v2 with DUR_EMPIRICAL=1 (no SIR): 0.6470,
0.6544, 0.6531. New confirmed pure single-net best 0.652 +/- 0.003,
against 0.675 +/- 0.002 for the same model without the prior fix. The
whole gain comes from conditioning the model on durations drawn from
the empirical (distance-binned) human duration distribution instead of
the fitted lognormal. Costs nothing at serving time and is pure T3.

## Anchored distillation also fails: the axis is closed (July 6, 13:50)

The --real-frac 0.5 rerun mixes a real human batch in for half the
steps as an anchor against self-training drift. Seed 42, pure model:
s500 0.6871, s1000 0.6815, s3000 0.6752, all still clearly worse than
the 0.6470 no-distillation control. The anchor changed the shape of
the damage (deeper is now mildly better, the opposite of the pure-
corpus run) but not the sign. Verdict: CE-based distillation of SIR
selection is dead on this architecture, drifted or anchored. The
judge's preference lives in joint feature structure of whole decoded
trajectories, and per-position masked-token training cannot receive
it. This is the same wall that killed the DM and adversarial
fine-tunes, seen from a third side. The picker stays an inference-time
system: "undetectable system" (0.5607 single-seed, 0.596 +/- 0.005
honest) versus "undetectable model" (0.652 +/- 0.003).

## Preference learning: the one distillation variant with a different mechanism (July 6, 14:30)

Imitation distillation is closed, but it tested only one way of using
the judge's signal: maximize likelihood of winners. The untested way is
preference learning: for each spec, keep the judge's best AND worst of
the K candidates, and train with the Diffusion-DPO objective, lower the
denoising loss on the winner and raise it on the loser relative to a
frozen reference copy of the model. Three reasons this can work where
imitation could not. The gradient is a contrast between two whole
sequences from the same pool, which is the trajectory-level judgment
SIR actually applies, not a marginal-token imitation target. The
reference model anchors the update, an implicit KL leash against the
manifold drift that both plain and anchored imitation showed. And no
sampling happens inside the training loop, so the straight-through
Gumbel gradient path that killed all three adversarial fine-tunes is
not involved.

Build: make_distill_corpus.py grew a DISTILL_SAVE_LOSER=1 mode (pairs
get max contrast: winner = argmax judge log-odds, loser = argmin, no
lottery), training/train_events_polar_dpo.py implements the paired
loss with one shared corruption level per pair for variance reduction,
dt head frozen as always. Smoke test: DPO loss starts at log 2 = 0.70
and preference accuracy at 0.5, exactly what an untrained-but-correct
DPO setup shows when policy equals reference; judge gap median 2.45
logits per pair, so the contrast signal is there. This moves the July 7
adversarial slot up a day: the classic adversarial axis is dead three
ways, and this is the strongest remaining mechanism against the same
wall. Pipeline: 6000-spec pair corpus (~2h), 1500 training steps,
snapshot evals every 250 steps, queued behind the two SIR probes.

Hardware note, July 6 14:15: fourth bluescreen of the week under GPU
load (bugcheck 0x1E, access violation; the three during the 4M pretrain
each showed different codes, the signature of flaky RAM or heat rather
than software). Cost 25 minutes of one eval. Every pipeline here is
crash-safe by construction (skip-if-done shards and logs), which is the
reason these losses stay small; the whole queue relaunched sequentially
at 14:41.

## Two stacking probes on the locked recipe (July 6, 16:00)

Wider character bandwidth (EVENT_FEAT_BW=0.5, a clear solo winner at
0.5761) stacked onto the locked recipe: 0.5619 vs 0.5607, a tie. The
sharper selection temperature already harvests whatever the wider
draws offered; the two knobs were picking the same fruit.

Per-candidate duration diversity (EVENT_SIR_DUR_DIVERSE=1, new knob):
all K candidates previously shared one duration draw per spec, so the
judge could choose among paths but never among durations, a feature
family the detector weights heavily. Resampling the duration for each
candidate: 0.5589 vs 0.5607. Within single-seed noise, but it is the
best single-seed number on record, the mechanism is sound, and the
cost is zero, so it joins the recipe pending the multi-seed
confirmation. Seeds 43 and 44 queued for tonight.

## Preference learning verdict: the judge teaches ranking, not generation (July 6, 18:25)

The pair corpus came out healthy (6,000 max-contrast pairs, judge gap
median 1.62 logits) and the DPO training itself worked exactly as
designed: preference accuracy climbed from 0.50 to 0.87 and the margin
between winner and loser losses grew steadily. The model genuinely
absorbed the judge's ranking. Then the generation evals, pure model,
seed 42, against the 0.6470 control:

| checkpoint | RF OOB AUC |
|---|---|
| fc_v2 control | 0.6470 |
| dpo s250 | 0.7168 |
| dpo s500 | 0.8626 |
| dpo s750 | 0.9327 |
| dpo s1000 | 0.9602 |
| dpo s1500 | 0.9782 |

Monotonic collapse, and much steeper than imitation distillation ever
was. This is Goodhart's law in weight space: pushing loser sequences
down does not remove the loser modes, it warps the marginal token
distributions everywhere, and the samples drift far off the human
manifold while the internal ranking keeps improving. The judge's
signal is real but it is only usable as a FILTER over finished
trajectories, never as a gradient into this architecture's weights.

That closes the second half of the distillation question, and with it
the whole teach-the-model family: imitation (plain and anchored),
three adversarial variants, and preference learning all fail on the
same wall from four different directions. The clean statement for the
writeup: on a masked-token event model, trajectory-level judgment
compresses into weights in the ranking direction but not in the
generation direction. Selection stays at inference time.

## New honest best: 0.568 +/- 0.010 at three seeds (July 6, 19:40)

The full recipe, fc_v2 + gumbel ct10 + snap 2.5 + dur_std 1.0 +
empirical duration prior + SIR K=16 at temp 0.7 + per-candidate
duration diversity, across seeds 42/43/44:

| seed | RF OOB | RF 5-fold | GBM 5-fold | Raw-NN |
|---|---|---|---|---|
| 42 | 0.5589 | 0.5720 | 0.5426 | 0.5091 |
| 43 | 0.5781 | 0.5753 | 0.5720 | 0.5395 |
| 44 | 0.5681 | 0.5885 | 0.5816 | 0.5320 |

RF OOB 0.568 +/- 0.010, a 0.028 improvement over the previous honest
best (0.596 +/- 0.005, the pre-sweep ct10/K8 recipe). The raw-NN
detector sits at 0.51-0.54, close to chance. Every gain in the recipe
is selection-side: sharper tempering, corrected duration prior, and
duration choice per candidate. The gap to 0.50 is now 0.068 of AUC on
the primary detector.

## The final lever: selection as a set-level problem (July 6, 20:10)

With the teach-the-model family closed, everything that ever moved the
number lives on the selection side. Working through why SIR still
leaves 0.068 on the table exposed a structural limit: SIR picks each
trajectory independently, but the detector never sees trajectories one
at a time. It trains on the whole selected population against the
whole human sample. Independent per-spec picks cannot trade one spec's
choice against another's to fix a marginal the entire pool overshoots,
and the tempered lottery itself distorts the selected distribution in
a way a freshly trained detector can spot. The eval is a distribution
test, so selection should optimize the distribution, not the item.

The mechanics make this cheap to explore. The pool of K=16 candidates
per spec is the expensive part (one GPU run); which candidate wins is
just an index. New plumbing: EVENT_POOL_SAVE caches every decodable
candidate (trajectory + features + owning spec) during a normal eval
run, and EVENT_POOL_LOAD / EVENT_POOL_PICKS replays an arbitrary
selection through the honest evaluator in about three minutes, no GPU
sampling. Selection strategies iterate offline in seconds.

selection_lab.py implements three: the current per-item lottery as a
calibration baseline; iterated adversarial reselection (fit a
discriminator between the reference and the current selected set,
accumulate log-odds, re-pick, repeat, which gives selection the
set-level feedback loop it lacked); and a histogram-matching exchange
that directly minimizes the L1 gap between selected-set and reference
histograms over all 18 feature marginals plus the twelve most
correlated feature pairs, the exact structures an axis-aligned forest
splits on. Honesty protocol: the 4000-row disjoint reference is split
in half, all fitting uses half A, and a proxy RF OOB AUC against the
untouched half B is reported for every strategy. Final numbers come
only from evaluate.py replay, where the human class is the real eval
sample that no stage of selection has ever seen. Seed-42 pool
generating now; seeds 43/44 queued for overnight so a multi-seed
confirmation is possible tomorrow.

## Offline round one: the residual signal is not in the marginals (July 6, 21:05)

Pool generation reproduced the recipe honestly (0.5743 at seed 42,
within the seed's run-to-run range) and cached 31,993 candidates over
2000 specs. First lab pass, proxy RF OOB AUC against the untouched
reference half B:

| strategy | proxy AUC |
|---|---|
| per-item SIR lottery (baseline) | 0.5566 |
| histogram exchange from SIR init | 0.5612 |
| histogram exchange from greedy init | 0.5723 |
| random-of-K | 0.6515 |
| greedy argmax reselection | 0.7619 |

Two lessons. First, the proxy is trustworthy: replaying the lab's SIR
picks through the honest evaluator gave 0.5699, and the histogram
picks 0.5696, both consistent with the recipe's seed-42 range. Second,
and more important: the exchange drove the summed histogram gap down
by a factor of five (all 18 marginals plus the twelve most correlated
pairs nearly indistinguishable from the reference) and the detector
did not care. The residual 0.06 of detection does not live in feature
marginals or pairwise structure. It lives in higher-order joint
structure that every candidate in a spec's pool shares, which is why
per-item argmax collapses (0.76): concentrating picks on the judge's
favorite region manufactures exactly the kind of joint artifact a
fresh detector finds instantly.

Round two targets the joint distribution directly: a multi-bandwidth
kernel MMD exchange (sensitive to interactions of every order), and a
trust-region version of the adversarial reselection loop that moves
only the best 15 percent of specs per round against a judge trained on
the current selected set. If neither moves the proxy below the SIR
baseline, the conclusion is support deficiency, the pool of 16
candidates simply does not contain a human-distributed subset, and the
remaining lever is a larger K.

# External validation

The headline result (RF-OOB near 0.50, synthetic trajectories indistinguishable
from held-out humans) is proven within the five datasets that built the
training pool. This folder asks the next question: does a detector trained on
the same 18-dim feature vector separate our synthetics from mouse movements it
has never seen, recorded on hardware and in software we never touched. The
answer, on both datasets tried, is no more than it separates two human
datasets from each other. Full numbers and the day's reasoning are in the July
8 entry of EXPERIMENTS.md at the repo root.

## Datasets

**AdSERP** (Latifzadeh, Gwizdka and Leiva, SIGIR 2025). 47 users, browser
mousemove and click events during search tasks, about 50 Hz. Zenodo record
15236546, file `mouse-movement-data.zip`, CC BY 4.0. Canonical segmentation
(the same pause-gap and distance rule `setup_data.py` uses for the training
pool) yields 11,580 valid movements.

**M4D** (Iliou et al., 2021). 94 human sessions across two phases of a
web-bot-detection study, about 59 Hz. Fetched from `m4d.iti.gr`, CC
BY-NC-SA 4.0. Because of the NonCommercial clause we do not redistribute a
copy; `fetch_external_data.py` always pulls it from source. The same
segmentation rule yields 17,018 valid movements.

Neither dataset was built for this purpose. Both happen to log timestamped
mouse coordinates at a rate close enough to our resample target to compare on
equal footing.

## Reproducing

```
.venv/Scripts/python.exe external_validation/fetch_external_data.py
.venv/Scripts/python.exe external_validation/adserp_features.py
.venv/Scripts/python.exe external_validation/m4d_features.py
.venv/Scripts/python.exe external_validation/validate_adserp.py
.venv/Scripts/python.exe external_validation/validate_m4d.py
.venv/Scripts/python.exe external_validation/ablate_curvature.py
```

`fetch_external_data.py` downloads and unzips both datasets into
`external_data/` (gitignored, so raw data never enters version control). The
two `_features.py` scripts run the canonical segmentation and the repo's
existing `features.extract_feature_matrix`, then cache a 2,000-movement
feature sample per dataset as a `.npy` file in `external_data/`. The
`validate_*.py` scripts load those caches plus the cached selected-synthetic
feature sets and run the same RF-OOB, RF 5-fold, and GBM 5-fold detector
suite `evaluate.py` uses. `validate_adserp.py` first reproduces the published
seed-42 headline number through its own loading path as a sanity check, and
halts before running anything else if that check fails.

## Results (RF-OOB AUC)

| comparison | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| AdSERP vs selected synthetics | 0.9479 | 0.9470 | 0.9496 |
| M4D vs selected synthetics | 0.920 | 0.922 | 0.920 |
| AdSERP vs internal held-out humans | 0.9473 | | |
| M4D vs internal held-out humans | 0.918 | | |
| M4D vs AdSERP (humans vs humans) | 0.805 | | |

The synthetics are not hiding from these two detectors; an AUC near 0.95 (or
0.92) means the detector separates them from AdSERP or M4D almost perfectly.
But that number is not the interesting one on its own, because AdSERP and M4D
separate almost as cleanly from our own held-out humans, and from each other.
What matters is the gap between "external vs synthetic" and "external vs our
humans," and that gap is 0.001 to 0.002 on both datasets across all three
seeds: a tie within noise. Selected synthetics sit exactly where our own
humans sit, relative to a dataset recorded somewhere else.

A curvature ablation (dropping the two features with the largest distribution
distance to either external set) leaves this picture unchanged, so no single
feature carries the result; velocity, acceleration, jerk, and angular-velocity
scale differences do most of the separating work instead.

This is a claim about indistinguishability under a fixed recording setup,
matched to the training pool's capture characteristics (device polling rate,
browser event timing, resample target). Whether the model transfers across a
change in recording setup is a different and apparently much harder question,
one that even two human datasets fail at each other.

# Training Scripts

Reference implementations showing how the trajectory synthesis models were trained.
These are provided for transparency and reproducibility, not as turnkey runnable scripts.

**Prerequisites:** These scripts require preprocessed pool data (`.npy` files) that are
not included in this repository. The pool data consists of millions of recorded human
mouse trajectories, preprocessed into position/timestamp/condition arrays.

See [METHODOLOGY.md](../METHODOLOGY.md) for the full pipeline description.

## Scripts

| Script | Description |
|---|---|
| `prepare_training_data.py` | Preprocesses raw trajectory pool into fixed-length position/timestamp arrays with train/val/test splits |
| `prepare_vqvae_data.py` | Extracts (dx, dy) displacement pairs at 125Hz from prepared training data for VQ-VAE training |
| `train_vqvae.py` | Trains the VQ-VAE motion tokenizer with k-means codebook initialization and dead-entry reset |
| `tokenize_trajectories.py` | Converts trajectories into discrete token sequences using the trained VQ-VAE |
| `train_transformer.py` | Trains the autoregressive trajectory transformer on tokenized sequences |
| `train_cfm.py` | Trains the Conditional Flow Matching model (OT-CFM) on joint position + timing data |
| `eval_holdout.py` | Holdout discriminator suite with 4 independent classifiers for evaluating trajectory quality |

## Pipeline Order

```
prepare_training_data.py
    -> prepare_vqvae_data.py
        -> train_vqvae.py
            -> tokenize_trajectories.py
                -> train_transformer.py
    -> train_cfm.py (parallel path)

eval_holdout.py (runs independently against any generator)
```

## Coordinate Normalization

`prepare_training_data.py` applies two transforms to every trajectory in order:

1. **Origin translation**: subtract the start point so the trajectory begins at (0, 0).
2. **Distance normalization**: divide all coordinates by `total_dist = np.hypot(end_x, end_y)`,
   computed after translation. This puts every trajectory into a unit-scale space where
   the endpoint sits at distance 1 from (0, 0).

The **DDPM checkpoint** was trained on this normalized representation. The **CFM checkpoint**
was trained on origin-translated (but not distance-normalized) raw pixel coordinates.
The **VQ-VAE** operates on displacement vectors `(dx, dy)` at 125 Hz, independent of
normalization. The inference code for each experiment handles the appropriate
coordinate convention.

## Setup

```bash
pip install -e .  # from repo root
```

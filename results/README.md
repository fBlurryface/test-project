# Results

This directory contains the quantitative artifacts currently bundled with the submission.

## Training

`training_metrics.json` records grouped held-out validation for the deployed 5D→3D behavior-cloning model.

Key values:

| Metric | Value |
| --- | ---: |
| Mean vector RMSE | **3.8669 mm / 100 ms** |
| Left active-direction accuracy | **92.2%** |
| Up active-direction accuracy | **84.6%** |
| Forward active-direction accuracy | **97.0%** |

These folds are used during model development/model selection and are reported as grouped held-out validation rather than a pristine final test benchmark.

## Example hybrid rollouts

`validated_hybrid_rollouts/` contains four previously recorded Panda hybrid-policy CSVs.

| Rollout | Preset | Observed outcome in log |
| --- | --- | --- |
| `...092348.csv` | baseline | reaches `SETTLE`, `phase_pierced=1`, ~78 mm lift |
| `...092525.csv` | random | fails before pierce; remains `RED_LOST_HOLD` |
| `...092554.csv` | random | reaches `SETTLE`, `phase_pierced=1`, ~77 mm lift |
| `...092636.csv` | baseline | reaches `SETTLE`, `phase_pierced=1`, ~75 mm lift |

This is an **illustrative four-rollout set**, not a statistically meaningful success-rate benchmark.

## Media

Before final submission, add the clearest short real-LeArm demonstration and Panda hybrid success video/GIF under `results/media/`, then surface the best one near the top of the root README.

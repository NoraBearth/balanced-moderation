# Changelog

## 0.1.0 (unreleased)

First release.

- `BGATE` estimator for GATE and BGATE differences with double/debiased
  machine learning (Bearth & Lechner, 2024), binary treatment and moderator.
- Cross-fitted propensity score calibration (Ballinari & Bearth, 2024):
  Venn-Abers (default), Platt, beta and isotonic calibration.
- Warning when estimated propensity scores are below 0.01 or above 0.99.
- `make_moderation_data` and `true_moderation_effects` for the simulation
  designs of Bearth & Lechner (2024).

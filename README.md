# balanced-moderation

**Balanced moderation analysis with causal machine learning.**

`balanced_moderation` estimates how the effect of a treatment differs between two
groups (for example women and men) while holding the distribution of other
characteristics fixed. It implements the *balanced group average treatment
effect* (BGATE) of [Bearth & Lechner (2024)](https://arxiv.org/abs/2401.08290),
estimated with double/debiased machine learning (DML), and calibrates
propensity scores with the cross-fitted procedure of
[Ballinari & Bearth (2024)](https://arxiv.org/abs/2409.04874).

## Why BGATE?

Suppose a training programme raises wages more for men than for women. Is that
because of gender, or because men and women in the sample differ in education,
experience or occupation? The usual group average treatment effects (GATEs)
cannot tell these apart: their difference mixes the moderation effect with
differences in the covariate distributions of the two groups.

The BGATE fixes the distribution of chosen **balancing variables** W to be the
same in both groups:

```
BGATE(z) = E_W[ E[ Y(1) - Y(0) | Z = z, W ] ]
```

so the difference `BGATE(1) - BGATE(0)` is the moderation effect of Z
**holding the distribution of W fixed**. Without balancing variables you get
the ordinary GATE difference, so you can compare both.

## Installation

```bash
pip install balanced-moderation
```

Requires Python 3.10 or newer. Dependencies: numpy, pandas, scikit-learn,
scipy and venn-abers.

## Quickstart

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from balanced_moderation import BGATE, make_moderation_data

df = make_moderation_data(n=4000, design="z_affects_x", random_state=42)
controls = [f"x{j}" for j in range(10)]

model = BGATE(
    outcome_model=RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0),
    propensity_model=RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0),
    n_folds=5,
    random_state=42,
)

result = model.fit(
    df,
    y="y",              # outcome
    d="d",              # binary treatment
    z="z",              # binary moderator
    x=controls,         # control variables (confounders)
    balance=controls,   # balancing variables W (a subset of x)
)
print(result.summary())
```

```
=============================================================================
Balanced GATE (BGATE) - balanced over: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9
Moderator: z    Observations: 4000
=============================================================================
            estimate  std_err  t_stat  p_value  ci_95_lower  ci_95_upper
BGATE(z=1)    1.4175   0.0944 15.0143   0.0000       1.2325       1.6026
BGATE(z=0)    1.0389   0.0584 17.7982   0.0000       0.9245       1.1533
Difference    0.3786   0.1116  3.3932   0.0007       0.1599       0.5973
=============================================================================
```

In this simulated design the true BGATE difference is 0.484 and the true GATE
difference is 0.414: because the moderator also shifts the distribution of
`x5`, the two parameters differ.

Leave out `balance=` to estimate the GATE difference instead. A complete
example is in [`examples/quickstart.py`](examples/quickstart.py).

### Working with the results

| | |
|---|---|
| `result.effect` | the moderation effect, `BGATE(z=1) - BGATE(z=0)` |
| `result.std_err`, `result.pvalue` | its standard error and two-sided p-value |
| `result.conf_int(alpha=0.05)` | its confidence interval |
| `result.to_frame()` | all three estimates as a pandas DataFrame |
| `result.summary()` | a printable table |
| `result.scores` | the observation-level scores (their means are the estimates) |

## Settings

| Parameter | Default | Meaning |
|---|---|---|
| `outcome_model` | random forest | any scikit-learn regressor, for E[Y \| D, X] |
| `propensity_model` | random forest | any scikit-learn classifier with `predict_proba`, for P(D=1 \| X) |
| `effect_model` | = `outcome_model` | BGATE only: regressor for E[τ \| W, Z=z] |
| `moderator_model` | = `propensity_model` | BGATE only: classifier for P(Z=1 \| W) |
| `n_folds` | 5 | cross-fitting folds |
| `n_folds_second` | = `n_folds` | BGATE only: folds of the second cross-fitting step |
| `calibration` | `"venn_abers"` | propensity score calibration: `"venn_abers"`, `"platt"`, `"beta"`, `"isotonic"` or `None` |
| `normalize_weights` | `True` | BGATE only: normalise the second-step weights to mean one |
| `propensity_clip` | `1e-10` | propensity scores are clipped to [clip, 1 − clip] |
| `random_state` | `None` | seed for the fold assignment |

Any scikit-learn compatible model works, including pipelines and
`GridSearchCV` objects for hyperparameter tuning. For fully reproducible
results, also set `random_state` in your models.

### Propensity score calibration

Flexible classifiers such as random forests are often over-confident: they
predict treatment probabilities close to, or exactly, 0 or 1. The inverse
probability weights in DML then explode. By default, `balanced_moderation`
calibrates both propensity scores, P(D=1 | X) and P(Z=1 | W), following
Ballinari & Bearth (2024): within each cross-fitting fold, the held-out
predictions are split again, and a calibration map fitted on one part is
applied to the other.

The default is Venn-Abers calibration, which keeps calibrated scores away from
0 and 1 even when the classifier predicts exactly 0 or 1. Platt scaling is a
good alternative. Beta calibration works on log-probabilities and can produce
extreme scores when a classifier outputs exact zeros or ones (as random
forests often do). Ballinari & Bearth recommend choosing the calibration
method by its Brier score and advise against isotonic regression. Use
`calibration=None` to switch calibration off.

If estimated propensity scores are still below 0.01 or above 0.99,
`balanced_moderation` issues a warning.

## How it works

**Step 1.** With K-fold cross-fitting, estimate E[Y | D=1, X, Z],
E[Y | D=0, X, Z] and P(D=1 | X, Z), and compute the doubly robust (AIPW) score
τᵢ of the treatment effect for every observation.

**Step 2.** Within each step-1 fold, cross-fit again: regress τ on the
balancing variables W separately for Z=1 and Z=0, estimate P(Z=1 | W), and
combine them into doubly robust scores for BGATE(1) and BGATE(0).

Estimates are the means of these scores, and standard errors come from their
variance. See Bearth & Lechner (2024) for the theory, including conditions for
√N-consistency and asymptotic normality.

Current limitations: the treatment D and the moderator Z must be binary.

## Replication

With `calibration=None`, the package reproduces the DML estimator used in the
simulation study of Bearth & Lechner (2024) exactly, for the same data and
random seed. The original replication code is at
[NoraBearth/Causal-Machine-Learning-for-Moderation-Effects](https://github.com/NoraBearth/Causal-Machine-Learning-for-Moderation-Effects),
and `make_moderation_data` generates the paper's simulation designs.

## Citation

If you use this package, please cite:

```bibtex
@article{bearth2024moderation,
  title   = {Causal Machine Learning for Moderation Effects},
  author  = {Bearth, Nora and Lechner, Michael},
  journal = {arXiv preprint arXiv:2401.08290},
  year    = {2024}
}
```

and, if you use propensity score calibration:

```bibtex
@article{ballinari2024calibration,
  title   = {Improving the Finite Sample Estimation of Average Treatment Effects
             using Double/Debiased Machine Learning with Propensity Score Calibration},
  author  = {Ballinari, Daniele and Bearth, Nora},
  journal = {arXiv preprint arXiv:2409.04874},
  year    = {2024}
}
```

## License

MIT, see [LICENSE](LICENSE).

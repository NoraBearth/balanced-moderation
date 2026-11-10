"""Pre-release Monte Carlo: bias and CI coverage with the package defaults.

Default settings (5 folds, Venn-Abers calibration, normalised weights), random
forests as nuisance learners, design "z_affects_x", n = 2000.
"""
import sys, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier as C, RandomForestRegressor as R
from balanced_moderation import BGATE, make_moderation_data, true_moderation_effects

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
X = [f"x{j}" for j in range(10)]
truth = true_moderation_effects(design="z_affects_x", n_draws=2_000_000)
rows, t0 = [], time.time()
for rep in range(REPS):
    df = make_moderation_data(n=2000, design="z_affects_x", random_state=1000 + rep)
    model = BGATE(R(200, min_samples_leaf=5, random_state=rep, n_jobs=-1),
                  C(200, min_samples_leaf=5, random_state=rep, n_jobs=-1),
                  random_state=rep)
    for key, balance in [("GATE", None), ("BGATE_x5", ["x5"])]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = model.fit(df, "y", "d", "z", X, balance=balance)
        lo, hi = res.conf_int()
        rows.append(dict(estimand=key, est=res.effect, se=res.std_err,
                         covered=lo <= truth[key] <= hi, truth=truth[key],
                         warned=any("propensity" in str(m.message) for m in w)))
    print(f"rep {rep+1}/{REPS} ({time.time()-t0:.0f}s)", flush=True)
r = pd.DataFrame(rows)
r["err"] = r.est - r.truth
out = r.groupby("estimand").agg(truth=("truth", "first"), mean_est=("est", "mean"),
    bias=("err", "mean"), rmse=("err", lambda e: np.sqrt(np.mean(e**2))),
    sd_est=("est", "std"), mean_se=("se", "mean"), coverage=("covered", "mean"),
    share_warned=("warned", "mean"))
print("\n" + out.round(4).to_string())

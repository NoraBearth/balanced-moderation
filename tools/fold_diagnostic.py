"""Diagnose BGATE bias: how it depends on the cross-fitting folds and n."""
import time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier as C, RandomForestRegressor as R
from balanced_moderation import BGATE, make_moderation_data, true_moderation_effects

REPS = 20
X = [f"x{j}" for j in range(10)]
truth = true_moderation_effects(design="z_affects_x", n_draws=2_000_000)["BGATE_all"]
configs = [  # (n, n_folds, n_folds_second)
    (2000, 2, 2), (2000, 3, 3), (2000, 5, 5), (2000, 5, 2), (8000, 5, 5)]
rows, t0 = [], time.time()
for n, k1, k2 in configs:
    for rep in range(REPS):
        df = make_moderation_data(n=n, design="z_affects_x", random_state=1000 + rep)
        model = BGATE(R(200, min_samples_leaf=5, random_state=rep, n_jobs=-1),
                      C(200, min_samples_leaf=5, random_state=rep, n_jobs=-1),
                      n_folds=k1, n_folds_second=k2, random_state=rep)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(df, "y", "d", "z", X, balance=X)
        lo, hi = res.conf_int()
        rows.append(dict(n=n, folds=f"{k1}x{k2}", est=res.effect, se=res.std_err,
                         covered=lo <= truth <= hi))
    print(f"done n={n} folds={k1}x{k2} ({time.time()-t0:.0f}s)", flush=True)
r = pd.DataFrame(rows); r["err"] = r.est - truth
print(f"\ntrue BGATE difference: {truth:.4f}")
print(r.groupby(["n", "folds"], sort=False).agg(bias=("err", "mean"),
      rmse=("err", lambda e: np.sqrt(np.mean(e**2))), sd_est=("est", "std"),
      mean_se=("se", "mean"), coverage=("covered", "mean")).round(4).to_string())

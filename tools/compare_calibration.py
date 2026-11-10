"""Small Monte Carlo: BGATE with random forests, with and without calibration."""
import sys, warnings, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor as R, RandomForestClassifier as C
from balanced_moderation import BGATE, make_moderation_data, true_moderation_effects

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
X = [f"x{j}" for j in range(10)]
truth = true_moderation_effects(design="z_affects_x", n_draws=1_000_000)["BGATE_all"]
rows = []
t0 = time.time()
for rep in range(REPS):
    df = make_moderation_data(n=2000, design="z_affects_x", random_state=100 + rep)
    for cal in [None, "platt", "beta", "venn_abers"]:
        est = BGATE(R(200, min_samples_leaf=5, random_state=0, n_jobs=-1),
                    C(200, min_samples_leaf=5, random_state=0, n_jobs=-1),
                    n_folds=3, calibration=cal, random_state=rep)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = est.fit(df, "y", "d", "z", X, balance=X)
        rows.append(dict(method=str(cal), rep=rep, est=res.effect, se=res.std_err,
                         warned=any("propensity" in str(x.message) for x in w)))
    print(f"rep {rep+1}/{REPS} done ({time.time()-t0:.0f}s)", flush=True)
r = pd.DataFrame(rows)
r["err"] = r.est - truth
r["covered"] = (r.err.abs() < 1.96 * r.se)
print(f"\ntrue BGATE difference (balanced over all X): {truth:.4f}")
print(r.groupby("method").agg(bias=("err", "mean"), rmse=("err", lambda e: np.sqrt(np.mean(e**2))),
      mean_se=("se", "mean"), coverage=("covered", "mean"), share_warned=("warned", "mean"))
      .round(4).to_string())

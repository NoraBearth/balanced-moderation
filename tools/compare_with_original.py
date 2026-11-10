"""Check that balanced_moderation reproduces the original replication code exactly.

Usage: python tools/compare_with_original.py /path/to/Causal-Machine-Learning-for-Moderation-Effects
"""
import sys, types
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

# The original estimators.py imports torch for the RieszNet estimator; stub it out.
torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
tud = types.ModuleType("torch.utils.data"); tud.DataLoader = tud.TensorDataset = object
sys.modules.update({"torch": torch, "torch.utils": types.ModuleType("torch.utils"), "torch.utils.data": tud})
sys.modules["src.functions_riesznet"] = types.ModuleType("src.functions_riesznet")
sys.path.insert(0, sys.argv[1])
from src import estimators as orig, data as orig_data  # noqa: E402

from balanced_moderation import BGATE, make_moderation_data  # noqa: E402

SEED, N, P = 7, 1500, 10
models = {
    "linear": (LinearRegression(), LogisticRegression(max_iter=10000, tol=1e-12)),
    "forest": (RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=0),
               RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0)),
}
ok = True
for mode, design in [(1, "linear"), (2, "nonlinear"), (3, "z_affects_x")]:
    np.random.seed(SEED)
    y, d, z, x = orig_data.simulate_data(mode=mode, n=N, p=P, true_effect=False)
    df = make_moderation_data(n=N, p=P, design=design, random_state=SEED)
    same_data = np.allclose(df[[f"x{j}" for j in range(P)]].to_numpy(), x) and np.allclose(df.y, y)
    print(f"[{design}] simulated data identical: {same_data}")
    ok &= same_data
    xcols = [f"x{j}" for j in range(P)]
    for mname, (reg, clf) in models.items():
        for balance in [None, [5], [0, 1, 5]]:
            np.random.seed(SEED)
            r_orig = orig.estimate_effect(y, d, z, x, reg, reg, clf, reg, reg, clf,
                                          balance, 3, 3)
            # original BGATE rows: [unnormalised, normalised] second-step weights
            new = {}
            for norm in ([True] if balance is None else [False, True]):
                res = BGATE(reg, clf, n_folds=3, normalize_weights=norm, calibration=None,
                            random_state=SEED).fit(
                    df, "y", "d", "z", xcols,
                    None if balance is None else [f"x{j}" for j in balance])
                new[norm] = (res.estimates[2], res.variances[2])
            pairs = ([(r_orig[0], new[True])] if balance is None
                     else [(r_orig[0], new[False]), (r_orig[1], new[True])])
            for (eo, vo), (en, vn) in pairs:
                match = np.isclose(eo, en, rtol=1e-9, atol=1e-12) and np.isclose(vo, vn, rtol=1e-9, atol=1e-14)
                ok &= match
                print(f"   {mname:6s} balance={str(balance):10s} original={eo: .6f} ({vo:.2e})  "
                      f"package={en: .6f} ({vn:.2e})  {'MATCH' if match else 'DIFFERENT'}")
print("ALL MATCH" if ok else "SOME RESULTS DIFFER")
sys.exit(0 if ok else 1)

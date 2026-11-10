"""Quickstart for balanced_moderation.

Simulates data where a binary moderator Z (say, gender) also shifts the
distribution of a covariate x5 (say, education). Comparing the plain group
effects (GATE) then mixes the moderation effect of Z with the effect of the
different x5 distributions. The balanced GATE (BGATE) holds the distribution
of the balancing variables fixed across both groups.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from balanced_moderation import BGATE, make_moderation_data, true_moderation_effects

df = make_moderation_data(n=4000, design="z_affects_x", random_state=42)
controls = [f"x{j}" for j in range(10)]

model = BGATE(
    outcome_model=RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0),
    propensity_model=RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0),
    calibration="venn_abers",  # calibrate propensity scores (Ballinari & Bearth, 2024)
    random_state=42,
)

# 1) GATE difference: no balancing
gate = model.fit(df, y="y", d="d", z="z", x=controls)
print(gate.summary(), "\n")

# 2) BGATE difference: balance x5, the covariate that z shifts
bgate = model.fit(df, y="y", d="d", z="z", x=controls, balance=["x5"])
print(bgate.summary(), "\n")

truth = true_moderation_effects(design="z_affects_x")
print(f"True GATE difference:  {truth['GATE']:.3f}")
print(f"True BGATE difference (balanced on x5): {truth['BGATE_x5']:.3f}")

# Results as a DataFrame, e.g. for export
print(bgate.to_frame())

"""Pick the final configuration by cross-validation, then evaluate once on the test set.

1. A small, fixed grid (feature set x LightGBM LambdaRank settings) is scored with
   the same 5-fold GroupKFold as before; the best mean NDCG@10 wins.
2. Every model from the comparison, plus the selected configuration, is re-trained
   on the whole training set and scored on the test applications.
The test labels are used only in step 2, never for any choice.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ltr import models as M  # noqa: E402
from ltr.cv import cross_validate, folds  # noqa: E402
from ltr.data import load  # noqa: E402
from ltr.encoding import TE_FEATURES, encode_train_and_apply  # noqa: E402
from ltr.features import FEATURE_GROUPS, all_feature_names, build_pairs  # noqa: E402
from ltr.metrics import per_job, summarize  # noqa: E402
from ltr.plotting import title, BLUE, plt, save  # noqa: E402

OUT = ROOT / "results"
d = load()
train = build_pairs(d.train, d)
test = build_pairs(d.test, d).merge(d.test_labels, on="application_id", validate="one_to_one")
fl = folds(train)

ALL = all_feature_names() + TE_FEATURES
drop = set(FEATURE_GROUPS["time_profile"]) | set(FEATURE_GROUPS["within_job_rank"])
LEAN = [f for f in ALL if f not in drop]
feature_sets = {"all": ALL, "lean (no time/profile, no within-job ranks)": LEAN}
grid = [
    {},
    {"num_leaves": 15},
    {"num_leaves": 63, "min_child_samples": 100},
    {"n_estimators": 1200, "learning_rate": 0.015},
    {"min_child_samples": 200, "reg_lambda": 5.0},
]

# ---- 1. selection by CV -------------------------------------------------------
rows = []
for fs_name, feats in feature_sets.items():
    for params in grid:
        r = cross_validate(lambda p=params: M.LGBMLambdaRank(**p), train, feats, fold_list=fl)
        rows.append({"features": fs_name, "params": json.dumps(params), "n_features": len(feats),
                     "ndcg10": r["ndcg10_mean"], "ndcg10_std": r["ndcg10_std"], "map5": r["map5_mean"]})
        print(f"{fs_name:45s} {json.dumps(params):50s} {r['ndcg10_mean']:.4f}", flush=True)
sel = pd.DataFrame(rows).sort_values("ndcg10", ascending=False)
sel.to_csv(OUT / "selection_grid.csv", index=False)
best = sel.iloc[0]
best_feats = feature_sets[best["features"]]
best_params = json.loads(best["params"])
print("selected:", best["features"], best_params, round(best["ndcg10"], 4))

# ---- 2. test evaluation -------------------------------------------------------
tr, te = encode_train_and_apply(train, test)


class Selected(M.LGBMLambdaRank):
    name = "Selected: LightGBM LambdaRank"


final_models = [
    (M.RandomScores(), ALL), (M.Heuristic(), ALL), (M.LogRegRaw(), ALL), (M.RidgeEngineered(), ALL),
    (M.LGBMPointwise(), ALL), (M.LGBMLambdaRank(), ALL), (M.XGBNDCG(), ALL),
    (M.RankAverage([M.LGBMLambdaRank(), M.XGBNDCG()]), ALL),
    (Selected(**best_params), best_feats),
]
rng = np.random.default_rng(0)
test_rows, selected_model = [], None
for model, feats in final_models:
    model.fit(tr, feats)
    te["score"] = model.predict(te)
    s = summarize(te, "score")
    pj = per_job(te, "score")["ndcg10"].dropna().to_numpy()
    boot = [rng.choice(pj, pj.size).mean() for _ in range(1000)]
    s.update(model=model.name, ndcg10_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
    test_rows.append(s)
    print(f"{model.name:55s} test NDCG@10 {s['ndcg10']:.4f}  MAP@5 {s['map5']:.4f}", flush=True)
    if isinstance(model, Selected):
        selected_model = model
        te["selected_score"] = te["score"]
pd.DataFrame(test_rows).to_csv(OUT / "test_results.csv", index=False)

# breakdowns for the selected model
seen = set(train["job_id"])
pj = per_job(te, "selected_score")
pj["job_seen_in_train"] = pj["job_id"].isin(seen)
pj["size_bin"] = pd.cut(pj["n"], [0, 10, 20, 40, 80, 200]).astype(str)
breakdown = {
    "by_job_seen_in_train": pj.groupby("job_seen_in_train")[["ndcg10", "ap5"]].mean().round(4)
                              .assign(jobs=pj.groupby("job_seen_in_train").size()).to_dict("index"),
    "by_applicants_per_job": pj.groupby("size_bin")[["ndcg10", "ap5"]].mean().round(4)
                               .assign(jobs=pj.groupby("size_bin").size()).to_dict("index"),
    "selected": {"features": best["features"], "params": best_params, "cv_ndcg10": float(best["ndcg10"])},
}
json.dump(breakdown, open(OUT / "test_breakdown.json", "w"), indent=2, default=str)
print(json.dumps(breakdown, indent=2, default=str))

# predictions file: application_id, score (higher = better fit within the job)
te[["application_id", "selected_score"]].rename(columns={"selected_score": "score"}) \
    .sort_values("application_id").to_csv(OUT / "test_predictions.csv", index=False)

# feature importance (gain) of the selected model
imp = pd.Series(selected_model.m.booster_.feature_importance("gain"), index=best_feats)
imp = (imp / imp.sum()).sort_values().tail(15)
imp.round(4).to_csv(OUT / "feature_importance_gain.csv", header=["share_of_gain"])
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.barh(imp.index, imp.values * 100, color=BLUE, height=0.6)
ax.set_xlabel("share of total split gain (%)")
title(fig, "Top 15 features of the selected model")
ax.grid(axis="y", visible=False)
save(fig, OUT / "figures" / "feature_importance.png")

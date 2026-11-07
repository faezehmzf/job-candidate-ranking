"""Compare models with 5-fold GroupKFold (groups = job_id) on the training set.

Also runs one time-based check for the main model: train on applications before
2024-04-01, validate on 2024-04-01 .. 2024-06-30.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ltr import models as M  # noqa: E402
from ltr.cv import cross_validate, folds  # noqa: E402
from ltr.data import load  # noqa: E402
from ltr.encoding import TE_FEATURES, encode_train_and_apply  # noqa: E402
from ltr.features import all_feature_names, build_pairs  # noqa: E402
from ltr.metrics import summarize  # noqa: E402
from ltr.plotting import title, BLUE, plt, save  # noqa: E402

OUT = ROOT / "results"
d = load()
train = build_pairs(d.train, d)
F = all_feature_names() + TE_FEATURES
fl = folds(train)

candidates = [
    M.RandomScores, M.Heuristic, M.LogRegRaw, M.RidgeEngineered,
    M.LGBMPointwise, M.LGBMLambdaRank, M.XGBNDCG,
    lambda: M.RankAverage([M.LGBMLambdaRank(), M.XGBNDCG()]),
]
rows = []
for make in candidates:
    t = time.time()
    r = cross_validate(make, train, F, fold_list=fl)
    name = make().name
    rows.append({"model": name, "ndcg10": r["ndcg10_mean"], "ndcg10_std": r["ndcg10_std"],
                 "map5": r["map5_mean"], "map5_std": r["map5_std"], "folds": r["ndcg10_folds"],
                 "seconds": round(time.time() - t)})
    print(f"{name:45s} NDCG@10 {r['ndcg10_mean']:.4f} ± {r['ndcg10_std']:.4f}  MAP@5 {r['map5_mean']:.4f}"
          f"  ({rows[-1]['seconds']}s)", flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "cv_model_comparison.csv", index=False)

# time-based check for the main model
cut, end = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30")
early = train[train["application_date"] < cut]
late = train[(train["application_date"] >= cut) & (train["application_date"] <= end)]
tr, va = encode_train_and_apply(early, late)
m = M.LGBMLambdaRank().fit(tr, F)
time_check = summarize(va.assign(score=m.predict(va)), "score")
print("time-based check:", time_check)
json.dump(time_check, open(OUT / "time_split_check.json", "w"), indent=2)

# figure
r = res.sort_values("ndcg10")
fig, ax = plt.subplots(figsize=(8.4, 3.8))
ax.barh(r["model"], r["ndcg10"], color=BLUE, height=0.6, xerr=r["ndcg10_std"], ecolor="#52514e")
for y, v in enumerate(r["ndcg10"]):
    ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9, color="#52514e")
ax.set_xlim(0, 1)
ax.set_xlabel("NDCG@10 (5-fold GroupKFold, mean ± sd)")
title(fig, "Model comparison (cross-validation)")
ax.grid(axis="y", visible=False)
save(fig, OUT / "figures" / "cv_model_comparison.png")

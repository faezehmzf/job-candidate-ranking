"""Feature-group ablation: drop one group at a time and re-run 5-fold GroupKFold.

Run for LambdaRank, plus the pointwise regressor without the within-job-rank
group, to see whether those rank features are what lets a pointwise model keep up.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ltr import models as M  # noqa: E402
from ltr.cv import cross_validate, folds  # noqa: E402
from ltr.data import load  # noqa: E402
from ltr.encoding import TE_FEATURES  # noqa: E402
from ltr.features import FEATURE_GROUPS, all_feature_names, build_pairs  # noqa: E402
from ltr.plotting import title, BLUE, ORANGE, plt, save  # noqa: E402

OUT = ROOT / "results"
d = load()
train = build_pairs(d.train, d)
fl = folds(train)
groups = {**FEATURE_GROUPS, "target_encodings": TE_FEATURES}
ALL = all_feature_names() + TE_FEATURES

base = cross_validate(M.LGBMLambdaRank, train, ALL, fold_list=fl)
rows = [{"model": "LambdaRank", "dropped": "(none)", "ndcg10": base["ndcg10_mean"], "map5": base["map5_mean"]}]
print("all features", round(base["ndcg10_mean"], 4), flush=True)
for g, cols in groups.items():
    feats = [f for f in ALL if f not in cols]
    r = cross_validate(M.LGBMLambdaRank, train, feats, fold_list=fl)
    rows.append({"model": "LambdaRank", "dropped": g, "ndcg10": r["ndcg10_mean"], "map5": r["map5_mean"]})
    print(f"drop {g:20s} {r['ndcg10_mean']:.4f}  delta {r['ndcg10_mean'] - base['ndcg10_mean']:+.4f}", flush=True)

# groups overlap (e.g. the heuristic and the rank features repeat skill information),
# so also train on each group alone
for g, cols in groups.items():
    r = cross_validate(M.LGBMLambdaRank, train, list(cols), fold_list=fl)
    rows.append({"model": "LambdaRank", "dropped": f"only:{g}", "ndcg10": r["ndcg10_mean"], "map5": r["map5_mean"]})
    print(f"only {g:20s} {r['ndcg10_mean']:.4f}", flush=True)

no_rank = [f for f in ALL if f not in FEATURE_GROUPS["within_job_rank"]]
for make, label in [(M.LGBMPointwise, "Pointwise regressor")]:
    full = cross_validate(make, train, ALL, fold_list=fl)
    r = cross_validate(make, train, no_rank, fold_list=fl)
    rows += [{"model": label, "dropped": "(none)", "ndcg10": full["ndcg10_mean"], "map5": full["map5_mean"]},
             {"model": label, "dropped": "within_job_rank", "ndcg10": r["ndcg10_mean"], "map5": r["map5_mean"]}]
    print(f"{label}: all {full['ndcg10_mean']:.4f}, without within-job ranks {r['ndcg10_mean']:.4f}", flush=True)

res = pd.DataFrame(rows)
res["delta_vs_all"] = res["ndcg10"] - res.groupby("model")["ndcg10"].transform("first")
res.to_csv(OUT / "ablation.csv", index=False)

a = res[(res.model == "LambdaRank") & (res.dropped != "(none)") & ~res.dropped.str.startswith("only:")]
a = a.sort_values("delta_vs_all")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
colors = [ORANGE if v < 0 else BLUE for v in a["delta_vs_all"]]
ax.barh(a["dropped"], a["delta_vs_all"], color=colors, height=0.6)
ax.axvline(0, color="#52514e", lw=1)
for y, v in enumerate(a["delta_vs_all"]):
    ax.text(v + (0.002 if v >= 0 else -0.002), y, f"{v:+.4f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=9, color="#52514e")
lo = min(a["delta_vs_all"].min() * 1.35, -0.01)
ax.set_xlim(lo, max(0.01, a["delta_vs_all"].max() * 1.5))
ax.set_xlabel("change in CV NDCG@10 when the group is removed")
title(fig, "Feature-group ablation (LightGBM LambdaRank)")
ax.grid(axis="y", visible=False)
save(fig, OUT / "figures" / "ablation.png")

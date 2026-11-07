"""Data-quality audit and dataset facts that drive the modelling choices.

Writes results/data_audit.json and two figures.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ltr.data import DATA_DIR, load  # noqa: E402
from ltr.plotting import title, BLUE, ORANGE, plt, save  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

raw_j = pd.read_csv(DATA_DIR / "jobs.csv")
raw_c = pd.read_csv(DATA_DIR / "candidates.csv")
d = load()
tr, te = d.train, d.test

issues = {}
cur = raw_j["salary_currency"]
issues["salary_currency_not_canonical"] = int((~cur.isin(["USD", "EUR", "IRR"])).sum())
issues["salary_currency_counts"] = d.jobs["salary_currency"].value_counts().to_dict()
issues["job_posted_date_dd_mm_yyyy"] = int(raw_j["job_posted_date"].str.contains("/").sum())
issues["min_years_experience_negative"] = int((raw_j["min_years_experience"] < 0).sum())


def n_variants(raw: pd.Series, clean: pd.Series) -> int:
    """Rows whose raw spelling differs from the most common spelling of the same cleaned value."""
    modal = raw.groupby(clean).agg(lambda x: x.mode().iloc[0])
    return int((raw != clean.map(modal)).sum())


issues["job_title_spelling_variants"] = n_variants(raw_j["job_title"], d.jobs["job_title_clean"])
issues["candidate_title_spelling_variants"] = n_variants(raw_c["current_title"], d.candidates["current_title_clean"])
issues["candidate_location_spelling_variants"] = n_variants(raw_c["candidate_location"], d.candidates["candidate_location_clean"])
edu_canon = {"Bachelor", "Master", "PhD", "High School"}
issues["education_spelling_variants"] = int((~raw_c["education_level"].isin(edu_canon)).sum())
issues["english_plain_language_scale"] = int((~raw_c["english_proficiency"].fillna("B1").str.match(r"^[ABC][12]$")).sum())
issues["missing_share_jobs"] = raw_j.isna().mean().loc[lambda s: s > 0].round(3).to_dict()
issues["missing_share_candidates"] = raw_c.isna().mean().loc[lambda s: s > 0].round(3).to_dict()

# Salary scale: candidate expectations look the same everywhere -> already USD.
irr_cities = ["Tehran", "Mashhad", "Isfahan", "Shiraz", "Tabriz", "Karaj", "Qom"]
is_irr = d.candidates["candidate_location_clean"].str.title().isin(irr_cities)
issues["expected_salary_median_irr_cities"] = float(d.candidates.loc[is_irr, "expected_salary"].median())
issues["expected_salary_median_other_cities"] = float(d.candidates.loc[~is_irr, "expected_salary"].median())
issues["irr_per_usd_estimated_from_data"] = round(d.irr_per_usd, 1)

# Label structure
lab = tr.groupby("job_id")["relevance_label"]
size = lab.size()
share = tr.assign(n=tr.groupby("job_id")["job_id"].transform("size"))
facts = {
    "train_rows": len(tr), "test_rows": len(te),
    "train_jobs": int(tr["job_id"].nunique()), "test_jobs": int(te["job_id"].nunique()),
    "train_dates": [str(tr["application_date"].min().date()), str(tr["application_date"].max().date())],
    "test_dates": [str(te["application_date"].min().date()), str(te["application_date"].max().date())],
    "test_jobs_also_in_train": round(float(te["job_id"].drop_duplicates().isin(tr["job_id"]).mean()), 3),
    "candidates_in_both": round(float(te["candidate_id"].drop_duplicates().isin(tr["candidate_id"]).mean()), 3),
    "applicants_per_job_median": float(size.median()),
    "label_share_train": tr["relevance_label"].value_counts(normalize=True).sort_index().round(3).to_dict(),
    # if labels are relative ranks inside a job, every job has (almost) the same label mix
    "std_across_jobs_of_mean_label_jobs_ge_40": round(float(lab.mean()[size >= 40].std()), 4),
    "share_jobs_with_label4_by_size": {
        "<20": round(float((lab.max()[size < 20] == 4).mean()), 3),
        ">=20": round(float((lab.max()[size >= 20] == 4).mean()), 3),
    },
}
json.dump({"issues": issues, "facts": facts}, open(OUT / "data_audit.json", "w"), indent=2, default=str)
print(json.dumps({"issues": issues, "facts": facts}, indent=2, default=str))

# Figure 1: share of applicants with each label as a function of job size
bins = [0, 10, 20, 30, 50, 80, 200]
share["size_bin"] = pd.cut(share["n"], bins)
mix = share.groupby("size_bin", observed=True)["relevance_label"].apply(lambda s: (s == 4).mean())
top = lab.max().groupby(pd.cut(size, bins), observed=True).apply(lambda s: (s == 4).mean())
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(len(top))
ax.plot(x, top.values * 100, color=BLUE, lw=2, marker="o", ms=7, label="jobs with at least one label-4 applicant")
ax.plot(x, mix.values * 100, color=ORANGE, lw=2, marker="o", ms=7, label="applicants with label 4")
ax.set_xticks(x, [f"{int(i.left)+1}-{int(i.right)}" for i in top.index])
ax.set_xlabel("applicants per job")
ax.set_ylabel("%")
ax.set_ylim(0, 105)
title(fig, "Label 4 is rare in jobs with fewer than 20 applicants")
ax.legend(frameon=False, loc="center right")
save(fig, FIG / "label4_vs_job_size.png")

# Figure 2: applicants per job
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.hist(size, bins=40, color=BLUE, edgecolor="#fcfcfb", linewidth=1)
ax.set_xlabel("applicants per job (train)")
ax.set_ylabel("jobs")
title(fig, "Most jobs have 15-50 applicants")
save(fig, FIG / "applicants_per_job.png")

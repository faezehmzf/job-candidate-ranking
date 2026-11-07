"""Job-candidate pair features.

`build_pairs` creates every feature that does not need labels. Features that are
learned from labels (out-of-fold target encodings) live in `ltr.encoding`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset, title_tokens

SENIORITY = {"junior": 0, "mid": 1, "senior": 2, "lead": 3}
_LEAD_WORDS = ("lead", "principal", "head", "cto", "vp", "director", "manager")

# Role families, written from the title vocabulary (38 distinct titles).
# A title can belong to several families (e.g. full stack = frontend + backend).
_FAMILIES = {
    "frontend": ("frontend", "web", "full stack"),
    "backend": ("backend", "full stack", "software", "systems"),
    "data": ("data", "database", "ml"),
    "ml": ("ml", "data scientist"),
    "devops": ("devops", "systems", "database administrator"),
    "architecture": ("architect", "principal", "tech lead", "cto", "engineering"),
    "management": ("manager", "director", "vp", "cto", "head"),
}

# Certification groups (by vendor / domain), used as candidate attributes; the
# model can combine them with the job's industry.
CERT_GROUPS = {
    "cloud": ("aws-", "azure-", "gcp-"),
    "containers_iac": ("docker-", "kubernetes-", "terraform-"),
    "security_compliance": ("ceh", "cissp", "pci-dss", "hipaa"),
    "delivery": ("pmp", "scrum-", "agile-"),
    "finance": ("cfa",),
}

FEATURE_GROUPS = {
    "skills": ["skill_overlap", "skill_coverage", "skill_jaccard", "n_required_skills", "n_candidate_skills"],
    "experience": ["years_experience", "min_years", "max_years", "exp_below_min", "exp_above_max",
                   "exp_in_range", "exp_missing"],
    "title": ["job_seniority", "cand_seniority", "seniority_gap", "family_match", "title_token_jaccard",
              "same_title"],
    "location": ["same_city", "job_remote_code", "willing_code", "cand_is_remote", "job_is_remote"],
    "salary": ["salary_ratio", "log_salary_ratio", "salary_in_band", "expected_salary", "job_salary_mid_usd"],
    "education_language": ["education_code", "english_code"],
    "certifications": ["n_certs"] + [f"certs_{g}" for g in CERT_GROUPS],
    "job_context": ["company_size_code", "industry_code", "n_applicants"],
    "time_profile": ["profile_completeness", "account_age_days", "days_since_posting",
                     "application_month", "application_dow"],
    "within_job_rank": ["rank_skill_coverage", "rank_exp_fit", "rank_salary_fit", "rank_title",
                        "rank_heuristic"],
}

# raw columns used by the "naive" baseline (no feature engineering)
RAW_COLUMNS = ["years_experience", "min_years", "max_years", "expected_salary", "job_salary_mid_usd",
               "education_code", "english_code", "profile_completeness", "company_size_code",
               "n_required_skills", "n_candidate_skills", "n_certs"]


def seniority(title) -> int:
    t = "" if title is None or title is pd.NA else str(title)
    if "junior" in t or "intern" in t:
        return SENIORITY["junior"]
    if "senior" in t:
        return SENIORITY["senior"]
    if any(w in t for w in _LEAD_WORDS):
        return SENIORITY["lead"]
    return SENIORITY["mid"]


def families(title) -> frozenset:
    t = "" if title is None or title is pd.NA else str(title)
    return frozenset(f for f, keys in _FAMILIES.items() if any(k in t for k in keys))


def _set_stats(a: pd.Series, b: pd.Series):
    inter = np.fromiter((len(x & y) for x, y in zip(a, b)), float, len(a))
    la = a.map(len).to_numpy(float)
    lb = b.map(len).to_numpy(float)
    union = la + lb - inter
    return inter, la, lb, union


def heuristic_score(df: pd.DataFrame) -> pd.Series:
    """A simple hand-written fit score (no learning), used as a baseline and a feature.

    Five parts, each scaled to [0, 1] and weighted equally: skill coverage,
    experience fit, title fit, location fit and salary fit.
    """
    years_off = df["exp_below_min"].fillna(0) + df["exp_above_max"].fillna(0)
    exp_fit = 1 / (1 + years_off)
    title_fit = df["family_match"] * (1 - df["seniority_gap"] / 3)
    loc_fit = np.maximum(df["same_city"], 0.5 * ((df["job_remote_code"].fillna(0) > 0) | (df["willing_code"].fillna(0) > 0)))
    sal_fit = 1 / (1 + df["log_salary_ratio"].abs())
    return df["skill_coverage"].fillna(0) + exp_fit + title_fit + loc_fit + sal_fit


def build_pairs(apps: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """Merge applications with job and candidate tables and compute pair features."""
    j = ds.jobs
    c = ds.candidates
    df = apps.merge(j, on="job_id", how="left", validate="many_to_one")
    df = df.merge(c, on="candidate_id", how="left", validate="many_to_one")
    assert len(df) == len(apps)
    out = df[["application_id", "job_id", "candidate_id", "application_date"]].copy()
    if "relevance_label" in df:
        out["relevance_label"] = df["relevance_label"]

    # skills
    inter, lreq, lcand, union = _set_stats(df["required_skills_set"], df["skills_set"])
    out["skill_overlap"] = inter
    out["skill_coverage"] = np.where(lreq > 0, inter / np.maximum(lreq, 1), np.nan)
    out["skill_jaccard"] = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    out["n_required_skills"] = lreq
    out["n_candidate_skills"] = lcand

    # experience
    exp = df["years_experience"]
    out["years_experience"] = exp
    out["min_years"] = df["min_years_experience"]
    out["max_years"] = df["max_years_experience"]
    out["exp_below_min"] = (df["min_years_experience"] - exp).clip(lower=0)
    out["exp_above_max"] = (exp - df["max_years_experience"]).clip(lower=0)
    out["exp_in_range"] = ((exp >= df["min_years_experience"].fillna(0)) &
                           (exp <= df["max_years_experience"].fillna(np.inf))).astype(float)
    out.loc[exp.isna(), "exp_in_range"] = np.nan
    out["exp_missing"] = exp.isna().astype(float)

    # title
    js = df["job_title_clean"].map(seniority)
    cs = df["current_title_clean"].map(seniority)
    out["job_seniority"] = js
    out["cand_seniority"] = cs
    out["seniority_gap"] = (js - cs).abs()
    jf = df["job_title_clean"].map(families)
    cf = df["current_title_clean"].map(families)
    out["family_match"] = np.fromiter((len(a & b) > 0 for a, b in zip(jf, cf)), float, len(df))
    jt = df["job_title_clean"].map(title_tokens)
    ct = df["current_title_clean"].map(title_tokens)
    ti, tl1, tl2, tu = _set_stats(jt, ct)
    out["title_token_jaccard"] = np.where(tu > 0, ti / np.maximum(tu, 1), 0.0)
    out["same_title"] = (df["job_title_clean"] == df["current_title_clean"]).astype(float)

    # location
    out["same_city"] = (df["job_location_clean"] == df["candidate_location_clean"]).astype(float)
    out["job_remote_code"] = df["remote_code"]
    out["willing_code"] = df["willing_code"]
    out["cand_is_remote"] = (df["candidate_location_clean"] == "remote").astype(float)
    out["job_is_remote"] = (df["job_location_clean"] == "remote").astype(float)

    # salary (job salary converted to USD; candidate expectation is already USD)
    ratio = df["expected_salary"] / df["salary_mid_usd"]
    out["salary_ratio"] = ratio
    out["log_salary_ratio"] = np.log(ratio)
    out["salary_in_band"] = ((df["expected_salary"] >= df["salary_min_usd"]) &
                             (df["expected_salary"] <= df["salary_max_usd"])).astype(float)
    out["expected_salary"] = df["expected_salary"]
    out["job_salary_mid_usd"] = df["salary_mid_usd"]

    # education / language
    out["education_code"] = df["education_code"]
    out["english_code"] = df["english_code"]

    # certifications
    certs = df["certs_set"]
    out["n_certs"] = certs.map(len).astype(float)
    for g, prefixes in CERT_GROUPS.items():
        out[f"certs_{g}"] = certs.map(lambda s, p=prefixes: sum(any(x.startswith(q) for q in p) for x in s)).astype(float)

    # job context
    out["company_size_code"] = df["company_size_code"]
    out["industry_code"] = df["industry_clean"].astype("category").cat.codes.astype(float)
    out["n_applicants"] = out.groupby("job_id")["job_id"].transform("size").astype(float)

    # time / profile
    out["profile_completeness"] = df["profile_completeness"]
    out["account_age_days"] = (df["application_date"] - df["account_created_date"]).dt.days
    out["days_since_posting"] = (df["application_date"] - df["job_posted_date"]).dt.days
    out["application_month"] = df["application_date"].dt.month
    out["application_dow"] = df["application_date"].dt.dayofweek

    # within-job ranks: labels are relative to the other applicants of the same job,
    # so each applicant's standing inside the job is informative.
    out["heuristic"] = heuristic_score(out)
    exp_fit = -(out["exp_below_min"].fillna(0) + out["exp_above_max"].fillna(0))
    sal_fit = -out["log_salary_ratio"].abs()
    title_fit = out["family_match"] * 3 - out["seniority_gap"]
    g = out["job_id"]
    out["rank_skill_coverage"] = out["skill_coverage"].groupby(g).rank(pct=True)
    out["rank_exp_fit"] = exp_fit.groupby(g).rank(pct=True)
    out["rank_salary_fit"] = sal_fit.groupby(g).rank(pct=True)
    out["rank_title"] = title_fit.groupby(g).rank(pct=True)
    out["rank_heuristic"] = out["heuristic"].groupby(g).rank(pct=True)

    # keys used by the target encoders
    out["key_title_pair"] = df["job_title_clean"].astype(str) + " | " + df["current_title_clean"].astype(str)
    out["key_industry"] = df["industry_clean"].astype(str)
    out["certs_set"] = certs
    return out


def all_feature_names() -> list[str]:
    return [f for cols in FEATURE_GROUPS.values() for f in cols]

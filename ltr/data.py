"""Loading and cleaning of the raw CSV files.

Every cleaning rule here was found by looking at the raw data
(see scripts/01_data_audit.py, which counts how often each issue occurs).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EDUCATION = {  # canonical ordinal: 0 high school, 1 bachelor, 2 master, 3 doctorate
    "high school": 0, "secondary": 0, "hs": 0,
    "bachelor": 1, "b.a.": 1, "b.s.": 1, "bsc": 1,
    "master": 2, "m.a.": 2, "m.s.": 2, "msc": 2,
    "phd": 3, "ph.d.": 3, "doctorate": 3,
}

ENGLISH = {  # CEFR ordinal: A1=0 ... C2=5; plain-English labels mapped to the closest level
    "a1": 0, "beginner": 0,
    "a2": 1, "elementary": 1,
    "b1": 2, "intermediate": 2,
    "b2": 3, "upper-intermediate": 3,
    "c1": 4, "advanced": 4,
    "c2": 5, "fluent": 5, "native": 5, "bilingual": 5,
}

COMPANY_SIZE = {"1-10": 0, "11-50": 1, "51-200": 2, "201-1000": 3, "1000+": 4}
REMOTE = {"no": 0, "hybrid": 1, "yes": 2}


@dataclass
class Dataset:
    jobs: pd.DataFrame
    candidates: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame          # test applications, without labels
    test_labels: pd.DataFrame | None  # application_id -> relevance_label (evaluation only)
    irr_per_usd: float


def norm_text(s: pd.Series) -> pd.Series:
    """Lower-case, trim and collapse whitespace; missing stays missing."""
    return s.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def norm_title(s: pd.Series) -> pd.Series:
    """Normalise job titles: case, whitespace and the Sr./Jr. abbreviations."""
    t = norm_text(s)
    t = t.str.replace(r"\bsr\b\.?", "senior", regex=True)
    t = t.str.replace(r"\bjr\b\.?", "junior", regex=True)
    return t


def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """Dates come either as ISO (YYYY-MM-DD) or as DD/MM/YYYY."""
    s = s.astype("string").str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return iso.fillna(dmy)


def split_pipe(s: pd.Series) -> pd.Series:
    """'A|B|C' -> frozenset({'a','b','c'}) with case/whitespace normalised."""
    return s.fillna("").map(
        lambda x: frozenset(p.strip().lower() for p in str(x).split("|") if p.strip())
    )


def estimate_irr_per_usd(jobs: pd.DataFrame) -> float:
    """Estimate the IRR->USD rate from the data itself.

    Salaries of comparable jobs (same title) should be similar in real terms, so
    the ratio of median IRR salary to median USD salary for the same title gives
    the exchange rate the data was produced with.
    """
    mid = (jobs["salary_min"] + jobs["salary_max"]) / 2
    df = pd.DataFrame({"title": jobs["job_title_clean"], "cur": jobs["salary_currency"], "mid": mid})
    med = df.groupby(["title", "cur"])["mid"].median().unstack()
    ratio = (med["IRR"] / med["USD"]).dropna()
    return float(ratio.median())


def clean_jobs(jobs: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    j = jobs.copy()
    j["job_title_clean"] = norm_title(j["job_title"])
    j["salary_currency"] = norm_text(j["salary_currency"]).str.upper()
    j["job_location_clean"] = norm_text(j["job_location"])
    j["remote_code"] = norm_text(j["remote_allowed"]).map(REMOTE).astype("float")
    j["company_size_code"] = j["company_size"].map(COMPANY_SIZE).astype("float")
    j["industry_clean"] = norm_text(j["industry"])
    j["job_posted_date"] = parse_mixed_dates(j["job_posted_date"])
    # a negative minimum experience is not meaningful
    j.loc[j["min_years_experience"] < 0, "min_years_experience"] = np.nan
    j["required_skills_set"] = split_pipe(j["required_skills"])

    irr = estimate_irr_per_usd(j)
    eur = 1.08  # EUR jobs are rare enough that a fixed, documented rate is fine
    rate = j["salary_currency"].map({"USD": 1.0, "EUR": eur, "IRR": 1.0 / irr}).fillna(1.0)
    j["salary_min_usd"] = j["salary_min"] * rate
    j["salary_max_usd"] = j["salary_max"] * rate
    j["salary_mid_usd"] = (j["salary_min_usd"] + j["salary_max_usd"]) / 2
    return j, irr


def clean_candidates(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["current_title_clean"] = norm_title(c["current_title"])
    c["education_code"] = norm_text(c["education_level"]).map(EDUCATION).astype("float")
    c["english_code"] = norm_text(c["english_proficiency"]).map(ENGLISH).astype("float")
    c["candidate_location_clean"] = norm_text(c["candidate_location"])
    c["willing_code"] = norm_text(c["willing_to_relocate"]).map({"no": 0, "yes": 1}).astype("float")
    c["account_created_date"] = parse_mixed_dates(c["account_created_date"])
    c["skills_set"] = split_pipe(c["skills"])
    c["certs_set"] = split_pipe(c["certifications"])
    # expected_salary has the same distribution in every city (checked in the audit),
    # so it is treated as already being in USD.
    return c


def load(data_dir: Path | str = DATA_DIR, with_test_labels: bool = True) -> Dataset:
    d = Path(data_dir)
    jobs, irr = clean_jobs(pd.read_csv(d / "jobs.csv"))
    cands = clean_candidates(pd.read_csv(d / "candidates.csv"))
    train = pd.read_csv(d / "applications_train.csv", parse_dates=["application_date"])
    test = pd.read_csv(d / "applications_test.csv", parse_dates=["application_date"])
    labels = None
    lab_path = d / "applications_test_labels.csv"
    if with_test_labels and lab_path.exists():
        labels = pd.read_csv(lab_path)[["application_id", "relevance_label"]]
    return Dataset(jobs, cands, train, test, labels, irr)


_word = re.compile(r"[a-z]+")


def title_tokens(title: str | None) -> frozenset:
    if title is None or title is pd.NA:
        return frozenset()
    return frozenset(_word.findall(str(title)))

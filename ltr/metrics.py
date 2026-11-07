"""Ranking metrics computed per job and averaged over jobs.

* NDCG@k with exponential gain (2^label - 1) and log2 discount.
  Jobs whose ideal DCG is 0 (every applicant has label 0) are skipped,
  because NDCG is undefined for them.
* AP@5 with binary relevance label >= 3, normalised by min(#relevant, 5).
  Jobs without any relevant applicant are skipped.
Ties in the score are broken by input order (stable sort), so results do not
depend on a random tie-break.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RELEVANT_LABEL = 3


def dcg(labels: np.ndarray, k: int) -> float:
    labels = np.asarray(labels, dtype=float)[:k]
    return float(np.sum((2.0 ** labels - 1.0) / np.log2(np.arange(2, labels.size + 2))))


def ndcg_at_k(labels, scores, k: int = 10) -> float | None:
    labels = np.asarray(labels, dtype=float)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ideal = dcg(np.sort(labels)[::-1], k)
    if ideal == 0:
        return None
    return dcg(labels[order], k) / ideal


def ap_at_k(labels, scores, k: int = 5, relevant: int = RELEVANT_LABEL) -> float | None:
    rel = np.asarray(labels) >= relevant
    n_rel = int(rel.sum())
    if n_rel == 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    top = rel[order][:k]
    hits = np.cumsum(top)
    precision = hits / np.arange(1, top.size + 1)
    return float(np.sum(precision * top) / min(n_rel, k))


def per_job(df: pd.DataFrame, score_col: str, label_col: str = "relevance_label",
            group_col: str = "job_id") -> pd.DataFrame:
    rows = []
    for job, g in df.groupby(group_col, sort=False):
        y = g[label_col].to_numpy()
        s = g[score_col].to_numpy()
        rows.append((job, len(g), ndcg_at_k(y, s, 10), ap_at_k(y, s, 5)))
    return pd.DataFrame(rows, columns=[group_col, "n", "ndcg10", "ap5"])


def summarize(df: pd.DataFrame, score_col: str, **kw) -> dict:
    pj = per_job(df, score_col, **kw)
    return {
        "ndcg10": float(pj["ndcg10"].dropna().mean()),
        "map5": float(pj["ap5"].dropna().mean()),
        "jobs": int(len(pj)),
        "jobs_scored_ndcg": int(pj["ndcg10"].notna().sum()),
        "jobs_scored_map": int(pj["ap5"].notna().sum()),
    }

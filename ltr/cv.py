"""Cross-validation with GroupKFold on job_id.

All applicants of a job are always in the same fold, which matches the test
situation (test jobs are mostly jobs the model has never seen). Target
encodings are re-fitted inside each fold so the validation fold's labels are
never used to build its own features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .encoding import encode_train_and_apply
from .metrics import summarize

N_SPLITS = 5


def folds(df: pd.DataFrame, n_splits: int = N_SPLITS):
    return list(GroupKFold(n_splits=n_splits).split(df, groups=df["job_id"]))


def cross_validate(make_model, df: pd.DataFrame, features: list[str],
                   n_splits: int = N_SPLITS, fold_list=None) -> dict:
    """Return mean/std of NDCG@10 and MAP@5 over folds, plus the OOF scores."""
    fold_list = fold_list or folds(df, n_splits)
    stats, oof = [], pd.Series(np.nan, index=df.index)
    for tr_idx, va_idx in fold_list:
        tr, va = encode_train_and_apply(df.iloc[tr_idx], df.iloc[va_idx])
        model = make_model().fit(tr, features)
        va = va.assign(score=model.predict(va))
        oof.iloc[va_idx] = va["score"].to_numpy()
        stats.append(summarize(va, "score"))
    nd = np.array([s["ndcg10"] for s in stats])
    mp = np.array([s["map5"] for s in stats])
    return {"ndcg10_mean": float(nd.mean()), "ndcg10_std": float(nd.std(ddof=1)),
            "map5_mean": float(mp.mean()), "map5_std": float(mp.std(ddof=1)),
            "ndcg10_folds": nd.round(4).tolist(), "oof": oof}

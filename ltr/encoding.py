"""Out-of-fold target encodings.

Labels are relative to the other applicants of the same job, so the target being
encoded is the *job-centred* label (label minus the job's mean label). Two
encodings are learned:

* te_title_pair   - how well a candidate's current title suits the job title
                    (38 x 38 title pairs);
* te_cert_industry - how much each certification helps in the job's industry,
                    summed over the candidate's certifications.

Rows used for training always get out-of-fold values (GroupKFold on job_id), so a
row's own label never contributes to its encoding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TE_FEATURES = ["te_title_pair", "te_cert_industry"]
SMOOTHING = 20.0


def _centred(df: pd.DataFrame) -> pd.Series:
    y = df["relevance_label"].astype(float)
    return y - y.groupby(df["job_id"]).transform("mean")


def _explode_certs(df: pd.DataFrame, target: pd.Series | None = None) -> pd.DataFrame:
    rows = pd.DataFrame({"row": np.arange(len(df)), "certs": df["certs_set"].to_numpy(),
                         "industry": df["key_industry"].to_numpy()})
    if target is not None:
        rows["t"] = target.to_numpy()
    rows = rows.explode("certs").dropna(subset=["certs"])
    rows["key"] = rows["certs"] + " @ " + rows["industry"]
    return rows


class TargetEncoder:
    def fit(self, df: pd.DataFrame) -> "TargetEncoder":
        t = _centred(df)
        self.title_map = self._smoothed(df["key_title_pair"], t)
        ex = _explode_certs(df, t)
        self.cert_map = self._smoothed(ex["key"], ex["t"])
        return self

    @staticmethod
    def _smoothed(keys: pd.Series, t: pd.Series) -> pd.Series:
        # centred labels have mean 0, so shrink towards 0
        g = pd.DataFrame({"k": keys.to_numpy(), "t": t.to_numpy()}).groupby("k")["t"].agg(["sum", "count"])
        return g["sum"] / (g["count"] + SMOOTHING)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["te_title_pair"] = df["key_title_pair"].map(self.title_map).fillna(0.0).to_numpy()
        ex = _explode_certs(df)
        vals = ex["key"].map(self.cert_map).fillna(0.0)
        summed = vals.groupby(ex["row"]).sum()
        out["te_cert_industry"] = summed.reindex(np.arange(len(df)), fill_value=0.0).to_numpy()
        return out


def oof_encode(train: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold encodings for training rows (GroupKFold on job_id)."""
    out = pd.DataFrame(0.0, index=train.index, columns=TE_FEATURES)
    for fit_idx, enc_idx in GroupKFold(n_splits=n_splits).split(train, groups=train["job_id"]):
        enc = TargetEncoder().fit(train.iloc[fit_idx])
        out.iloc[enc_idx] = enc.transform(train.iloc[enc_idx]).to_numpy()
    return out


def encode_train_and_apply(train: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF encodings for `train`; encodings fitted on all of `train` for `other`."""
    tr = train.copy()
    ot = other.copy()
    tr[TE_FEATURES] = oof_encode(train).to_numpy()
    ot[TE_FEATURES] = TargetEncoder().fit(train).transform(other).to_numpy()
    return tr, ot

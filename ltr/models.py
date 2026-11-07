"""Models compared in this project.

Every model exposes fit(train_df, features) and predict(df) -> scores, where a
higher score means a better fit for that job. Rankers need rows grouped by job,
so frames are sorted by job_id before fitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker

from .features import RAW_COLUMNS

SEED = 42

LGBM_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=SEED, n_jobs=-1, verbose=-1, deterministic=True, force_row_wise=True)


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["job_id", "application_id"], kind="stable")


def _groups(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("job_id", sort=False).size().to_numpy()


class RandomScores:
    name = "Random"

    def fit(self, df, features):
        return self

    def predict(self, df):
        return np.random.default_rng(SEED).random(len(df))


class Heuristic:
    """Hand-weighted fit score, no learning."""
    name = "Heuristic score"

    def fit(self, df, features):
        return self

    def predict(self, df):
        return df["heuristic"].to_numpy()


class LogRegRaw:
    """Naive baseline: logistic regression on raw columns, label >= 3 vs rest."""
    name = "LogReg (raw columns)"

    def fit(self, df, features):
        self.m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(max_iter=2000))
        self.m.fit(df[RAW_COLUMNS], (df["relevance_label"] >= 3).astype(int))
        return self

    def predict(self, df):
        return self.m.predict_proba(df[RAW_COLUMNS])[:, 1]


class RidgeEngineered:
    """Linear model on the engineered features (pointwise regression on the label)."""
    name = "Ridge (engineered)"

    def fit(self, df, features):
        self.features = features
        self.m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        self.m.fit(df[features], df["relevance_label"])
        return self

    def predict(self, df):
        return self.m.predict(df[self.features])


class LGBMPointwise:
    name = "LightGBM regressor (pointwise)"

    def __init__(self, **params):
        self.params = {**LGBM_PARAMS, **params}

    def fit(self, df, features):
        self.features = features
        self.m = LGBMRegressor(**self.params).fit(df[features], df["relevance_label"])
        return self

    def predict(self, df):
        return self.m.predict(df[self.features])


class LGBMLambdaRank:
    name = "LightGBM LambdaRank"
    short = "LambdaRank"

    def __init__(self, **params):
        self.params = {**LGBM_PARAMS, **params}

    def fit(self, df, features):
        self.features = features
        d = _sorted(df)
        self.m = LGBMRanker(objective="lambdarank", lambdarank_truncation_level=20, **self.params)
        self.m.fit(d[features], d["relevance_label"], group=_groups(d))
        return self

    def predict(self, df):
        return self.m.predict(df[self.features])


class XGBNDCG:
    name = "XGBoost rank:ndcg"
    short = "XGBoost"

    def __init__(self, **params):
        self.params = dict(n_estimators=600, learning_rate=0.03, max_depth=6, min_child_weight=5,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                           random_state=SEED, n_jobs=-1, lambdarank_pair_method="topk",
                           lambdarank_num_pair_per_sample=10)
        self.params.update(params)

    def fit(self, df, features):
        self.features = features
        d = _sorted(df)
        self.m = XGBRanker(objective="rank:ndcg", **self.params)
        self.m.fit(d[features], d["relevance_label"], qid=d["job_id"].to_numpy())
        return self

    def predict(self, df):
        return self.m.predict(df[self.features])


class RankAverage:
    """Average of the within-job rank of several models' scores."""

    def __init__(self, models):
        self.models = models
        self.name = "Rank-average ensemble (" + " + ".join(m.short for m in models) + ")"

    def fit(self, df, features):
        for m in self.models:
            m.fit(df, features)
        return self

    def predict(self, df):
        ranks = [pd.Series(m.predict(df), index=df.index).groupby(df["job_id"]).rank(pct=True)
                 for m in self.models]
        return np.mean(ranks, axis=0)

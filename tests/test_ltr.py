import numpy as np
import pandas as pd
import pytest

from ltr.data import norm_title, parse_mixed_dates, split_pipe
from ltr.encoding import TE_FEATURES, oof_encode
from ltr.features import families, seniority
from ltr.metrics import ap_at_k, dcg, ndcg_at_k, summarize


# ---------- metrics ----------
def test_ndcg_perfect_and_reversed():
    y = np.array([4, 3, 2, 1, 0])
    assert ndcg_at_k(y, -np.arange(5)) == pytest.approx(1.0)
    rev = ndcg_at_k(y, np.arange(5))
    assert 0 < rev < 1


def test_ndcg_hand_computed():
    # ranking puts label 1 first, label 3 second
    y = np.array([3, 1])
    got = ndcg_at_k(y, np.array([0.1, 0.9]), k=10)
    expected = (1 / np.log2(2) + 7 / np.log2(3)) / (7 / np.log2(2) + 1 / np.log2(3))
    assert got == pytest.approx(expected)


def test_ndcg_undefined_when_all_zero():
    assert ndcg_at_k(np.zeros(4), np.arange(4)) is None


def test_ndcg_only_top_k_counts():
    y = np.array([0] * 10 + [4])
    assert ndcg_at_k(y, -np.arange(11), k=10) == 0.0
    assert dcg([4], 1) == 15.0


def test_ap_at_5():
    # relevant (>=3) at ranks 1 and 3, two relevant in total
    y = np.array([3, 0, 4, 0, 0, 0])
    got = ap_at_k(y, -np.arange(6), k=5)
    assert got == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert ap_at_k(np.array([0, 1, 2]), np.arange(3)) is None


def test_summarize_averages_over_jobs():
    df = pd.DataFrame({"job_id": [1, 1, 2, 2], "relevance_label": [4, 0, 0, 4], "score": [1, 0, 1, 0]})
    s = summarize(df, "score")
    assert s["jobs"] == 2
    assert s["ndcg10"] == pytest.approx((1 + 1 / np.log2(3) / 1) / 2)


# ---------- cleaning ----------
def test_parse_mixed_dates():
    out = parse_mixed_dates(pd.Series(["2024-07-03", "05/06/2024", "bad"]))
    assert out.iloc[0] == pd.Timestamp("2024-07-03")
    assert out.iloc[1] == pd.Timestamp("2024-06-05")  # day first
    assert pd.isna(out.iloc[2])


def test_norm_title_abbreviations():
    s = pd.Series(["Sr. Backend Developer", "JR. ML ENGINEER ", "Solutions Architect (Sr.)"])
    assert norm_title(s).tolist() == ["senior backend developer", "junior ml engineer",
                                      "solutions architect (senior)"]


def test_split_pipe():
    assert split_pipe(pd.Series(["Python| SQL |python", None])).tolist() == [frozenset({"python", "sql"}), frozenset()]


def test_seniority_and_families():
    assert seniority("junior web developer") == 0
    assert seniority("software engineer") == 1
    assert seniority("senior data scientist") == 2
    assert seniority("head of data") == 3
    assert "frontend" in families("full stack developer") and "backend" in families("full stack developer")


# ---------- target encoding ----------
def _toy_frame(n_jobs=40, per_job=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_jobs):
        for i in range(per_job):
            rows.append({"job_id": j, "relevance_label": int(rng.integers(0, 5)),
                         "key_title_pair": f"t{i % 3}", "key_industry": f"ind{j % 2}",
                         "certs_set": frozenset({"aws-csa"}) if i % 2 else frozenset()})
    return pd.DataFrame(rows)


def test_oof_encoding_ignores_own_job_labels():
    df = _toy_frame()
    a = oof_encode(df)
    df2 = df.copy()
    df2.loc[df2["job_id"] == 0, "relevance_label"] = 4 - df2.loc[df2["job_id"] == 0, "relevance_label"]
    b = oof_encode(df2)
    rows_job0 = df["job_id"] == 0
    # job 0's own encodings do not depend on job 0's labels
    np.testing.assert_allclose(a.loc[rows_job0, TE_FEATURES], b.loc[rows_job0, TE_FEATURES])

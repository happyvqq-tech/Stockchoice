import numpy as np
import pandas as pd

from picker.config import load_config
from picker.features.build import FEATURES
from picker.models.lgbm import fit_window, predict, reasons, split_train_val

CFG = load_config()


def make_df(n_days=700, n_sym=40, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    rows = []
    for d in dates:
        X = rng.random((n_sym, len(FEATURES)))
        f = pd.DataFrame(X, columns=FEATURES)
        f["date"], f["symbol"] = d, np.arange(n_sym)
        score = f["close_to_ma20"] + rng.normal(0, 0.3, n_sym) if signal else rng.random(n_sym)
        f["label"] = (score.rank(pct=True, ascending=False) <= 0.2).astype(float)
        f["exit_date"] = d + pd.tseries.offsets.BDay(10)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def test_purge_keeps_labels_out_of_test_period():
    df = make_df()
    ts = pd.Timestamp("2020-06-01")
    fit, val = split_train_val(df, ts)
    assert fit["exit_date"].max() < ts and val["exit_date"].max() < ts
    assert fit["date"].max() < val["date"].min() and fit["exit_date"].max() < val["date"].min()
    assert len(fit) > 0 and len(val) > 0


def test_lgbm_learns_signal_and_not_noise():
    cfg = dict(CFG)
    df = make_df()
    ts = pd.Timestamp("2020-06-01")
    m, info = fit_window(df, cfg, ts)
    test = df[(df["date"] >= ts) & (df["date"] < ts + pd.DateOffset(months=1))]
    s = predict(m, test)
    from_ic = s.groupby(test["date"]).apply(lambda x: x.rank().corr(test.loc[x.index, "close_to_ma20"].rank()))
    assert from_ic.mean() > 0.4
    assert abs(info["importance"].sum() - 1) < 1e-6 and info["importance"].idxmax() == "close_to_ma20"
    top = test[test["close_to_ma20"] > 0.95].head(5)
    r = reasons(m, top)
    assert len(r) == len(top) and "股價相對月線" in " ".join(r)


def test_too_little_data_returns_none():
    df = make_df(n_days=40)
    m, info = fit_window(df, CFG, pd.Timestamp("2019-02-15"))
    assert m is None

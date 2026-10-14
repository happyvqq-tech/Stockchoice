"""RuleScore（規格 8.1）：選定特徵的截面排名百分位，依方向等權平均。研究用途，非投資建議。"""
import numpy as np

RULE_FEATURES = {  # 特徵: 方向（+1 越大越好 / -1 越小越好）
    "rel_strength_20": 1, "close_to_ma20": 1, "ma20_slope": 1, "vol_ratio_20": 1,
    "close_to_high20": 1, "trust_net_5d": 1, "foreign_net_5d": 1, "rev_yoy": 1,
}
MIN_FEATURES = 6  # 8 個特徵至少要有 6 個有值才打分


def rule_score(ranked, features=None):
    """ranked: 已做截面排名標準化的面板。回傳 Series（沒有足夠特徵者為 NaN）。"""
    feats = features or RULE_FEATURES
    cols = []
    for f, d in feats.items():
        r = ranked[f]
        cols.append(r if d > 0 else 1 - r)
    m = np.column_stack([c.values for c in cols])
    ok = (~np.isnan(m)).sum(axis=1) >= min(MIN_FEATURES, len(feats))
    with np.errstate(all="ignore"):
        s = np.nanmean(m, axis=1)
    return ranked["date"].to_frame().assign(score=np.where(ok, s, np.nan))["score"]

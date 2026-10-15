"""LightGBM（規格 8.2）：二元分類（是否為當日股票池超額報酬前 20%），輸出機率當分數。
固定參數、不大規模調參。Early stopping 只用訓練窗口最後 3 個月，絕不看測試期。研究用途，非投資建議。"""
import numpy as np
import pandas as pd

from picker.features.build import FEATURE_NAMES, FEATURES


def _params(cfg):
    p = cfg["lgbm"]
    return dict(objective="binary", metric="binary_logloss", num_leaves=p["num_leaves"], learning_rate=p["learning_rate"],
                feature_fraction=p["feature_fraction"], bagging_fraction=p["bagging_fraction"], bagging_freq=p["bagging_freq"],
                min_data_in_leaf=p["min_child_samples"], seed=p["seed"], verbose=-1, num_threads=2)


def split_train_val(df, test_start, val_months=3):
    """訓練列：date < test_start 且標籤的出場日也 < test_start（purge，避免標籤跨進測試期）。
    再切出最後 val_months 個月當驗證集；訓練集的出場日必須早於驗證起點（兩者之間同樣 purge）。"""
    test_start = pd.Timestamp(test_start)
    tr = df[(df["date"] < test_start) & (df["exit_date"] < test_start) & df["label"].notna()]
    val_start = test_start - pd.DateOffset(months=val_months)
    fit = tr[(tr["date"] < val_start) & (tr["exit_date"] < val_start)]
    val = tr[tr["date"] >= val_start]
    return fit, val


def fit_window(df, cfg, test_start, features=None):
    """回傳 (booster, info)。df 需含 date, exit_date, label 與特徵欄。資料太少時回傳 (None, info)。"""
    import lightgbm as lgb
    feats = features or FEATURES
    fit, val = split_train_val(df, test_start)
    if len(fit) < 1000 or len(val) < 200 or fit["label"].nunique() < 2:
        return None, {"fit_rows": len(fit), "val_rows": len(val)}
    dtr = lgb.Dataset(fit[feats], fit["label"])
    dva = lgb.Dataset(val[feats], val["label"], reference=dtr)
    m = lgb.train(_params(cfg), dtr, num_boost_round=cfg["lgbm"]["n_estimators"], valid_sets=[dva],
                  callbacks=[lgb.early_stopping(cfg["lgbm"]["early_stopping_rounds"], verbose=False)])
    imp = pd.Series(m.feature_importance("gain"), index=feats)
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    return m, {"fit_rows": len(fit), "val_rows": len(val), "best_iter": int(m.best_iteration), "importance": imp}


def predict(model, df, features=None):
    return pd.Series(model.predict(df[features or FEATURES], num_iteration=model.best_iteration), index=df.index)


def reasons(model, df, features=None, top=3):
    """每檔貢獻最大的前 top 個特徵（pred_contrib），轉成中文。只列正向貢獻。"""
    feats = features or FEATURES
    c = model.predict(df[feats], num_iteration=model.best_iteration, pred_contrib=True)[:, :-1]
    out = []
    for row in c:
        idx = np.argsort(-row)[:top]
        out.append("、".join(f"{FEATURE_NAMES[feats[i]]}" for i in idx if row[i] > 0))
    return pd.Series(out, index=df.index)

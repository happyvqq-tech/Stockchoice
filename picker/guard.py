"""期末考（holdout）保護（規格第 10 節、核心規則 2）。研究用途，非投資建議。"""
import pandas as pd


class HoldoutError(RuntimeError):
    pass


def assert_no_holdout(dates, cfg, final_exam=False, what="資料"):
    """任何訓練或評估用到的日期（含進出場日）只要 ≥ holdout_start 就報錯，除非加 --final-exam。"""
    if final_exam:
        return
    h = pd.Timestamp(cfg["holdout_start"])
    d = pd.to_datetime(pd.Series(list(dates))).dropna()
    if len(d) and d.max() >= h:
        raise HoldoutError(f"{what}碰到期末考期間（{d.max().date()} ≥ {h.date()}）。"
                           "開發期間禁止；真要做最終考試請加 --final-exam，並記錄在實驗日誌。")

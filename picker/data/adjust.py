"""用除權息資料自行計算還原價（FinMind 免費版沒有 TaiwanStockPriceAdj）。
報酬計算用 adj_close；漲跌停判斷用原始 close。研究用途，非投資建議。"""
import pandas as pd


def add_adj_close(bars, dividends):
    """bars: 單一股票的日線（date, close …）；dividends: date, before_price, after_price（date＝除權息日）。
    除權息日當天起價格已反映除權息，所以只有「除權息日之前」的價格要乘上 after/before。"""
    bars = bars.sort_values("date").reset_index(drop=True).copy()
    factor = pd.Series(1.0, index=bars.index)
    if dividends is not None and len(dividends):
        for ex, before, after in zip(dividends["date"], dividends["before_price"], dividends["after_price"]):
            if pd.notna(before) and pd.notna(after) and before > 0 and after > 0:
                factor[bars["date"] < pd.Timestamp(ex)] *= after / before
    bars["adj_close"] = bars["close"] * factor
    return bars


def combine_adjustments(dividends=None, splits=None, reductions=None):
    """合併三種還原事件。同一檔、同一天若同時有分割/減資與除權息，以分割/減資為準（避免重複還原）。"""
    parts = []
    need = {"symbol", "date", "before_price", "after_price"}
    for prio, df in ((0, splits), (0, reductions), (1, dividends)):
        if df is not None and len(df) and need <= set(df.columns):
            parts.append(df[["symbol", "date", "before_price", "after_price"]].assign(_prio=prio))
    if not parts:
        return pd.DataFrame(columns=["symbol", "date", "before_price", "after_price"])
    allv = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date", "_prio"])
    return allv.drop_duplicates(["symbol", "date"], keep="first").drop(columns="_prio").reset_index(drop=True)

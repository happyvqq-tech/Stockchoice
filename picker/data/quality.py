"""資料品質檢查（規格 5.5），輸出 data_quality_report.md。研究用途，非投資建議。"""
import pandas as pd


def check_bars(bars, calendar):
    """bars: 所有股票原始日線。回傳 dict of DataFrame / 數字。"""
    cal = pd.DatetimeIndex(pd.to_datetime(calendar))
    res = {}
    dup = bars[bars.duplicated(["symbol", "date"], keep=False)]
    res["duplicates"] = dup[["symbol", "date"]].head(50)
    bad = bars[(bars[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    res["nonpositive"] = bad[["symbol", "date", "open", "high", "low", "close"]].head(50)
    res["not_in_calendar"] = int((~bars["date"].isin(cal)).sum())
    b = bars.sort_values(["symbol", "date"]).copy()
    b["ret"] = b.groupby("symbol")["close"].pct_change()
    res["big_moves"] = b[b["ret"].abs() > 0.10][["symbol", "date", "close", "ret"]].sort_values("ret", key=abs, ascending=False).head(30)
    res["n_big_moves"] = int((b["ret"].abs() > 0.10).sum())
    # 缺漏交易日：個股在上市區間內缺幾天（停牌也會算進來，僅供參考）
    gaps = []
    for sym, g in bars.groupby("symbol"):
        span = cal[(cal >= g["date"].min()) & (cal <= g["date"].max())]
        gaps.append((sym, len(span) - g["date"].nunique(), len(span)))
    gp = pd.DataFrame(gaps, columns=["symbol", "missing_days", "span_days"])
    gp["missing_pct"] = gp["missing_days"] / gp["span_days"].where(gp["span_days"] > 0)
    res["gaps"] = gp.sort_values("missing_pct", ascending=False).head(30)
    res["median_missing_pct"] = float(gp["missing_pct"].median()) if len(gp) else 0.0
    res["coverage"] = bars.groupby(bars["date"].dt.year)["symbol"].nunique().rename("檔數").reset_index().rename(columns={"date": "年"})
    return res


def render_report(res, n_symbols, n_rows, extra=""):
    def tbl(df):
        return df.to_markdown(index=False) if len(df) else "（無）"
    return "\n".join([
        "# 資料品質報告", "", "> 研究用途，非投資建議。", "",
        f"- 股票檔數：{n_symbols}　日線總列數：{n_rows}",
        f"- 重複列：{len(res['duplicates'])}　價格 ≤ 0：{len(res['nonpositive'])}　不在交易日曆內的日期：{res['not_in_calendar']}",
        f"- 單日漲跌超過 10% 的筆數：{res['n_big_moves']}（台股漲跌幅 10%，超過者多為除權息日或新上市日，待查）",
        f"- 個股缺漏交易日比例中位數：{res['median_missing_pct']:.2%}（含停牌）", "",
        "## 各年度有資料的檔數", tbl(res["coverage"]), "",
        "## 缺漏最多的個股（前 30）", tbl(res["gaps"]), "",
        "## 單日漲跌最大的紀錄（前 30）", tbl(res["big_moves"]), "",
        "## 重複列（前 50）", tbl(res["duplicates"]), "",
        "## 價格 ≤ 0（前 50）", tbl(res["nonpositive"]), "", extra])

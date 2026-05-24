# =====================================================================
#  美股策略分析師 — Streamlit 網頁版 (app.py)
#  框架 Ver. 2026.6 | 自己手機可用的網頁工具
# =====================================================================
#  本地測試： streamlit run app.py
#  部署：見同資料夾的「部署教學.md」
#  ⚠️ 僅供個人研究參考，非投資建議。資料源 yfinance 僅限個人非商業用途。
# =====================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, json, urllib.request

# ---------------------------------------------------------------
#  設定
# ---------------------------------------------------------------
SEC_UA = {"User-Agent": "stock-analyst-learning happyvqq@famunity.cloud"}
_CIK_CACHE = {}

WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
             "AMD", "AVGO", "NFLX", "COST", "JPM", "V", "WMT", "LLY",
             "UNH", "CRM", "ORCL", "PLTR", "MU"]
MARKET_ETFS = ["SPY", "QQQ"]
SECTOR_ETF = {"Technology": "XLK", "Communication Services": "XLC",
              "Healthcare": "XLV", "Financial Services": "XLF",
              "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
              "Energy": "XLE", "Industrials": "XLI", "Basic Materials": "XLB",
              "Real Estate": "XLRE", "Utilities": "XLU"}
SIMILAR_UNIVERSE = [
    "NVDA", "AMD", "AVGO", "MU", "QCOM", "ARM", "ASML", "AMAT", "LRCX", "TSM",
    "MSFT", "GOOGL", "META", "PLTR", "CRWD", "SNOW", "DDOG", "NET", "PANW",
    "NOW", "ANET", "MDB", "AMZN", "TSLA", "SHOP", "MELI", "ABNB", "UBER",
    "NFLX", "COST", "NKE", "LLY", "UNH", "ISRG", "NVO", "V", "MA", "AXP",
    "ENPH", "FSLR", "PYPL", "COIN", "HOOD", "SOFI", "CELH", "DKNG", "RBLX",
    "INTC", "ORCL", "ADBE", "SMCI", "WMT", "SBUX"]
MA_DEF, MA_TREND = 20, 200
RSI_LO, RSI_HI = 40, 60
VOL_SPIKE, VOL_RESONANCE, EPS_DROP = 1.5, 2.0, 0.15

# ---------------------------------------------------------------
#  基礎工具
# ---------------------------------------------------------------
def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default

def _rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + g / l)

def _slope_up(s, lb=5):
    s = s.dropna()
    return None if len(s) < lb + 1 else float(s.iloc[-1]) > float(s.iloc[-1 - lb])

def _pct(p, m):
    return (p - m) / m * 100 if m else None

# 用 Streamlit 快取，避免重複抓同一檔（10分鐘內重用）
@st.cache_data(ttl=600, show_spinner=False)
def _hist(ticker, period="1y"):
    h = _safe(lambda: yf.Ticker(ticker).history(period=period))
    return h if (h is not None and not h.empty) else None

@st.cache_data(ttl=600, show_spinner=False)
def _bulk(tickers, period="1y"):
    d = _safe(lambda: yf.download(tickers, period=period, group_by="ticker",
                                  auto_adjust=True, progress=False))
    return d

# ---------------------------------------------------------------
#  M1 大盤
# ---------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def assess_market():
    above, slopes = [], []
    for etf in MARKET_ETFS:
        h = _hist(etf, "3mo")
        if h is None or len(h) < MA_DEF + 6:
            continue
        ma = h["Close"].rolling(MA_DEF).mean()
        above.append(float(h["Close"].iloc[-1]) > float(ma.iloc[-1]))
        sp = _slope_up(ma)
        if sp is not None:
            slopes.append(sp)
    if not above:
        return "Neutral", "⚠️ 無法取得大盤資料"
    if sum(above) == len(above) and sum(slopes) == len(slopes):
        return "Risk-On", "指數站上20MA且斜率向上"
    if sum(above) == 0 and sum(slopes) == 0:
        return "Risk-Off", "指數跌破20MA且斜率向下"
    return "Neutral", "位置與斜率分歧，高檔震盪"

# ---------------------------------------------------------------
#  Stooq 交叉驗證 + EDGAR 財報
# ---------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def stooq_close(ticker):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=10).read().decode()
        if "Date" not in raw:
            return None
        df = pd.read_csv(io.StringIO(raw))
        if df.empty or "Close" not in df.columns:
            return None
        return float(df["Close"].dropna().astype(float).iloc[-1])
    except Exception:
        return None

def cross_check(ticker, yf_close):
    sq = stooq_close(ticker)
    if sq is None:
        return False, "Stooq無資料（單源，信心降一級）"
    diff = abs(yf_close - sq) / yf_close * 100
    if diff > 1.0:
        return True, f"⚠️ 價格源衝突 yf {yf_close:.2f} vs Stooq {sq:.2f}（差{diff:.1f}%）"
    return False, f"✅ 雙源一致 yf {yf_close:.2f}/Stooq {sq:.2f}（差{diff:.2f}%）"

@st.cache_data(ttl=86400, show_spinner=False)
def _cik(ticker):
    global _CIK_CACHE
    if not _CIK_CACHE:
        try:
            req = urllib.request.Request(
                "https://www.sec.gov/files/company_tickers.json", headers=SEC_UA)
            data = json.loads(urllib.request.urlopen(req, timeout=10).read())
            for row in data.values():
                _CIK_CACHE[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
        except Exception:
            return None
    return _CIK_CACHE.get(ticker.upper())

def _concept(cik, tag):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    try:
        req = urllib.request.Request(url, headers=SEC_UA)
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        q = {}
        for it in data["units"]["USD"]:
            if it.get("form") not in ("10-Q", "10-K"):
                continue
            s, e = it.get("start"), it.get("end")
            if not s or not e:
                continue
            if 80 <= (pd.Timestamp(e) - pd.Timestamp(s)).days <= 100:
                q[e] = float(it["val"])
        return sorted(q.items())
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def edgar_fundamentals(ticker):
    cik = _cik(ticker)
    if not cik:
        return None
    ni = _concept(cik, "NetIncomeLoss")
    if len(ni) < 5:
        return None
    o = {"source": "SEC EDGAR", "eps_yoy": None, "gross": None, "downgrade": False, "notes": []}
    try:
        if ni[-5][1] != 0:
            o["eps_yoy"] = (ni[-1][1] - ni[-5][1]) / abs(ni[-5][1])
    except Exception:
        pass
    gp = _concept(cik, "GrossProfit")
    rev = _concept(cik, "RevenueFromContractWithCustomerExcludingAssessedTax") or _concept(cik, "Revenues")
    try:
        if len(gp) >= 3 and len(rev) >= 3:
            gpm, revm = dict(gp), dict(rev)
            common = sorted(set(gpm) & set(revm))[-3:]
            if len(common) == 3:
                gm = [gpm[e] / revm[e] for e in common if revm[e]]
                if len(gm) == 3 and gm[0] > gm[1] > gm[2]:
                    o["gross"] = "連兩季下滑"
    except Exception:
        pass
    if o["eps_yoy"] is not None and o["eps_yoy"] <= -EPS_DROP:
        o["downgrade"] = True
        o["notes"].append(f"[EDGAR] 核心淨利YoY {o['eps_yoy']*100:.0f}%，評級降級")
    if o["gross"]:
        o["notes"].append("[EDGAR] 毛利率連兩季下滑")
    o["notes"].append("✅ 財報以 SEC EDGAR 官方為準")
    return o

# ---------------------------------------------------------------
#  S0–S3 / E 診斷
# ---------------------------------------------------------------
def _diag_s0_yf(fin):
    o = {"source": "yfinance(二手)", "eps_yoy": None, "gross": None, "downgrade": False, "notes": []}
    if fin is None or getattr(fin, "empty", True):
        o["notes"].append("⚠️ 無財報資料"); return o
    try:
        idx = {str(i).lower(): i for i in fin.index}
        def row(*ns):
            for n in ns:
                for k, orig in idx.items():
                    if n in k: return fin.loc[orig]
            return None
        ni, rev, gp = row("net income"), row("total revenue", "revenue"), row("gross profit")
        if ni is not None and len(ni) >= 5 and float(ni.iloc[4]) != 0:
            o["eps_yoy"] = (float(ni.iloc[0]) - float(ni.iloc[4])) / abs(float(ni.iloc[4]))
        if gp is not None and rev is not None and len(gp) >= 3 and len(rev) >= 3:
            gm = [float(gp.iloc[i]) / float(rev.iloc[i]) for i in range(3) if float(rev.iloc[i])]
            if len(gm) == 3 and gm[0] < gm[1] < gm[2]: o["gross"] = "連兩季下滑"
        if o["eps_yoy"] is not None and o["eps_yoy"] <= -EPS_DROP:
            o["downgrade"] = True
        o["notes"].append("⚠️ 未剔除一次性項目，且為二手資料")
    except Exception:
        o["notes"].append("⚠️ 財報解析失敗")
    return o

def _diag_s1(h):
    c, v = h["Close"], h["Volume"]
    ma = c.rolling(MA_DEF).mean()
    p, m = float(c.iloc[-1]), float(ma.iloc[-1])
    v5 = float(v.rolling(5).mean().iloc[-1])
    b2 = bool(c.iloc[-1] < ma.iloc[-1] and c.iloc[-2] < ma.iloc[-2])
    bb = bool(p < m and float(c.iloc[-1] / c.iloc[-2] - 1) < -0.02 and float(v.iloc[-1]) >= VOL_SPIKE * v5)
    return {"price": p, "ma20": m, "above": p > m, "broke": b2 or bb,
            "low5": float(c.tail(5).min()), "prev": float(c.iloc[-2]),
            "note": "連2日收盤<20MA" if b2 else "放量大黑K破20MA" if bb else "未破位"}

def _diag_s2(ins, sr):
    o = {"level": "常態", "notes": []}
    if ins is not None and not getattr(ins, "empty", True):
        try:
            txt = ins.astype(str).apply(lambda r: " ".join(r), axis=1).str.lower()
            s = int(txt.str.contains("sale|sell").sum()); b = int(txt.str.contains("purchase|buy").sum())
            o["notes"].append(f"內部人申報：賣{s}/買{b}")
            if s >= 3 and s > b: o["level"] = "注意區"
            o["notes"].append("⚠️ 無法辨識10b5-1計畫性賣股")
        except Exception:
            o["notes"].append("⚠️ 內部人解析失敗")
    else:
        o["notes"].append("⚠️ 無內部人資料")
    if sr:
        o["notes"].append(f"放空比 {sr:.1f} 天" + ("（偏高）" if sr >= 7 else ""))
    return o

def _sector_ret(sector):
    etf = SECTOR_ETF.get(sector, "SPY")
    h = _hist(etf, "2mo")
    if h is None or len(h) < 21: return 0.0, etf
    c = h["Close"]; return float(c.iloc[-1] / c.iloc[-21] - 1), etf

def _diag_s3(h, sret, setf):
    c = h["Close"]
    s20 = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) >= 21 else 0.0
    at_high = float(c.iloc[-1]) >= float(c.tail(20).max()) * 0.995
    flag = sret > 0.03 and not at_high and s20 < sret - 0.03
    note = (f"⚠️ 板塊({setf})20日漲{sret*100:.1f}%，個股僅{s20*100:.1f}%且未創高→疑似不聯動"
            if flag else "個股與板塊大致同步")
    return {"unlinked": flag, "note": note}

def _diag_ex(h, sret, earn):
    c, v = h["Close"], h["Volume"]
    p = float(c.iloc[-1]); m = float(c.rolling(MA_DEF).mean().iloc[-1])
    dev = _pct(p, m); v5 = float(v.rolling(5).mean().iloc[-1])
    thr = max(10, min(30, 15 * (1 + sret)))
    g15 = float(c.iloc[-1] / c.iloc[-16] - 1) if len(c) >= 16 else None
    e6p = p >= float(c.max()) * 0.999; e6v = float(v.iloc[-1]) >= VOL_RESONANCE * v5
    e7, e7n = False, "⚠️ 無財報日"
    if earn is not None and not getattr(earn, "empty", True):
        try:
            now = pd.Timestamp.now(tz=earn.index.tz)
            fut = earn.index[earn.index > now]
            if len(fut):
                dd = (fut.min() - now).days; e7 = dd <= 5
                e7n = f"下次財報約{dd}天後" + ("（鎖定期）" if e7 else "")
            else: e7n = "近期無排定財報"
        except Exception: e7n = "⚠️ 財報日解析失敗"
    return {"dev": dev, "thr": thr, "e5": dev is not None and dev > thr,
            "g15": g15, "e3": g15 is not None and g15 >= 0.20,
            "e6pv": e6p and e6v, "e6p": e6p, "e6v": e6v, "e7": e7, "e7n": e7n}

def _decide(ms, s0, s1, s2, s3, ex):
    action, pos, add, hold = "持有", "40%–70%", "需量能確認", "可"
    if ms == "Risk-Off": action, pos, add = "凍結加碼", "≤40%", "禁止(Risk-Off)"
    if s3["unlinked"]: action, pos, add = "凍結加碼", "≤50%", "禁止(龍頭不聯動)"
    if ex["e5"]:
        add = "禁止追價(E5乖離超限)"
        if ms == "Risk-On" and s1["above"] and not s1["broke"] and s2["level"] in ("常態", "注意區"):
            action, hold = "可續抱但不追價", "可(E6-B強趨勢豁免)"
        else: action = "可續抱但不追價"
    if ex["e7"]: pos, add = "≤30%–50%", "限制重倉(E7鎖定期)"
    if s2["level"] == "注意區": add = "凍結加碼(內部人賣)"
    if s1["broke"]: action, pos, add, hold = "減碼/出場", "20%–40%", "禁止", "視站回20MA"
    if s0["downgrade"]: action, hold = "減碼(基本面降級)", "保守"
    if (ms == "Risk-On" and s1["above"] and not s1["broke"] and s2["level"] == "常態"
            and not s0["downgrade"] and not s3["unlinked"] and not ex["e5"] and not ex["e7"]):
        action, pos, add = "持有/可分批加碼", "70%–100%", "可(量能配合)"
    return action, pos, add, hold

# ---------------------------------------------------------------
#  相似 + 回測
# ---------------------------------------------------------------
def _features(close):
    c = close.dropna()
    if len(c) < 130: return None
    rets = c.pct_change().dropna()
    ma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else c.mean()
    return {"r1m": float(c.iloc[-1]/c.iloc[-21]-1), "r3m": float(c.iloc[-1]/c.iloc[-63]-1),
            "r6m": float(c.iloc[-1]/c.iloc[-126]-1), "vol": float(rets.std()*np.sqrt(252)),
            "rsi": float(_rsi(c).iloc[-1]), "pos200": float(c.iloc[-1]/ma200-1)}

def _sim_pick(feats, ref, top_n):
    fdf = pd.DataFrame(feats).T
    z = (fdf - fdf.mean()) / fdf.std(ddof=0).replace(0, 1)
    w = pd.Series({"r1m":1.3,"r3m":1.5,"r6m":1.3,"vol":0.8,"rsi":0.7,"pos200":1.0})
    zw = z * w
    dist = ((zw - zw.loc[ref]) ** 2).sum(axis=1) ** 0.5
    return dist.drop(ref).sort_values().head(top_n)

# =====================================================================
#  Streamlit 介面
# =====================================================================
st.set_page_config(page_title="美股策略分析師", page_icon="📈", layout="centered")
st.title("📈 美股策略分析師")
st.caption("框架 Ver.2026.6 · 個人研究參考 · 非投資建議")

with st.sidebar:
    st.header("功能選單")
    mode = st.radio("選擇功能", ["📊 篩選股票", "🔍 個股分析", "🚀 找相似飆股", "🧪 批次回測"])
    st.divider()
    ms, ms_note = assess_market()
    color = {"Risk-On": "🟢", "Neutral": "🟡", "Risk-Off": "🔴"}[ms]
    st.metric("大盤 M1 狀態", f"{color} {ms}")
    st.caption(ms_note)

# ---- 功能一：篩選 ----
if mode == "📊 篩選股票":
    st.subheader("篩選股票")
    c1, c2 = st.columns(2)
    f_mom = c1.checkbox("近3月報酬>0", True)
    f_vol = c2.checkbox("量能放大", True)
    f_high = c1.checkbox("靠近52週高(85%)", False)
    rlo, rhi = st.slider("RSI 區間", 0, 100, (RSI_LO, RSI_HI))
    if st.button("開始篩選", type="primary"):
        rows = []
        prog = st.progress(0.0)
        for i, t in enumerate(WATCHLIST):
            prog.progress((i + 1) / len(WATCHLIST), f"掃描 {t}")
            h = _hist(t, "1y")
            if h is None or len(h) < MA_TREND: continue
            c, v = h["Close"], h["Volume"]
            p = float(c.iloc[-1]); m200 = float(c.rolling(MA_TREND).mean().iloc[-1])
            m20 = float(c.rolling(MA_DEF).mean().iloc[-1]); rsi = float(_rsi(c).iloc[-1])
            v5 = float(v.rolling(5).mean().iloc[-1]); v20 = float(v.rolling(20).mean().iloc[-1])
            r3 = float(c.iloc[-1]/c.iloc[-63]-1) if len(c) >= 63 else 0.0
            ok = p > m200 and rlo <= rsi <= rhi and p > m20
            if f_mom: ok = ok and r3 > 0
            if f_vol: ok = ok and v5 > v20
            if f_high: ok = ok and p >= float(c.tail(252).max()) * 0.85
            rows.append({"代號": t, "價": round(p, 2), "RSI": round(rsi, 1),
                         "3月%": round(r3*100, 1), ">200MA": p > m200, ">20MA": p > m20,
                         "量增": v5 > v20, "候選": ok})
        prog.empty()
        df = pd.DataFrame(rows)
        cand = df[df["候選"]]
        st.success(f"找到 {len(cand)} 檔候選")
        st.dataframe(cand, use_container_width=True, hide_index=True)
        with st.expander("看全部掃描結果"):
            st.dataframe(df, use_container_width=True, hide_index=True)

# ---- 功能二：個股分析 ----
elif mode == "🔍 個股分析":
    st.subheader("個股深度分析")
    ticker = st.text_input("股票代號", "NVDA").upper().strip()
    if st.button("分析", type="primary"):
        h = _hist(ticker, "1y")
        if h is None:
            st.error(f"{ticker}：抓不到資料")
        else:
            tk = yf.Ticker(ticker)
            info = _safe(lambda: tk.info, {}) or {}
            sret, setf = _sector_ret(info.get("sector"))
            edg = edgar_fundamentals(ticker)
            s0 = edg if edg else _diag_s0_yf(_safe(lambda: tk.quarterly_income_stmt))
            s1 = _diag_s1(h)
            s2 = _diag_s2(_safe(lambda: tk.insider_transactions), info.get("shortRatio"))
            s3 = _diag_s3(h, sret, setf)
            ex = _diag_ex(h, sret, _safe(lambda: tk.get_earnings_dates(limit=8)))
            action, pos, add, hold = _decide(ms, s0, s1, s2, s3, ex)
            conflict, xnote = cross_check(ticker, s1["price"])
            if conflict and "加碼" in add:
                add = "凍結加碼（資料源衝突）"

            # 結論卡片
            verdict_color = ("red" if ("減碼" in action or "出場" in action)
                             else "orange" if ("凍結" in action or "不追價" in action) else "green")
            st.markdown(f"### :{verdict_color}[{action}]")
            m1, m2, m3 = st.columns(3)
            m1.metric("倉位上限", pos)
            m2.metric("現價", f"{s1['price']:.2f}")
            m3.metric("20MA", f"{s1['ma20']:.2f}", "站上" if s1["above"] else "跌破")
            st.caption(f"新增部位：{add}　|　續抱：{hold}")

            # 價量圖
            d = h.tail(126)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), height_ratios=[3, 1], sharex=True)
            ax1.plot(d.index, d["Close"], color="#111", lw=1.3, label="Close")
            ax1.plot(d.index, h["Close"].rolling(20).mean().tail(126), color="#e8853a", lw=1, label="20MA")
            ax1.plot(d.index, h["Close"].rolling(200).mean().tail(126), color="#3a6ee8", lw=1, label="200MA")
            ax1.axhline(s1["low5"], color="#888", ls=":", lw=0.9, label="5d low")
            ax1.legend(fontsize=7); ax1.grid(alpha=0.25)
            colors = ["#c0392b" if d["Close"].iloc[i] < d["Close"].iloc[i-1] else "#27ae60"
                      for i in range(len(d))]
            ax2.bar(d.index, d["Volume"], color=colors, width=1)
            ax2.grid(alpha=0.25)
            st.pyplot(fig)

            # 燈號
            st.markdown("#### 診斷燈號")
            def badge(label, ok, good="OK", bad="風險"):
                return f"- **{label}**：{'🟢 ' + good if ok else '🔴 ' + bad}"
            st.markdown("\n".join([
                f"- **M1 大盤**：{ms}",
                badge("S1 20MA", s1["above"], "站上", "跌破"),
                badge("S0 基本面", not s0["downgrade"], "正常", "降級"),
                badge("S2 籌碼", s2["level"] == "常態", "常態", s2["level"]),
                badge("S3 龍頭聯動", not s3["unlinked"], "同步", "不聯動"),
                badge("E5 乖離", not ex["e5"], f"{ex['dev']:.0f}%/{ex['thr']:.0f}%", f"超限{ex['dev']:.0f}%"),
                badge("E7 財報期", not ex["e7"], "非鎖定", "鎖定期"),
            ]))

            # 分析師
            with st.expander("📊 分析師看法"):
                rec = info.get("recommendationKey")
                tgt = info.get("targetMeanPrice")
                if rec:
                    up = f"，目標 {tgt:.0f}（{(tgt-s1['price'])/s1['price']*100:+.0f}%）" if tgt else ""
                    st.write(f"評等：**{rec}**{up}，共 {info.get('numberOfAnalystOpinions','?')} 位")
                else:
                    st.write("無分析師資料")
                st.caption("⚠️ 分析師評等無免費權威源，僅供參考")

            # 資料源
            st.info(f"🔎 價量：{xnote}　|　財報源：{s0.get('source','yfinance')}")
            for n in s0["notes"] + s2["notes"]:
                st.caption(n)
            st.warning("⚠️ E1黑天鵝與催化劑須人工核實。本工具非投資建議。")

# ---- 功能三：找相似 ----
elif mode == "🚀 找相似飆股":
    st.subheader("找相似飆股")
    st.caption("⚠️ 只找『目前型態相似』的標的，非漲幅預測，有倖存者偏誤，僅供研究起點")
    ref = st.text_input("範本股代號", "NVDA").upper().strip()
    top_n = st.slider("找幾檔", 3, 12, 6)
    if st.button("掃描", type="primary"):
        uni = list(dict.fromkeys(SIMILAR_UNIVERSE + [ref]))
        with st.spinner("批次下載中…"):
            data = _bulk(tuple(uni), "1y")
        if data is None or data.empty:
            st.error("下載失敗")
        else:
            feats = {}
            for t in uni:
                try:
                    if t in data.columns.get_level_values(0):
                        f = _features(data[t]["Close"])
                        if f: feats[t] = f
                except Exception:
                    pass
            if ref not in feats:
                st.error(f"範本 {ref} 資料不足")
            else:
                picks = _sim_pick(feats, ref, top_n)
                rows = []
                for t in picks.index:
                    f = feats[t]
                    rows.append({"代號": t, "相似距離": round(float(picks[t]), 2),
                                 "1月%": round(f["r1m"]*100, 1), "3月%": round(f["r3m"]*100, 1),
                                 "6月%": round(f["r6m"]*100, 1), "RSI": round(f["rsi"], 0)})
                rf = feats[ref]
                st.caption(f"範本 {ref}：1月{rf['r1m']*100:.1f}% / 3月{rf['r3m']*100:.1f}% / 6月{rf['r6m']*100:.1f}%")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.warning("⚠️ 這是型態相似清單，不是漲幅預測。請對有興趣的標的再做個股分析。")

# ---- 功能四：批次回測 ----
elif mode == "🧪 批次回測":
    st.subheader("批次回測（驗證選股邏輯）")
    st.caption("跑多個基準日，看平均超額報酬是否穩定為正")
    ref = st.text_input("範本股代號", "NVDA").upper().strip()
    c1, c2 = st.columns(2)
    sd = c1.date_input("起始基準日", pd.Timestamp("2025-06-01"))
    ed = c2.date_input("結束基準日", pd.Timestamp("2026-05-01"))
    every = st.slider("每隔幾天取一基準日", 7, 60, 21)
    fwd = st.slider("往後看幾天報酬", 5, 90, 15)
    if st.button("跑批次回測", type="primary"):
        uni = list(dict.fromkeys(SIMILAR_UNIVERSE + [ref]))
        dl_s = (pd.Timestamp(sd) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
        dl_e = (pd.Timestamp(ed) + pd.Timedelta(days=fwd + 12)).strftime("%Y-%m-%d")
        with st.spinner("下載全期間資料…"):
            data = _safe(lambda: yf.download(uni + ["SPY"], start=dl_s, end=dl_e,
                                             group_by="ticker", auto_adjust=True, progress=False))
        if data is None or data.empty:
            st.error("下載失敗")
        else:
            def close_of(t):
                try:
                    if t in data.columns.get_level_values(0):
                        return data[t]["Close"].dropna()
                except Exception:
                    return None
                return None
            dates = pd.date_range(sd, ed, freq=f"{every}D")
            results = []
            prog = st.progress(0.0)
            for i, dd in enumerate(dates):
                prog.progress((i + 1) / len(dates), f"{dd.date()}")
                as_of = pd.Timestamp(dd)
                feats, fwdr = {}, {}
                for t in uni:
                    c = close_of(t)
                    if c is None: continue
                    hist = c[c.index <= as_of]; fut = c[c.index >= as_of]
                    if len(hist) < 130 or len(fut) < 2: continue
                    f = _features(hist)
                    if not f: continue
                    feats[t] = f
                    entry = float(hist.iloc[-1])
                    win = fut[fut.index <= as_of + pd.Timedelta(days=fwd)]
                    if len(win) >= 2 and entry:
                        fwdr[t] = float(win.iloc[-1] / entry - 1)
                if ref not in feats: continue
                spy = close_of("SPY"); spy_fwd = None
                if spy is not None:
                    sh = spy[spy.index <= as_of]; sf = spy[spy.index >= as_of]
                    win = sf[sf.index <= as_of + pd.Timedelta(days=fwd)]
                    if len(sh) and len(win) >= 2:
                        spy_fwd = float(win.iloc[-1] / float(sh.iloc[-1]) - 1)
                picks = _sim_pick(feats, ref, 6)
                rr = [fwdr[t] for t in picks.index if t in fwdr]
                beat = sum(1 for t in picks.index if t in fwdr and spy_fwd is not None and fwdr[t] > spy_fwd)
                if rr and spy_fwd is not None:
                    avg = float(np.mean(rr))
                    results.append({"as_of": str(dd.date()), "選股%": round(avg*100, 1),
                                    "大盤%": round(spy_fwd*100, 1),
                                    "超額%": round(avg*100 - spy_fwd*100, 1),
                                    "勝": beat, "n": len(rr)})
            prog.empty()
            if not results:
                st.error("無有效結果（可能期間太短）")
            else:
                rdf = pd.DataFrame(results)
                avg_ex = round(float(rdf["超額%"].mean()), 2)
                std_ex = round(float(rdf["超額%"].std()), 2)
                pwr = round((rdf["超額%"] > 0).sum() / len(rdf) * 100, 1)
                tb, tp = int(rdf["勝"].sum()), int(rdf["n"].sum())
                hit = round(tb / tp * 100, 1) if tp else 0
                k1, k2, k3 = st.columns(3)
                k1.metric("平均超額報酬", f"{avg_ex:+.2f}%")
                k2.metric("期間勝率", f"{pwr}%")
                k3.metric("個股勝率", f"{hit}%")
                st.caption(f"超額報酬波動(標準差) {std_ex}%　|　共 {len(rdf)} 個基準日")
                if avg_ex > 0 and pwr >= 55:
                    st.success("✅ 略有正超額且勝率過半，可能有一點edge，仍需更長期間+計入成本驗證")
                elif avg_ex <= 0:
                    st.error("❌ 平均超額為負：不如直接買大盤，建議調整選股邏輯")
                else:
                    st.warning("⚠️ 結果模稜兩可，多半是噪音，不足以當有效策略")
                st.line_chart(rdf.set_index("as_of")[["選股%", "大盤%"]])
                st.dataframe(rdf, use_container_width=True, hide_index=True)
                st.caption("未計交易成本/滑價/稅；股票池偏大型成長股有倖存者偏誤；過去不保證未來")

st.divider()
st.caption("⚠️ 本工具僅供個人研究與學習，所有輸出非投資建議，買賣請自負盈虧。")

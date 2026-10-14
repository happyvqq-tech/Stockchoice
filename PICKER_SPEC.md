# 短線選股系統 開發規格（PICKER_SPEC）

> 本文件是給 Claude Code 的開工規格。放在 repo 根目錄，Claude Code 每次開工前先讀本文件與 CLAUDE.md。
> 版本：v1.0（2026-10-13）

---

## 0. 給 Claude Code 的工作規則（必讀）

1. **先盤點再動手**：Phase 0 只讀不寫，先回報現有 Stockchoice repo 結構與整合方案，經使用者同意後才開始寫程式。
2. **一次只做一個 Phase**：完成該 Phase 的「驗收條件」後停下來，回報結果與下一步，等使用者確認。
3. **不擅自擴大範圍**：本文件「非目標」與「禁止事項」中的項目不做。遇到規格沒寫清楚的地方，先提出選項讓使用者決定。
4. **每完成一個 Phase**：更新本文件第 15 節的進度勾選，並在 `CHANGELOG.md` 記一行。
5. **使用者沒有程式背景**：回報時用白話說明做了什麼、怎麼執行、怎麼確認結果正確；指令要能直接複製貼上（Windows 環境）。
6. **把本文件的「核心規則」（第 13 節）摘要寫進專案 CLAUDE.md**，確保之後每次對話都遵守。

---

## 1. 目標與非目標

### 目標
- 每個交易日收盤後，對指定市場輸出 **前 10 名短線候選股**（預設持有 10 個交易日），附分數與前 3 個主要理由。
- 用 **walk-forward（滾動前進驗證）** 評估選股能力，每季自動重訓並產出複盤報告。
- 用歷史資料先驗證、優化流程，再進入紙上交易，最後才考慮實盤。

### 成功標準
- 扣除交易成本後，前 10 名的平均超額報酬（相對基準指數）> 0。
- 在多數 walk-forward 測試期（≥ 60% 的輪次）為正。
- 同時勝過兩個基準：「隨機挑 10 檔」與「純 20 日動能前 10 名」。

### 非目標（不做）
- 自動下單、券商 API 串接
- 盤中即時訊號
- 深度學習模型（LSTM / Transformer 等）
- 保證獲利或「明牌」式輸出

---

## 2. 關鍵設計決策

| 項目 | 決定 | 備註 |
|---|---|---|
| 首發市場 | **台股（TW）** | 可在 config 改 `primary_market`；架構需市場無關，之後用 adapter 擴充 US、CN |
| 持有期 | 10 個交易日 | config `hold_days` |
| 每期選股數 | 10 檔，等權重 | config `top_k` |
| 模型 | 先「規則打分」當基準，再上 LightGBM | 不做大量調參 |
| 新聞 / 市場資訊 | LLM 只做「排雷」（剔除重大利空），**不進回測** | 避免 LLM 訓練資料造成未來資訊洩漏 |
| 資料儲存 | **Supabase Storage 私人 bucket `picker-data`**（Parquet，歷史與訓練用）＋ Supabase 資料庫（每日名單、紙上交易，供 Streamlit 顯示）。共用既有 Supabase 專案，資料表放獨立 `picker` schema | v1.1 改為雲端；本機 `data_store/` 僅作快取，gitignore。repo 為公開，資料與金鑰一律不進 git |
| 介面 | 沿用 Stockchoice 的 Streamlit，新增頁面 | 整合方式在 Phase 0 提案 |

---

## 3. 技術棧

- Python 3.11+
- 資料：`pandas`、`pyarrow`（Parquet）
- 資料來源：`FinMind`（台股）、`yfinance`（美股）、`akshare`（A股）
- 模型：`lightgbm`、`scikit-learn`（評估指標）
- 儲存：`supabase`（supabase-py）
- 介面：`streamlit`、`plotly`
- LLM：`anthropic` SDK（Phase 4 新聞排雷）
- 測試：`pytest`
- 金鑰：`.env`（本機）與 Streamlit secrets（雲端），**絕不進 git**

---

## 4. 目錄結構（建議，Phase 0 可依現有 repo 調整）

```
picker/
  config.yaml
  config.py              # 讀取設定
  data/
    adapters/
      base.py            # 統一介面：get_daily_bars / get_universe / get_calendar / get_events
      tw.py              # Phase 1
      us.py              # Phase 5
      cn.py              # Phase 5
    store.py             # Parquet 讀寫、增量更新
    quality.py           # 資料品質檢查
    universe.py          # 動態股票池
  features/
    technical.py
    chips_tw.py          # 台股籌碼、月營收
    regime.py            # 大盤環境
    build.py             # 組合特徵 + 截面排名標準化
  labels.py
  models/
    rule_score.py
    lgbm.py
  backtest/
    walkforward.py
    portfolio.py         # 選股、成本、漲跌停處理
    metrics.py
    baselines.py
  live/
    daily_run.py
    news_filter.py
    paper_trade.py
  reports/
    retro.py             # 季度複盤報告
experiments/
  EXPERIMENT_LOG.md
  runs/                  # 每次實驗一個資料夾
data_store/              # gitignore，本機快取（正本在 Supabase Storage）
tests/
```

---

## 5. 資料規格

### 5.1 統一日線格式
`date, market, symbol, name, open, high, low, close, adj_close, volume, turnover`

- **報酬計算用還原價（adj_close）**；**漲跌停判斷用原始價**。若來源沒有還原價，用除權息資料自行還原。
- 只保留普通股（台股排除 ETF、ETN、權證、特別股；能辨識時排除全額交割股）。

### 5.2 股票池（避免未來資訊）
- **每月第一個交易日**，以「過去 60 個交易日平均成交金額」排名，取前 N 名（台股預設 300，上市 + 上櫃）。
- 不可用「今天的市值排名」回推歷史，否則會引入未來資訊與存活者偏差。
- 已下市股票：資料來源有提供就納入；沒有的話，在報告中註明「結果可能偏樂觀」。

### 5.3 事件資料必須用「可取得日」
| 資料 | 可取得日規則 |
|---|---|
| 三大法人買賣超 | 當日收盤後公布 → t 日資料可用於 t+1 進場 |
| 融資融券 | 同上 |
| 月營收 | 用實際公布日；若無公布日，保守設為次月 11 日 |
| 財報 | 用實際公布日 |

### 5.4 基準指數
- TW：0050；US：SPY；CN：510300（滬深 300 ETF）

### 5.5 資料品質檢查（quality.py）
- 缺漏交易日、重複列、價格為 0 或負值
- 單日漲跌超過合理範圍（台股 > 10%、未除權息）列出待查
- 輸出 `data_quality_report.md`

---

## 6. 特徵清單 v1（約 20 個）

所有特徵只能用 **t 日收盤（含）以前** 的資料。每個交易日內做 **截面排名百分位標準化**（rank pct），讓不同時期可比較。

| 類別 | 特徵 | 定義 |
|---|---|---|
| 動能 | `ret_5` / `ret_20` / `ret_60` | 過去 N 日報酬 |
| 動能 | `rel_strength_20` | `ret_20` 減基準 `ret_20` |
| 動能 | `rsi_14` | 14 日 RSI |
| 趨勢 | `close_to_ma20` / `close_to_ma60` | 收盤 / 均線 − 1 |
| 趨勢 | `ma20_slope` | MA20 的 5 日變化率 |
| 突破 | `close_to_high20` | 收盤 / 過去 20 日最高價 |
| 突破 | `dist_52w_high` | 收盤距 52 週高點的距離 |
| 量能 | `vol_ratio_20` | 當日量 / 20 日均量 |
| 波動 | `atr14_pct` | ATR(14) / 收盤 |
| 波動 | `bb_width_pctile` | 布林寬度在過去 120 日的百分位（低 = 壓縮） |
| 流動性 | `turnover_60` | 60 日平均成交金額 |
| 籌碼（TW） | `foreign_net_5d` | 外資 5 日買賣超 / 5 日成交量 |
| 籌碼（TW） | `trust_net_5d` | 投信 5 日買賣超 / 5 日成交量 |
| 籌碼（TW） | `trust_consec_days` | 投信連續買超天數 |
| 籌碼（TW） | `margin_chg_5d` | 融資餘額 5 日變化率 |
| 基本面（TW） | `rev_yoy` | 最新已公布月營收 YoY |
| 基本面（TW） | `rev_yoy_new_high` | 月營收是否創 12 個月新高（0/1） |
| 大盤 | `bench_above_ma60` | 基準是否在 60 日線上（0/1） |
| 大盤 | `bench_ret_20` | 基準 20 日報酬 |

每個特徵要有中文顯示名稱對照表（給「選股理由」用）。

---

## 7. 標籤

- 訊號日 t（收盤後產生）→ **t+1 開盤進場** → **t+hold_days 收盤出場**
- `fwd_ret` = 出場價 / 進場價 − 1 − 交易成本
- `fwd_excess` = `fwd_ret` − 基準同期報酬
- **訓練標籤** `label` = 1，若 `fwd_excess` 在當日股票池中排名前 20%（config `label_top_pct`），否則 0
- `fwd_excess` 原始值保留，用於評估

---

## 8. 模型

### 8.1 RuleScore（基準線）
- 選定特徵的截面排名百分位，依方向（越大越好 / 越小越好）等權平均。
- 初始採用：`rel_strength_20`、`close_to_ma20`、`ma20_slope`、`vol_ratio_20`、`close_to_high20`、`trust_net_5d`、`foreign_net_5d`、`rev_yoy`。

### 8.2 LightGBM
- 二元分類，輸出機率當分數，排序取前 `top_k`。
- 固定參數起步，不做大規模調參：
  `num_leaves=31, learning_rate=0.05, n_estimators=最多 500 + early stopping, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, min_child_samples=100, seed=42`
- Early stopping 用訓練窗口內「最後 3 個月」當驗證集（不可用測試期）。
- **選股理由**：用 LightGBM 內建 `pred_contrib=True` 取每檔貢獻最大的前 3 個特徵，轉成中文說明。

### 8.3 對照基準（baselines.py）
- 隨機挑 `top_k` 檔（100 次取平均）
- 純 `ret_20` 動能前 `top_k`
- 基準指數買進持有

---

## 9. 回測規格

### 9.1 兩種評估都要做
1. **訊號品質**：每個交易日的前 10 名，計算其平均 `fwd_excess`（窗口會重疊，用來衡量選股能力）。
2. **資金曲線**：每 `hold_days` 換股一次（不重疊），等權重，計算淨值、回撤。

### 9.2 交易成本（config 可調，預設偏保守）
| 市場 | 手續費（單邊） | 稅 | 滑價（單邊） |
|---|---|---|---|
| TW | 0.1425% × 折扣（預設 1.0） | 賣出證交稅 0.3% | 0.1% |
| US | 0 | — | 0.05% |
| CN | 0.025% | 賣出印花稅 0.05% | 0.1% |

### 9.3 市場規則
- **漲跌停**：TW ±10%；CN 主板 ±10%、創業板 / 科創板 ±20%、ST ±5%（排除北交所）。
  - 進場日開盤即漲停 → 視為買不到，該名額空著（不遞補）。
  - 出場日跌停 → 順延至下一個可賣出日。
- **停牌**：停牌中不可進出，順延。
- **CN T+1**：持有期 ≥ 1 日，已自然滿足，但程式需有檢查。

---

## 10. Walk-forward 設定

```
訓練窗口：24 個月（滾動）
測試窗口：3 個月
每次前進：3 個月
Purge：訓練期最後 hold_days 天的樣本移除（避免標籤跨入測試期）
資料起始：2018-01-01
期末考（holdout）：2026-04-01 之後 → 開發期間禁止使用
```

- 程式必須有保護：任何訓練或評估若觸及 `holdout_start` 之後的資料，直接報錯；只有加上 `--final-exam` 參數才允許，且執行結果要記錄在實驗日誌。
- 每一輪輸出指標，最後彙總。

---

## 11. 評估指標與採用規則

### 11.1 指標（每輪 + 彙總）
- 前 10 名平均超額報酬（每期）
- 勝率（`fwd_excess > 0` 的比例）
- 盈虧比（平均賺 / 平均賠）
- 每筆期望值 = 勝率 × 平均賺 − (1 − 勝率) × 平均賠
- IC（分數與 `fwd_excess` 的 Spearman 相關，每日計算後取平均）
- 年化報酬、最大回撤、Sharpe、換手率
- 正報酬輪次比例
- 與三個基準的比較表

### 11.2 新版本採用規則（三項都要成立）
1. 彙總期望值比現行版本高
2. 在 ≥ 60% 的 walk-forward 輪次表現較好
3. 同時勝過「隨機」與「純動能」兩個基準

---

## 12. 實驗紀錄規範

- **每次實驗只改一個地方**（一個特徵、一個參數、一條規則）。
- 每次實驗建立 `experiments/runs/YYYYMMDD_簡短名稱/`，內含：
  - `config.yaml`（當次完整設定）
  - `metrics.json`
  - `report.md`（含圖表與白話結論）
- 在 `experiments/EXPERIMENT_LOG.md` 追加一行：

| 日期 | 版本 | 改了什麼 | 結果摘要 | 決定（採用 / 不採用） |
|---|---|---|---|---|

---

## 13. 核心規則（需寫入 CLAUDE.md）

1. 不使用未來資料：特徵只用 t 日以前；事件資料用可取得日；t+1 開盤進場。
2. 不可用 holdout 期間調參或做決策。
3. 回測中不呼叫 LLM。LLM 只在每日實盤流程中做排雷。
4. 資料正本放 Supabase Storage（私人 bucket）；`data_store/` 本機快取不進 git、不放 OneDrive 同步路徑；API 金鑰只放 GitHub Secrets / Streamlit secrets / `.env`（repo 為公開，絕不進 git）。
5. 不做自動下單。
6. 每次實驗只改一個地方並記錄。
7. 所有輸出註明「研究用途，非投資建議」。

### 必要測試（tests/）
- **截斷測試（防未來資訊）**：用「截至 t 日的資料」與「完整資料」分別計算特徵，t 日的特徵值必須完全相同。
- 標籤測試：確認 `fwd_ret` 用的是 t+1 開盤與 t+hold_days 收盤。
- 成本測試：已知案例的成本計算正確。
- Holdout 保護測試：未加 `--final-exam` 觸及 holdout 必須報錯。

---

## 14. 每日流程與 Supabase（Phase 4）

### 14.1 每日流程（daily_run.py）
1. 收盤後增量更新資料
2. 計算特徵 → 模型打分 → 取前 15 名（預留排雷剔除的空間）
3. 新聞排雷（news_filter.py）→ 剔除後取前 10 名
4. 寫入 Supabase `picks`
5. 紙上交易：自動建立進場紀錄（隔日開盤價）、到期自動結算

排程方式（Windows 工作排程器 vs GitHub Actions）由 Claude Code 評估資料來源的速率限制後提案。

### 14.2 新聞排雷規格
- 輸入：股票代號、名稱、近 3 日新聞標題與摘要（新聞來源由 Claude Code 調查後提案）
- 模型：`claude-sonnet-5-5`（config 可改）
- 輸出 JSON：`{symbol, red_flag: bool, severity: low|mid|high, reason}`
- 只能剔除，**不能加分或新增標的**
- `severity=high` 才剔除；每次判斷都要寫入紀錄，供日後評估排雷是否有效

### 14.3 Supabase 資料表
- `picks`：date, market, symbol, name, rank, score, top_reasons (jsonb), news_flag, news_reason, model_version
- `paper_trades`：id, pick_date, market, symbol, entry_date, entry_price, exit_date, exit_price, ret, bench_ret, excess_ret, status
- `model_versions`：version, trained_at, train_start, train_end, metrics (jsonb)

### 14.4 Streamlit 新頁面
- 今日名單（分數、理由、排雷結果）
- 紙上交易績效（累積超額報酬、勝率，與回測預期對照）
- 實驗紀錄與 walk-forward 結果瀏覽

---

## 15. 開發階段與驗收條件

### Phase 0：盤點與整合提案（只讀不寫）
- [x] 閱讀 Stockchoice 現有結構、CLAUDE.md、既有回測模組
- [x] 提出整合方案（同 repo 新資料夾 or 新 repo），說明理由
- [x] 評估既有程式哪些可重用
- **驗收**：使用者同意整合方案

### Phase 1：資料層（台股）
- [x] `adapters/base.py` 統一介面 + `adapters/tw.py`
- [x] 日線、三大法人、融資融券、月營收（含公布日：保守設為次月 11 日）、交易日曆；另加除權息、分割／面額變更、減資（用於還原價）
- [x] 動態股票池（universe.py）
- [x] Parquet 儲存與增量更新（資料正本在 Supabase Storage；每日增量改走證交所／櫃買官方檔，延到 Phase 4）
- [x] 資料品質報告（零價格列排除、還原價驗證、分割／減資偵測）
- **驗收**：✅ 2018-01 至 2026-10-12 資料齊全（2,089 檔、約 399 萬列）；✅ 品質報告通過；✅ 抽樣 3 檔 × 3 日與證交所／櫃買官方資料比對：開高低收 9/9 完全一致，成交量上市股 6/6 一致、上櫃股 3 筆差 0.4–0.5%（官方為整張口徑、我們含零股，推測，不影響價格與報酬）。

### Phase 2：特徵、標籤、規則打分、回測引擎
- [ ] 特徵 v1（第 6 節）＋截面排名標準化
- [ ] 標籤（第 7 節）
- [ ] RuleScore（8.1）與三個基準（8.3）
- [ ] 回測引擎（第 9 節）＋ walk-forward（第 10 節）
- [ ] 指標與報告（第 11 節）
- [ ] 必要測試全部通過（第 13 節）
- **驗收**：產出 RuleScore 的 walk-forward 報告，含與基準的比較表與白話結論

### Phase 3：LightGBM
- [ ] lgbm.py 接入 walk-forward
- [ ] 特徵重要性（每輪）與選股理由（pred_contrib）
- [ ] 與 RuleScore 比較，依 11.2 規則判定是否採用
- **驗收**：比較報告 + 實驗日誌記錄

### Phase 4：每日流程與紙上交易
- [ ] daily_run.py、news_filter.py、paper_trade.py
- [ ] Supabase 資料表與 Streamlit 頁面
- [ ] 排程方案提案並設定
- **驗收**：連續 5 個交易日自動執行成功，無需人工介入

### Phase 5：季度複盤與擴充市場
- [ ] retro.py：每季重訓、紙上交易 vs 回測差異、特徵重要性變化、LLM 撰寫白話複盤摘要
- [ ] 美股 adapter（us.py）、A股 adapter（cn.py），沿用同一套流程
- **驗收**：產出第一份季度複盤報告；第二個市場跑完 Phase 2 等級的 walk-forward 報告

---

## 16. config.yaml 範本

```yaml
primary_market: TW          # TW / US / CN
data_start: 2018-01-01
holdout_start: 2026-04-01   # 期末考，開發期間禁止使用

universe:
  TW:
    top_n_by_turnover: 300
    lookback_days: 60
    rebalance: monthly
    include_otc: true       # 含上櫃

hold_days: 10
top_k: 10
label_top_pct: 0.2

walkforward:
  train_months: 24
  test_months: 3
  step_months: 3
  purge_days: 10            # = hold_days

benchmark:
  TW: "0050"
  US: "SPY"
  CN: "510300"

costs:
  TW: {commission: 0.001425, commission_discount: 1.0, sell_tax: 0.003, slippage: 0.001}
  US: {commission: 0.0, sell_tax: 0.0, slippage: 0.0005}
  CN: {commission: 0.00025, sell_tax: 0.0005, slippage: 0.001}

lgbm:
  num_leaves: 31
  learning_rate: 0.05
  n_estimators: 500
  early_stopping_rounds: 50
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 1
  min_child_samples: 100
  seed: 42

llm:
  model: claude-sonnet-5-5
  news_lookback_days: 3
  veto_severity: high

paths:
  data_store: "data_store"             # 本機快取；正本在 Supabase Storage bucket picker-data
```

---

## 17. 待使用者確認（有預設值，可直接開工）

| 項目 | 預設 | 說明 |
|---|---|---|
| 首發市場 | 台股 | 改 `primary_market` |
| 持有天數 | 10 個交易日 | 之後可做實驗比較 5 / 20 日 |
| 每期檔數 | 10 | |
| 台股手續費折扣 | 1.0（無折扣） | 填入實際券商折扣 |
| 資料位置 | Supabase Storage（共用既有專案） | v1.1 起不再放本機 |
| 新聞來源 | 待 Phase 4 提案 | |

---


---

## 18. v1.1 修訂（2026-10-13，使用者已同意全部採用）

整體檢視後新增以下設計，與前文衝突時以本節為準。

### 18.1 降低換手率與成本（Phase 2 回測必做）
- 台股每輪（買＋賣）成本約 0.79%，每 10 日全換一年約 20%，故換手率是設計重點。
- **緩衝區規則**：已持有者只要仍在前 `buffer_rank`（預設 30）名就續抱，跌出才賣；新買入仍取前 `top_k`。
- **持有天數 10 日 vs 20 日**提前到 Phase 2 比較（各為一次實驗，一次只改一處）。
- 手續費折扣暫用預設 1.0（使用者暫不提供）。

### 18.2 風險控制
- **大盤紅綠燈控制持股**：基準在 60 日線上＝全數持股；跌破＝只持 `top_k_bear`（預設 5）檔，其餘留現金。
- **同產業上限** `max_per_industry`（預設 3 檔）。
- **排除處置股、注意股、全額交割股**（證交所/櫃買每日公布名單，Phase 1 蒐集歷史名單來源，Phase 2 起套用；歷史取不到時須在報告註明）。

### 18.3 資料來源分工
- 歷史（2018 起）：FinMind 免費版（600 次/小時，單檔一次請求抓完整區間）。還原價用除權息資料自行計算（`TaiwanStockPriceAdj` 為付費）。月營收公布日依 5.3 保守規則（次月 11 日）。
- 每日增量（Phase 4）：改用證交所/櫃買每日全市場檔，不佔 FinMind 額度。
- 兩階段下載：先全市場日線（排名用）→ 只針對曾進過股票池的股票抓法人、融資券、月營收、除權息。
- 下市股若免費版取不到，報告須註明「結果可能偏樂觀」。

### 18.4 介面（取代 14.4 的呈現方式）
- 首頁為**今日行動清單**（手機優先）：大盤燈號與建議持股、明天要買（附「開盤漲停不追」）、明天要賣（附原因）、續抱、被剔除者與原因。
- 第二頁「這套系統可信嗎」：回測 vs 紙上交易並排、與 0050 累積報酬比較。
- 分數不得標示為「上漲機率」。圖表用 plotly。Streamlit 頁面設為私人（僅限本人 email）。

### 18.5 路線調整
- Phase 5 的美股／A股延後到台股驗證成功之後。
- 紙上交易至少 3 個月才考慮實盤。
- config 新增：`buffer_rank: 30`、`top_k_bear: 5`、`max_per_industry: 3`。


### 18.6 Phase 1 已知資料限制（2026-10-14）
1. 價格為 0 的列（約 5.8 萬列、涉及 1,096 檔）是「當天無資料」，一律排除，不得當成價格 0。
2. 79 檔查無日線資料；FinMind 下市名單只到 2023-04，之後下市的股票可能完全缺漏 → **回測結果可能偏樂觀（存活者偏差）**，所有回測報告必須註明。
3. 少數除權息事件還原仍有誤差（品質報告「還原後反而更離譜」約 36 個，多數是除息日剛好漲停；真正異常如 2429 於 2024-07-02）。
4. 新上市／興櫃轉上櫃初期沒有漲跌幅限制，出現大於 ±10% 的真實價格跳動，不是錯誤；Phase 2 判斷漲跌停須考慮。
5. 上櫃股成交量與櫃買官方口徑差約 0.4–0.5%（零股），不影響價格與成交金額排名。
6. 每日增量更新尚未改用證交所／櫃買官方檔（Phase 4）；歷史資料全由 FinMind 免費版取得（每小時 600 次）。

*研究用途，非投資建議。*

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Two things live (or will live) in this repo:

1. **`app.py`** — the existing single-file Streamlit app "美股策略分析師" (US stocks, yfinance). Working, deployed, no tests.
2. **The short-term stock picker** — specified in `PICKER_SPEC.md`, not yet implemented. Read `PICKER_SPEC.md` before any picker work; its §0 workflow rules are binding (see below).

The user has no programming background and works on **Windows**. Explanations should be plain-language (Traditional Chinese), and commands given to the user must be copy-pasteable in a Windows terminal.

## Commands

```bash
pip install -r requirements.txt
streamlit run app.py          # local run; devcontainer auto-starts this on port 8501
```

There is no lint config and no test suite yet. Per the spec, picker tests will live in `tests/` and run with `pytest` (single test: `pytest tests/test_x.py::test_name`).

## app.py architecture

One file, top to bottom: config constants → cached data helpers → diagnostic modules → Streamlit UI dispatched by a sidebar `st.radio` (`mode`) into four branches (篩選 / 個股分析 / 找相似飆股 / 批次回測).

- Data fetchers are wrapped in `@st.cache_data(ttl=600)` (EDGAR/CIK: 86400). Failures are swallowed via `_safe(fn, default)` and surface as `None`, so every caller null-checks.
- 個股分析 runs a pipeline of diagnostics, each returning a dict: `assess_market` (M1, SPY/QQQ vs 20MA), `edgar_fundamentals` or fallback `_diag_s0_yf` (S0), `_diag_s1` (20MA break), `_diag_s2` (insiders/short ratio), `_diag_s3` (sector-ETF linkage), `_diag_ex` (E3/E5/E6/E7 extension & earnings-window rules). `_decide()` combines them into action/position text; **rule order in `_decide` matters** — later rules override earlier ones.
- Price is cross-checked against Stooq (`cross_check`); SEC EDGAR requests need the `SEC_UA` User-Agent header.
- 找相似 / 批次回測 share `_features()` + `_sim_pick()` (weighted z-score Euclidean distance over a hardcoded `SIMILAR_UNIVERSE`). The 批次回測 is known to be lookahead/survivorship-biased (enters at signal-day close, calendar-day horizon, no costs) — do not reuse it for the picker.
- `requirements.txt` is what the deployed Streamlit app installs; keep heavy picker-only deps (lightgbm, FinMind, akshare, …) out of it.

## Picker work rules (from PICKER_SPEC.md §0)

- Work one Phase at a time (§15). Phase 0 is read-only. After finishing a Phase's acceptance criteria, stop and report; wait for user confirmation before the next Phase.
- Don't expand scope beyond the spec's 非目標; when the spec is unclear, present options for the user to choose.
- On completing each Phase: tick its boxes in `PICKER_SPEC.md` §15 and add one line to `CHANGELOG.md`.

## Picker core rules (PICKER_SPEC.md §13 — always enforce)

1. No future data: features use only data up to and including day t; event data (法人, 融資券, 月營收, 財報) keyed by its **availability date**; entry at **t+1 open**, exit at t+hold_days close.
2. Never use the holdout period (`holdout_start`, default 2026-04-01) for tuning or decisions. Code must raise if training/evaluation touches it unless run with `--final-exam`, and such runs are logged in the experiment log.
3. No LLM calls inside backtests. The LLM is used only in the live daily flow, and only to veto picks (never to add or upweight).
4. The repo is **public**: data and keys never go in git. Data of record lives in the private Supabase Storage bucket `picker-data` (`data_store/` is only a gitignored local cache); keys live in GitHub Secrets (`FINMIND_TOKEN`, `SUPABASE_URL`, `SUPABASE_SECRET_KEY`), Streamlit secrets or `.env`. `SUPABASE_URL` may end in `/rest/v1` — strip it.
5. No automated order placement.
6. Each experiment changes exactly one thing and is recorded in `experiments/EXPERIMENT_LOG.md` plus `experiments/runs/YYYYMMDD_name/`.
7. Every output carries 「研究用途，非投資建議」.

Required tests: truncation test (features computed on data cut at t equal features from full data), label test (t+1 open → t+hold_days close), cost test on known cases, holdout-guard test.

## Spec v1.1 revisions (PICKER_SPEC.md §18 — takes precedence over earlier sections)

Turnover/cost is the main threat (~0.79% per round trip in TW): backtests must include the buffer rule (hold while rank ≤ 30), compare 10- vs 20-day holds, cap 3 names per industry, cut to 5 names when the benchmark is below its 60-day MA, and exclude disposition/attention/full-delivery stocks. History comes from FinMind free tier (600 req/hour; whole-market and adjusted-price datasets are paid, so adj_close is computed from dividend data); daily increments in Phase 4 use TWSE/TPEx files. UI home page is an action list (buy/sell/hold/vetoed), never "probability of rising". US/CN adapters are postponed until TW is validated; paper-trade ≥3 months before any real money.

Returns use adjusted prices (`adj_close`); limit-up/limit-down checks use raw prices.

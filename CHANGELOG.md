# CHANGELOG

- 2026-10-13 Phase 0 完成：盤點現有 repo，採「同 repo 新增 picker/ 資料夾」方案；資料改放 Supabase Storage，排程用 GitHub Actions。
- 2026-10-13 規格更新至 v1.1（緩衝區規則、大盤控制持股、產業上限、行動清單首頁、資料改放 Supabase）。
- 2026-10-14 Phase 1 完成：台股 2018 起歷史資料（日線、法人、融資券、月營收、除權息、分割、減資）存於 Supabase；動態股票池、還原價、品質報告；抽樣 9 筆與證交所／櫃買官方資料比對（價格全一致）。已知限制見 PICKER_SPEC §18.6。
- 2026-10-14 Phase 2 完成：特徵 v1、標籤、RuleScore 與基準、回測引擎（成本／漲跌停／緩衝／紅綠燈／同業上限）、walk-forward 與期末考保護、31 個測試；首份報告顯示 RuleScore 打不贏 0050。

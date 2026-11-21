# E44｜第三次反彈的保存點預測

## E43 的四次短暫越線

種子 2827、2867、2873、2875 在 \(t=2.10\) 時的 loss 增量仍低於 \(10^{-4}\)。到第四次接觸（約 2.131）升至門檻的 100.9%–103.1%。四例的方向導數 \(\nabla f\cdot v\) 在接觸前皆為正，反彈後皆為負，到下一個保存點 2.15 已回到門檻內。接觸的瞬間改變速度，位置及 loss 連續。由於上升和回落都發生在兩個保存點之間，保存點口徑沒有記到越線。讀封存資料，新增查詢 0。[逐例診斷](../output/collision_third_reflection_saved_window/e43_contact_only_diagnosis.json)

## 80 個新種子、480 組同初態配對

固定 2920–2999 各六種初始動能。兩政策共用 E40 到第三次反彈的狀態。原規則檢查到下一次接觸，包含接觸的 loss。新版保留接觸前的保存點預測，若都安全則算到接觸後第一個保存點。只用保存點的預測觸發轉移，追到 \(t=5.00\)。[固定協議](superpowers/specs/2026-11-12-third-reflection-saved-window-design.md)

| 指標 | 原規則 | 保存點規則 |
|---|---:|---:|
| 好粒子途中曾明顯偏離初始解的次數，至 5.00 | 1/480 | 1/480 |
| 防止 E40 保存點偏離 | 14 | 14 |
| 納入精確接觸，至 5.00 曾越線 | 2/480 | 4/480 |
| 第三次反彈轉移 | 16 | 14 |
| 保存點口徑下不必要轉移 | 2/16 | 0/14 |
| 第三次反彈預測查詢 | 478 | 1,406 |
| 政策總查詢 | 688,270 | 689,198 |
| 5.00 當下在門檻內 | 480/480 | 480/480 |

E40 共有 15 組在保存點越線。新版保留原規則全部 14 次保存點預防，無新增或提早的保存點越線，通過預定篩選。少做的兩次轉移在種子 2947、2957，只於精確接觸約 2.131 越線，均在 2.15 恢復，至 5.00 無再越線。若納入精確接觸，原規則防止 16 次，新版防止 14 次，兩者不必要轉移均為零。[正式指標](../output/collision_third_reflection_saved_window/third_reflection/saved_window_results.json) · [逐組配對](../output/collision_third_reflection_saved_window/third_reflection/paired_outcomes.json)

464 組需要跨接觸預測，各增加一個梯度和一個方向路徑查詢，全部完成。其餘 14 組在接觸前的保存點即有警報，一次查詢即可。兩政策的後續查詢淨差為零，因此新版比原規則多 928 次查詢，政策總查詢增加約 0.135%。實驗共用兩政策的 16 段第三次反彈轉移續跑，另實付 16,054 次，連預測共 17,460 次。[實驗實付](../output/collision_third_reflection_saved_window/third_reflection/execution_counts.json)

## 已越線者的恢復

兩政策均略過第三次預測並保留種子 2944、2977。2944 在第二次精確接觸越線，2.00 前的 0.95 已恢復，之後維持。2977 也在 0.95 初次恢復，1.45 再越線，第三次反彈時仍越線，於 1.70 再恢復。兩組在第三次反彈之後都沒有再越線，到 5.00 都在門檻內。[恢復逐例](../output/collision_third_reflection_saved_window/third_reflection/recovery_timelines.json)

## 執行與範圍

正式實付 1,010,855 次，含預檢 16,596 次，共 1,027,451 次。兩批解析另 308,199 值、14,435 梯度。正式無求解失敗，接觸後保存點的 loss 預測差最大 2.08e−17，轉移總動能與線動量分量殘差最大 4.44e−16／2.50e−16。290 項測試全過。此輪在同一合成雙井及精確方向路徑介面上驗證，量化的是少兩次短暫接觸介入和多 928 次查詢的取捨。[正式稽核](../output/collision_third_reflection_saved_window/third_reflection/audit.json) · [物理帳本](../output/collision_third_reflection_saved_window/third_reflection/physical_ledgers.json)

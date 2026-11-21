# E40｜新種子的第二次反彈受限轉移

用 80 個新種子（2720–2799），各六種初始動能，共 480 組相同初態。兩政策沿用 E37 前段。第二次好粒子反彈時，如果已有保存點或精確接觸的 loss 高出初始值超過 \(10^{-4}\)，直接保留動量。其餘狀態用一次方向路徑查詢預測到下一次接觸或 \(t=2.00\)。只有預測越線才按 E39 的守恆規則降低好粒子的水平動能。[固定協議](superpowers/specs/2026-11-12-second-reflection-capped-policy-design.md)

| 指標 | E37 | 第二次反彈有條件受限轉移 |
|---|---:|---:|
| 好粒子途中曾明顯偏離初始解的次數，至 \(t=1.00\) | 0/480 | 0/480 |
| 好粒子途中曾明顯偏離初始解的次數，至 \(t=2.00\) | 20/480 | 0/480 |
| 含後續精確接觸的偏離，至 \(t=2.00\) | 22/480 | 2/480 |
| \(t=2.00\) 當下在門檻內 | 480/480 | 480/480 |
| 政策預測查詢 | 1,213 | 1,691 |
| 政策總查詢（含分岔重驗） | 270,807 | 271,365 |
| 政策轉移次數 | 359 | 379 |

新增的 20 次受限轉移全數防止偏離。20 個警報都在初始動能 \(8\times10^{-5}\) 的層，距原 E37 首次越線的提前量中位數為 0.288。兩個偏離口徑均無配對新增或提早越線。478 個可預警狀態加查，20 個觸發轉移、458 個維持。相對 E37 到 \(t=2.00\) 的偏離，及時警報漏報 0/20，不必要新增轉移 0/20。全批增加 478 次預測查詢，20 個新分支各多付 4 次邊界重驗，總查詢淨增 558 次（約 0.21%）。通過預定篩選。[配對結果](../output/collision_second_reflection_capped_policy/second_reflection/paired_outcomes.json) · [正式指標](../output/collision_second_reflection_capped_policy/second_reflection/results.json)

## 已偏離與恢復

種子 2742、2768 的好粒子在第二次精確接觸（約 \(t=0.937\)）當下首次越線。兩者跳過預測和轉移，均在下一保存點 \(t=0.95\) 回到門檻內，到 \(t=2.00\) 均無再越線。這兩條也解釋了保存點口徑與後續精確接觸口徑的差異。另有 20 條 E37 在第二次反彈之後才越線，保留動量下均在 \(t=1.60–1.95\) 恢復，到 \(t=2.00\) 均無再越線。新增規則防止了這 20 次。這 20 組的最終 loss 增量中位數由 \(9.16\times10^{-5}\) 降到 \(4.20\times10^{-5}\)，全 480 組的中位數則同為 \(1.43\times10^{-6}\)，歷史最佳 loss 中位數同為 −0.05060950。[已偏離逐組](../output/collision_second_reflection_capped_policy/second_reflection/already_deviated_cases.json)

## 執行與核對

主批共用 711 前綴、711 首接觸、1,070 段 E37 比較樹續跑，再新增 20 段受限續跑。正式實付 572,185 次，含設定 1,920 次、預測增量 478 次和新分支續跑 6,050 次。加六組工程預檢 10,939 次與一個舊陽性工程探針 309 次，共 583,433 次。這些含共用實驗與診斷分支，單政策計費見上表。解析核對另 176,469 次值、7,113 次梯度，無求解失敗。20 次介入均可把好粒子的水平速度降到零，總動能及線動量分量最大殘差分別為 \(4.44\times10^{-16}\)、\(2.22\times10^{-16}\)。290 項測試全過。此為同一合成雙井的新種子驗證，依賴精確路徑查詢，其他地形與探索者的搜尋成效仍未測量。[基礎實付](../output/collision_second_reflection_capped_policy/execution_counts.json) · [新增實付](../output/collision_second_reflection_capped_policy/second_reflection/execution_counts.json) · [物理帳本](../output/collision_second_reflection_capped_policy/second_reflection/physical_ledgers.json) · [工程探針](../output/collision_second_reflection_capped_policy_preflight/engineering_probe/audit.json)

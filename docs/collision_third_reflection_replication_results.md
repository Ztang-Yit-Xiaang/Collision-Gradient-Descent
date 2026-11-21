# E43｜第三次反彈預警的新種子驗證

80 個新種子（2820–2899）、每種子六初始動能，共 480 組相同初態。E40 對照與新增政策都沿用同一早期軌跡。新增政策在第三次反彈略過曾越線者，對其餘狀態預測到下一次接觸。預測的 loss 若嚴格高於自身初始值加 \(10^{-4}\)，就做 E39 受限轉移。比較到 \(t=5.00\)，無調參。[固定協議](superpowers/specs/2026-11-12-third-reflection-replication-design.md)

| 指標 | E40 | E40 加第三次反彈預警 |
|---|---:|---:|
| 好粒子途中曾明顯偏離初始解的次數，至 \(t=5.00\) | 6/480 | 1/480 |
| 納入精確接觸，至 \(t=5.00\) 曾越線 | 11/480 | 2/480 |
| \(t=2.00\) 後越線，含精確接觸 | 10/480 | 1/480 |
| \(t=5.00\) 當下在門檻內 | 480/480 | 480/480 |
| 政策總查詢 | 687,474 | 687,952 |
| 政策總轉移 | 384 | 393 |

新增規則防止五次保存點越線及四次只在精確接觸觀察到的越線，兩口徑均無新增或提早越線。九次警報全在初始動能 \(8\times10^{-5}\) 層，距原分支首次晚發越線的提前量中位數約 0.559。四個只在接觸越線的案例於約 2.131 出現，下一保存點已在門檻內。以保存點為準，不必要轉移為 4/9，未通過預定的不必要轉移至多 10% 條件。納入精確接觸時為 0/9。兩口徑在執行前即固定。[正式指標](../output/collision_third_reflection_replication/third_reflection/replication_results.json) · [逐組配對](../output/collision_third_reflection_replication/third_reflection/paired_outcomes.json)

478 個符合資格的狀態各查一次，9 組觸發轉移、469 組保留。配對續跑的後續成本無差異，政策淨增 478 次查詢，約 0.07%。九次轉移的總動能與線動量分量殘差最大分別為 4.44e−16 和 3.33e−16。[第三次反彈實付](../output/collision_third_reflection_replication/third_reflection/execution_counts.json) · [第三次反彈物理帳本](../output/collision_third_reflection_replication/third_reflection/physical_ledgers.json)

## 已越線者

種子 2840 與 2855 在第二次精確接觸越線，第三次反彈時兩者已回到門檻內，因此跳過查詢並保留動量。2840 在 2.10 再越線，2.25 再恢復。2855 維持在門檻內。兩組均在 5.00 結束時在門檻內，第三次反彈時仍在門檻外的案例為零。[恢復逐例](../output/collision_third_reflection_replication/third_reflection/already_deviated_cases.json)

## 執行與範圍

主批 713 前綴、713 首接觸、1,084 段 E37 續跑、13 段第二次反彈轉移續跑、480 段延長至 5.00 的對照、9 段第三次反彈轉移續跑。實付 1,002,088 次，含第三次預測 478 次和轉移續跑 9,016 次。加入工程預檢後總實付 1,017,944 次。解析另 305,892 值、14,200 梯度。無求解失敗，290 項測試通過。此輪驗證同一合成地形的新種子，是否將短暫的接觸越線列為需介入的事件會影響成本效益判讀。[長窗稽核](../output/collision_third_reflection_replication/third_reflection/audit.json) · [延長段物理帳本](../output/collision_third_reflection_replication/extension/physical_ledgers.json)

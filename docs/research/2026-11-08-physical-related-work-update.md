# 物理模型直接相關文獻：補充閱讀紀錄

日期：2026-11-08。本筆與 E16 群體實驗同時整理，未據此改變已凍結的碰撞規則、種子或評估額度。

## Colliding Bodies Optimization：不能只以「多維」區分

Kaveh、Mahdavi 的 [Colliding Bodies Optimization Algorithms，2015 專書第 2 章](https://link.springer.com/chapter/10.1007/978-3-319-19659-6_2)摘要明確提到 1D-CBO、2D-CBO，以及後者加入記憶。本次只能讀出版社公開摘要／書目，正文為訂閱內容。因此不能把「考慮二維碰撞」本身當作我們已確認的新穎性。

要進一步比較的是：事件是否來自連續軌跡的實際接觸、質量與速度怎麼產生、是否使用 fitness 排序、恢復係數如何設定、哪個子系統守恆，以及計算成本。尚未取得正文的更新式，不推測其細節，也不宣稱已證明它與我們相同或不同。原始 2014 年文章仍僅由主要來源的參考文獻確認書目，未完成全文核對。

## EASGD：已有神經網路的物理耦合背景

Zhang、Choromanska、LeCun，[Deep learning with Elastic Averaging SGD，NeurIPS 2015](https://proceedings.neurips.cc/paper/2015/hash/d18f655c3fce66ca401d5f38b48c89af-Abstract.html)。本次由摘要進一步讀了 [原論文第 2–3 節](https://proceedings.neurips.cc/paper_files/paper/2015/file/d18f655c3fce66ca401d5f38b48c89af-Paper.pdf)。

其式 (2) 使用 local objectives 加上與中心變數的二次耦合項；式 (5)–(6) 將各 worker 的隨機梯度更新與對中心的彈性拉動結合。這與由幾何接觸決定瞬間法向衝量的模型不同，不能把它當作剛體碰撞或單次動能／動量守恆的同義詞。

它是未來多個訓練者相互作用、探索與通信成本的必要背景候選。本次沒有重現其性能，也沒有把其圖像分類結果當作本專案 NanoGPT 的證據。

## 對本專案的定位

本輪 E16 先檢驗完整法向碰撞規則相對獨立粒子與守恆隨機出射的價值。相關文獻用來界定下一步應比較的機制；不以名稱相似合併算法，也不以物理命名本身作新穎性或性能證據。

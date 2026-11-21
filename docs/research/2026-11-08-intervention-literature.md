# 有限成本介入與保護好解：前人經驗及研究缺口

檢索日期：**2026-11-08**。這是問題導向的範疇檢索，涵蓋基礎方法與 2023–2026 年相關進展，不是完整系統性回顧或原創性認證。完整查詢、書目、閱讀深度與存取限制見 [檢索與引用登錄](2026-11-08-search-and-citations.json)。

## 目前的回答

**有相當接近的前人經驗。** 保存既有點、有限時間擾動、接受／拒絕、容許非單調下降，以及按成本分配搜索操作，都已有研究。下一輪應借用這些保護與決策思想，再檢驗碰撞／衝量的具體訊息價值。

目前最直接可借鏡的是：

1. **擾動有觀察時域與退出條件。** 不因第一步上坡就拒絕，也不無限重複擾動。
2. **把探索中的狀態與準備交付的解分開。** 探索可以暫時較差，輸出有明確的保留／接受機制。
3. **把保護與成本公平分開驗證。** 保存舊解並不代表勝過把相同額度全部用於原版更新。
4. **帶噪接受有錯誤機率。** 一個 batch 上較低，不能保證真實目標或 test loss 不受傷害。

這四項是本次閱讀的綜合判斷；下面分別列出來源與適用限制，沒有宣稱現有方法已解決我們的完整問題。

## 「不傷害好解」需要分成哪些意思

| 保護對象 | 可以怎麼定義 | 不能因此推出什麼 |
|---|---|---|
| 保存的既有解 | 對確定性同一目標保留目前最佳可輸出點 | 不代表探索過程不會暫時上坡 |
| 介入前的狀態 | 有限探索後不夠好就拒絕／回復 | 不代表比同額度繼續訓練更好 |
| 不介入的未來路徑 | 與原版在相同總成本下比較 | 需要付出比較或估計的成本 |
| 真實期望 loss／泛化 | 控制錯誤接受率，最後獨立評估 | 有噪樣本判定不是逐次零風險保證 |

我們先前把「歷史最佳仍然很好」與「最後保住好解」分開評分，正是第一列與後面幾列的差別。這個區分需要保留在後續報告。

## 最相關的八篇：方法、成本與能借用的部分

### R01：擾動後等待，再決定是否保留原點

Jin、Ge、Netrapalli、Kakade、Jordan（ICML 2017）的 [How to Escape Saddle Points Efficiently](https://proceedings.mlr.press/v70/jin17a.html)，Algorithm 2 在梯度小時保存點並擾動，經固定步數若沒有足夠下降，就回傳保存點。**原算法此時終止；改成回滾後繼續是額外設計。** 理論針對光滑性假設下的近似二階駐點，不保證逃出任意壞局部最小值。[全文 Algorithm 2，PDF 第 5 頁](https://proceedings.mlr.press/v70/jin17a/jin17a.pdf)

對我們的啟發：保留「短暫上坡的機會」，同時為無效探索設明確終點。觀察期的梯度／loss 必須計費；不能把論文中的理論參數直接當成 NanoGPT 已驗證設定。

### R02：比較普通候選與外推候選

Li、Zhou、Liang、Varshney（ICML 2017）的 [Convergence Analysis of Proximal Gradient with Momentum for Nonconvex Optimization](https://proceedings.mlr.press/v70/li17g.html)，分析 APGnc 比較普通近端梯度點與動量外推點、選較低目標值的規則。方法的非凸／KŁ 條件與隨機變體需分別核對；不能直接移植成一般 minibatch 保證。候選的額外函數評估也不是免費。[全文 Algorithms 2–3](https://proceedings.mlr.press/v70/li17g/li17g.pdf)

對我們的啟發：物理方向可以當候選，而接受層有獨立依據。它與「只看速度方向就必定減速」是不同的決策方式。

### R03：信賴區域也不能保證每次有噪接受都正確

Chen、Menickelly、Scheinberg 的 [Stochastic Optimization Using a Trust-Region Method and Random Models](https://arxiv.org/abs/1504.04231)（2015 預印本；本次讀 2016 v2），用模型與函數估計判定步驟是否接受，並調整信賴區域。文中明確區分真／假成功：**依估計接受的步驟，仍可能使真實目標上升。** 理論依賴估計準確度等條件，收斂目標是一階駐點。[全文接受規則與成功／失敗討論](https://arxiv.org/html/1504.04231v2)

對我們的啟發：噪聲下應報錯誤接受與傷害程度，不能只把 eps 換成 adaptive 就宣稱「安全」。模型建構與增加估計精度的樣本都要列成本。

### R04：把接受判定做成機率判定

Mahsereci、Hennig（NeurIPS 2015）的 [Probabilistic Line Searches for Stochastic Optimization](https://arxiv.org/abs/1502.02846)，以輕量一維高斯過程及帶噪函數／方向導數，評估 Wolfe 條件成立的機率。模型推論便宜與目標／梯度評估便宜是不同事情；機率門檻也不等於零錯誤。[全文噪聲、接受條件與成本段落](https://arxiv.org/html/1502.02846)

對我們的啟發：可考慮用已有統計量決定是否值得追加少量評估；成本報表須包括那些新增樣本。

### R05：容許非單調步驟，避免把每次上坡都擋掉

Galli、Rauhut、Schmidt（NeurIPS 2023）的 [Don't be so Monotone: Relaxing Stochastic Line Search in Over-Parameterized Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6d0bf1265ea9635fb4f9d56f16d7efb2-Abstract-Conference.html)，放寬 minibatch 上每步都下降的要求。PoNoS 用歷史參考值構成接受界限；接受上坡仍有規則。其插值等理論條件不能自動套到我們的語言模型。[全文 Algorithm 1](https://arxiv.org/html/2306.12747)

對我們的啟發：使用者最早提出的「短暫上坡可能越嶺」有成熟的非單調接受思想可對照；不能把非單調直接等同任意加能。

### R06：2026 年的動量＋隨機線搜尋工作

Lapucci、Pucci（2026）的 [Effectively leveraging momentum terms in stochastic line search frameworks for fast optimization of finite-sum problems](https://link.springer.com/article/10.1007/s10589-026-00776-2)，直接研究動量方向與隨機線搜尋的相容性，使用 mini-batch persistency 與動量參數規則。資料持續／重疊能讓相鄰資訊更有關聯，也需要處理估計偏差。[作者全文 v4](https://arxiv.org/html/2411.07102v4)

對我們的啟發：先檢查「繼承動量與當前不同 batch 梯度衝突」有多少是抽樣造成，才把它當成壞方向。該文於 2026-03-11 出版，非僅 2024 年預印本；本文未重現其實驗。

### R07：選哪個操作，與如何衡量操作價值，是兩個問題

DaCosta、Fialho、Schoenauer、Sebag（GECCO 2008）的 [Adaptive Operator Selection with Dynamic Multi-Armed Bandits](https://www.cs.ubc.ca/~hutter/EARG.shtml/earg/papers08/pap333s1-dacosta.pdf)，區分 credit assignment 與操作選擇；Section 2 回顧也包含以改善除以評估成本的觀念。這不是該文替我們的 Momentum 控制器給出的理論保證。

對我們的啟發：先定義同額度下的有效改善，再考慮 bandit 等選擇器。不能因為碰撞次數少或供能少，就直接視為更有效率。

### R09：物理能量地形中的探索與保留低能結構

Wales、Doye（1997）的 [Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms](https://www-wales.ch.cam.ac.uk/pdf/JPCA.101.5111.1997.pdf)，結合擾動、局部最小化與轉換後地形的搜索，並保存找到的低能結構再精化。這是物理地形中的既有經驗，不是標準動量算法或一般全域最優保證。1998 是 arXiv 上傳年份，不是原出版年。

對我們的啟發：物理探索和最後交付解可以分開；保留好解本身不是新穎性，仍需證明探索花費值得。

## 三篇背景補充

- **R08，2025。** Vaswani、Babanezhad Harikandeh，[Armijo Line-search Can Make (Stochastic) Gradient Descent Provably Faster](https://proceedings.mlr.press/v267/vaswani25a.html)。本次核對摘要／書目；其隨機部分有凸與插值設定等條件，不是一般非凸動量訓練的保證。
- **R10，2012 預印本。** O'Donoghue、Candès，[Adaptive Restart for Accelerated Gradient Schemes](https://arxiv.org/abs/1204.3982)。本專案早已引用的背景，說明因高動量週期行為而重啟已有研究。本次沒有把摘要閱讀冒稱全文複現。
- **R11，2019。** Vaswani 等，[Painless Stochastic Gradient: Interpolation, Line-Search, and Convergence Rates](https://arxiv.org/abs/1905.09997)。隨機 Armijo 與插值背景，本次讀摘要／核對書目；不是已完成的實作對照。

## 文獻對我們既有結果的解釋

以下是依文獻與本專案結果作的推論，**不是原論文替我們驗證的結論**：

| 本專案的發現 | 下一版需要處理的問題 |
|---|---|
| 盲目加能破壞好井，最後達標 10/10→4/10 | 探索狀態與受保護的輸出點分開；設定觀察期與接受／退出條件 |
| 預演能拒絕好井介入，但成本很高 | 保護收益需扣除試探、評估與少做基底更新的代價 |
| 依狀態減速未優於原版／固定減速 | 方向衝突不一定是可利用訊號；加入噪聲與資料相容性檢驗 |
| 小球與相同首次衝量可重放相同目標軌跡 | 碰撞作為物理實現，仍需證明額外幾何／多粒子資訊的價值 |

對應的原始證據見 [實驗總紀錄 E07–E13](../experiment_log.md)，不把不同種子、時間／成本協議的數字混成改版配對效果。

## 碰撞或衝量到底增加了什麼訊息？

### 只有速度變換公式時

若方向／法線完全由同一個已知梯度與速度算出，碰撞映射主要是重組已有資訊。若求接觸點或法線額外查詢了函數／梯度，新增資訊來自那些查詢；比較組也應取得相同資料並計費。

同理，已知首次衝量 J 後，直接施加 J 可以重現該瞬間效果。我們的 E08 已經驗證這個限制。之後的多次互動可能不同，不能由一次重放就否定所有群體效果。

### 第二個粒子或多次觀測時

其他粒子的位置、速度、找到的區域與歷史，可以帶入單粒子缺少的搜索資訊。值得問的是：使用同樣群體資訊的直接衝量／方向選擇器，是否已經同樣有效？如果是，不能把收益歸給碰撞本身。

### 若希望取得地形變化資訊

例如沿方向 d 比較兩處梯度，可構成方向曲率的有限差分估計。這可能幫助區分某些鞍點與局部谷底，但要增加梯度查詢，並處理步長／噪聲；它仍不能在一般問題上廉價辨認「此局部最小值是不是全域最好」。這是待檢驗假說，尚未在本輪執行。

## 建議的下一輪設計，尚未實作／凍結

**先研究接受層，再研究碰撞資訊。** 建議分兩步：

1. **保護與成本基線。** 在確定性壞井、好井、鞍點及一般下降地形，保留完整優化器狀態；給候選有限觀察期，允許途中上坡，屆時按固定準則接受或拒絕。比較普通更新、只保留最佳解、隨機衝量＋相同保護、幾何衝量＋相同保護。歷史最佳不變是保護檢查，不是方法勝出的主要指標。
2. **資訊消融。** 所有衝量分支共用相同接受層、額度與供能上限，再比較隨機方向、同梯度幾何方向、群體資訊方向；有具體再次互動假說才加入真實小球。對照需要相同可用資訊，不能讓物理分支免費多看曲面。

主要評分應是同總額度下的最後／交付 loss、成功且保住成果的比例、傷害率、每單位成本的有效改善，以及額外評估／狀態複製耗時。碰撞數、接受數、供能量是機制量，不作唯一效果指標。

在 Momentum／Adam 類方法，回復必須包含動量／矩估計、步數、資料與隨機狀態；只還原參數不是回復同一算法狀態。若為比較而保留不介入分支，其更新也要付費。噪聲下還需獨立的確認評估或明確的錯誤接受容忍；不能反覆讀 test split 作選擇。

這兩步尚未指定正式種子、門檻或進行訓練。先讓保護版隨機衝量通過好解／成本基線，再問碰撞幾何是否帶來超額收益，會比直接新增一輪大型 NanoGPT 更容易回答問題。

## 檢索與引用的可追溯性

- 使用 `literature-review-lean` 的問題分解、跨來源檢索與引用核對流程；本次不宣稱完成系統性回顧或統計整合。
- 主來源：arXiv、PMLR、NeurIPS 論文集；補充作者公開稿與 Springer 出版頁。OpenReview 指定頁遇到瀏覽器驗證，沒有繞過或把未讀內容當證據。
- 共保留 **11 篇去重後的書目：8 篇核心、3 篇背景**。搜尋工具未可靠提供全體命中／重複數，未編造 PRISMA 數字；排除不相干的對抗隱私擾動、量子操作選擇及僅名稱相近的文章。
- 書目與方法判斷以原論文／作者稿／出版社頁核對。閱讀深度逐篇標示，沒有宣稱逐字精讀全部十一篇或重現其性能。
- 部分 DOI resolver／Crossref API 無法存取。工具偵測 1 個 DOI、未能自動驗證；該 DOI 的題名、作者、卷頁與出版日期已由 Springer 出版頁直接核對。[原始自動檢查](2026-11-08-intervention-literature_citation_report.json)與[人工主要來源核對說明](2026-11-08-citation-verification.json)分開保留，不宣稱自動檢查全通過。
- [BibTeX 書目](intervention-references.bib)可供 proposal／報告引用；原論文的主張與我們的推論仍需在正文分開標明。
- 本次沒有尋遍引用網路、付費資料庫或所有 2026 年文章，不能據此聲稱特定組合無前人。碰撞相關更廣泛的物理優化文獻仍可在假說確定後加查。

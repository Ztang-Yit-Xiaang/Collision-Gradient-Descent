# E17：耗散設計的文獻定位

日期2026-11-08。針對 E16 的「曾達標後離開」現象補查，這是兩個相關來源的有限閱讀，不是完整回顧，也不構成方法原創性判定。

1. Su, Boyd & Candès, *A Differential Equation for Modeling Nesterov's Accelerated Gradient Method: Theory and Insights*, JMLR17(153),2016，較短版本於NeurIPS2014。讀了[作者頁摘要](https://stanford.edu/~boyd/papers/ode_nest_grad.html)：以二階ODE理解加速法。這只支持慣性動力學是已有研究脈絡；本文沒有為我們的硬接觸模型或遞增阻尼排程提供收斂保證。
2. Cherubini, Metafune & Paparella, *On the Stopping Time of a Bouncing Ball*, arXiv0706.1978,2007。讀了[原始摘要](https://arxiv.org/abs/0706.1978)：研究含可變形與內部耗散的彈跳模型，並分析避免無限碰撞累積的性質。這提醒我們接觸／停留依賴具體模型；不能據此斷言本專案的彈性碰撞加分步阻尼必然有或沒有同一問題。

檢索詞：`Su Boyd Candes differential equation modeling Nesterov accelerated gradient damping 2014 arxiv`；`bouncing ball Zeno phenomenon restitution contact finite time paper`。在新正式結果前讀取，沒有用文獻冒充本輪排程的實證支持。

因此先保留碰撞律，選可逐步驗算外部熱與衝量的速度衰減。比較固定弱阻尼、固定強阻尼、預定漸增阻尼；此排程是待測假設，不是已證明優化演算法。若之後要非彈性碰撞或靜止接觸，需要另外指定約束反力／接觸模型、數值方法與驗證。

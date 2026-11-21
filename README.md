# EE 5239：反射、重力與多粒子碰撞的可重現原型

這個專案實作主動反射、融合接觸、純重力及多粒子彈性碰撞，並提供 momentum SGD、Adam／AdamW 與外部介入控制。除了 **NumPy 合成函數**，已新增官方 NanoGPT 小模型的標準優化器及介入基準；真實地形碰撞版尚未接入神經網路。

## 開始使用

指標用語統一為「**好粒子途中曾明顯偏離初始解的次數**」：在一次實驗已保存的觀察點中，好粒子的loss曾比自身初始值上升超過0.0001，就記1次。20/20表示20次全發生，0/20表示全未超過門檻；途中曾發生與最後是否恢復分開報告。

研究歷程與報告入口：[實驗總紀錄及轉向原因](docs/experiment_log.md)、[完整設定／原始檔案登錄](docs/experiment_registry.json)、[實驗清單 CSV](docs/experiment_inventory.csv)。後續每輪都需補上動機、固定協議、所有結果含失敗，以及改方向的依據。

目前的文獻整理：[有限成本介入與保護好解](docs/research/2026-11-08-intervention-literature.md)，含前人方法、適用限制及尚未執行的下一輪設計。

在專案資料夾執行：

```sh
python3 run_smoke.py
python3 -m pytest -q
```

需要 Python 3.10 以上、NumPy、Matplotlib；測試另需 pytest。若使用乾淨環境，可先建立虛擬環境，再安裝 `.[test]`。不需要 GPU。

可調整測試規模與物理參數：

```sh
python3 run_smoke.py --seeds 10 --steps 300 --h 0.05 --beta 0.995 --mixing 0.5 --gap 0.2 --vz -0.1 --output output/smoke
```

程式會輸出 `config.json`、`summary.json`、`runs.csv`、每條軌跡與碰撞事件，以及三張診斷圖。圖中的陰影是不同初始條件的四分位範圍，不是信賴區間。報告中的數字屬於未調參的小型測試，不能用來推定 NanoGPT 效果。

## 方法一：主動反射

實作位於 `src/collision_optim/active.py`。

1. 計算候選速度 `v_trial = beta*v - h*gradient_scale*g`。
2. 候選位置為 `y = x + h*v_trial`。固定 `h` 時，普通 momentum SGD 的等價學習率為 `gradient_scale*h**2`。
3. 計算方向餘弦 `c`，以及同一目標／資料實現下的配對變化 `d=(loss(y)-loss(x))/(1+abs(loss(x)))`。
4. 只有 `c > epsilon_c` 且 `d > epsilon_f` 連續成立 `patience` 次才介入。沒有違反或完成介入後，計數器歸零。
5. 反射只反轉梯度法向分量，保留切向分量。這是人為控制策略，不是物理接觸事件，也不保證反射後 loss 下降。

預設 `epsilon_c=0.1`、`epsilon_f=0.001`、`patience=3` 都是可調的實驗設定，尚未被驗證為最佳值。梯度或候選速度接近零時，略過方向判斷。

`ActiveConfig(adaptive=True)` 支援以配對診斷樣本的平均值與 `epsilon_f + kappa*SE` 比較。對固定、確定性的合成函數，抽樣 SE 為零，因此等同固定門檻。若傳入 `paired_sampler(x,y,rng)`，呼叫端必須確保一對 `(x,y)` 共用同一批資料與隨機實現；每對呼叫仍計入兩次值評估。這個介面目前只供診斷抽樣，並非完整的 PyTorch 隨機訓練介面。

### API 範例

安裝本機套件後，或設定 `PYTHONPATH=src`，可使用：

```python
from collision_optim import (
    ActiveConfig, ActiveReflection, CountedObjective, Quadratic, State, run,
)

method = ActiveReflection(ActiveConfig(h=0.05, beta=0.995))
result = run(method, CountedObjective(Quadratic()), State([1.0], [0.3]), steps=300)
print(result.best_x, result.best_value, result.stop_reason)
```

`MomentumRestart` 使用完全相同的判準，介入時清除繼承動量，重新使用梯度步；`MomentumSGD` 不做反射或重置。

## 方法二：平滑受力與表面接觸的融合模型

實作位於 `src/collision_optim/hybrid.py`。

狀態為 `q=(x,z)`、速度 `w=(v,vz)`，單位質量，固定可行域為 `z >= f(x)`。使用：

```text
U(x,z) = mixing*f(x) + (1-mixing)*z
F(x,z) = (-mixing*grad(f(x)), -(1-mixing))
H = 0.5*||w||^2 + U
```

在表面 `z=f(x)` 上，`U=f(x)`。高度和目標值的尺度在這個原型中以無因次方式統一；這是一個明確的建模選擇，改變目標或座標尺度會改變接觸幾何。

每個大步依序進行阻尼、力的速度更新，以及含碰撞事件的位移子步。水平方向的自由更新具有 momentum SGD 形式，但完整演算法含輔助高度與接觸映射，並非原生 momentum SGD。

反射只在粒子真正接觸表面並朝表面內部運動時發生。接觸點法線為 `n=(-grad(f(x_contact)),1)/sqrt(1+||grad(f(x_contact))||^2)`，速度更新為 `w_after=w_before-2*(w_before@n)*n`。接觸後會繼續完成剩餘時間，必要時在同一大步處理多次碰撞。不使用 loss 惡化或耐心次數來觸發碰撞。

```python
from collision_optim import (
    CountedObjective, HybridConfig, HybridContact, Quadratic, run,
)

oracle = CountedObjective(Quadratic())
method = HybridContact(HybridConfig(h=0.05, beta=0.995, mixing=0.5))
initial = method.initialize(oracle, [1.0], [0.3], gap=0.2, vz=-0.1)
result = run(method, oracle, initial, steps=300)
print(result.best_x, result.best_value, result.stop_reason)
```

### 接觸求解的支援範圍

目前支援 `Affine`、`Quadratic`、`TiltedDoubleWell`、`Rosenbrock`。它們提供沿位移直線的解析多項式，因此可以搜尋所有接觸根，避免「起終點都在表面上方、途中卻穿牆」的錯誤。`x` 可以是多維向量，不限一維。

**這個解析幾何介面不是可直接套到 NanoGPT 的黑箱接觸偵測器。** 自訂目標除了 `value(x)`、`gradient(x)`，還需提供 `path_coefficients(x,displacement)`。沒有此介面時程式會明確拒絕，不會用不可靠的步末判斷假裝精確接觸。

遇到需要持續接觸約束力的精確掠碰，或單步碰撞次數超限時，會回報 `ContactResolutionError`。這類狀態尚未加入完整的持續接觸求解器；程式不會偷偷把位置投影回牆面或刪掉速度。已完成的軌跡和停止原因會保留在報告中。

## 能量與成本如何記錄

兩個方法都把阻尼、瞬間介入的動能變化，以及剩餘積分誤差分開記錄。`energy_residual` 是診斷量，不會被用來修改速度以強行維持能量。

- 主動反射：記錄 `K + gradient_scale*f(x)`。
- 融合模型：記錄完整的 `K + mixing*f(x) + (1-mixing)*z`。
- 固定牆反射不保證粒子的動量守恆；牆面承擔反作用衝量。完全彈性時，反射瞬間動能守恆。
- `elastic_pair_collision` 提供等質量雙體的衝量映射，並有動量與動能守恆測試。早期單粒子比較隔離反射機制；新增的群體排程器與成對比較見後文。

`CountedObjective` 分別計算所有值評估、梯度評估和解析路徑查詢。診斷與接觸點驗證也計費。超過預算時保留最後完整狀態；未完成步驟已花費的評估仍包含在總數。

Smoke test 共用水平作用力、初始位置／速度、步長及阻尼，並加入 `momentum_extra_reserve` 控制組，把融合模型起始的垂直動能與表面上方高度儲能轉配給普通方法的水平動能。**這不代表兩種狀態空間和位能完全相同，也不是已證明的公平能量匹配。** 解析路徑成本另外列出，合成函數的時間不能直接外推到神經網路。

## 接下來的擴充邊界

新增的 NanoGPT 基準比較標準優化器及外部位移控制，沒有宣稱物理碰撞優於 AdamW。真實地形碰撞若要接入神經網路，仍需處理一般黑箱曲面的接觸定位、mini-batch 導致的表面變化，以及群體中的持續接觸與同時多點接觸。

## 高維診斷：100 到 100,000 維

```sh
python3 run_highdim.py --dimensions 100 1000 10000 100000 --seeds 3 --steps 500
```

預設輸出到 `output/highdim`，包含原雙井實驗的事件分析、設定、全部 240 次執行的摘要、每條壓縮純量軌跡，以及 `highdim_comparison.png`。這輪固定兩個方法的既有設定，沒有針對結果調參。

四種測試設定為：

1. 隱式旋轉、條件數 100 的二次函數，所有座標都參與。
2. 每個座標都有一個傾斜雙井的可分離控制組。
3. 同樣的雙井，加上相鄰座標耦合。
4. 有鏈狀耦合與彎曲谷地的 Rosenbrock 目標。

非線性目標使用 `r=sqrt(d)*Q*x` 並平均局部能量，讓函數值、梯度與局部曲率的主要尺度在不同維度大致可比。`Q` 以三個 Householder 反射隱式表示；不建立 `d*d` 矩陣。旋轉本身不會讓旋轉等變的演算法變強或變弱，真正的交互作用來自雙井相鄰耦合及 Rosenbrock 鏈。

初始水平速度的總動能固定，不隨維度線性增加。方法間使用相同初始位置及水平速度；額外能量控制組另外列出。每次執行使用相同的函數／梯度呼叫上限，接觸定位的解析路徑查詢仍分開計數。此輪依然是單粒子反射測試，不能替代多粒子動量交換實驗。

比較指標為 `(best_f-known_minimum)/(initial_f-known_minimum)`，越小越好。雙井的成功標準另外要求至少 90% 的井座標落入較低的負向井，且相對差距不大於 0.1。這個標準和一維測試不同，不能直接比較兩輪成功次數。

`run(..., store_states=False)` 僅保存每步純量診斷，不保留每個時間點的十萬維向量；最後狀態與最佳參數仍在回傳值中。這避免軌跡記憶體隨 `steps*d` 成長。計算主體、曲面多項式與旋轉均採 O(d) 的結構化運算。只有三組種子，圖中四分位範圍不是統計信賴區間。

數值溢位會記為 `numerical_error`，保留最後完整狀態和實際已用成本，不會讓整批停止，也不會把發散當成有效收斂。摘要分開列出接觸失敗與數值失敗，最終差距的中位數排除失敗執行；圖中的失敗方法使用虛線並標註失敗數。

長批次每次執行後寫入 `checkpoint.json`。如果只是執行中斷、程式和設定沒有改變，可用相同指令加上 `--resume` 沿用已保存結果。設定不一致時會拒絕續跑；修改演算法程式後應使用新的輸出資料夾，不應以此功能混合不同版本。

### 數值穩定性診斷

```sh
python3 run_stability.py --dimension 100000 --duration 25 --h-values 0.05 0.025 0.0125
```

此測試針對十萬維 Rosenbrock，固定物理時間及連續阻尼率。較小步長會使用更多步數及評估，因此這不是等計算成本的優化器排名。主動版同時依步長調整耐心次數與 loss 變化門檻，以維持近似相同的觀察時間及變化率容忍度。設定、停止原因、評估次數與能量曲線會保存在 `output/stability`。

## 方法三：純重力、彈性碰撞與持續接觸

```sh
python3 run_gravity.py
python3 run_gravity_duration.py
```

`GravityContact` 位於 `src/collision_optim/gravity.py`。質量為 1，地形為 `z >= height_scale*f(x)`，只有垂直重力，位能為 `gravity*z`。預設 `gravity=0.5`、`height_scale=1`，因此表面上的位能係數與普通動量控制組的 `0.5*f` 相同。這個模型沒有另加水平梯度力，也沒有 loss 惡化觸發器。

自由飛行沿精確的拋體軌跡定位彈性碰撞，碰撞後繼續剩餘時間；表面上的零法向速度狀態則使用法向支撐力，沿曲面積分滑動。支撐力需要變負時離地。曲率由解析路徑的方向二階導數取得，沒有密集 Hessian。滑動採 RK4、步長加倍誤差估計與能量誤差檢查，不能據此宣稱整段模擬精確守恆或具有辛結構。

每個觀察大步開始施加一次速度阻尼，其動能改變獨立記錄；空中重力做功是 `gravity*(z_before-z_after)`。表面座標直接滿足接觸約束；進入此模式時若有浮點級別的約束修正，會另記 `constraint_correction_energy`，也包含在總能量殘差中。沒有以速度縮放補平能量。

```python
from collision_optim import CountedObjective, GravityConfig, GravityContact, Quadratic, run

method = GravityContact(GravityConfig(h=0.025, gravity=0.5))
oracle = CountedObjective(Quadratic())
initial = method.initialize(oracle, [1.0], [0.2], gap=0, on_surface=True)
result = run(method, oracle, initial, 500)
print(result.best_value, result.stop_reason)
```

`run_gravity.py` 預設比較 2、1,000、100,000 維，各三個種子與三種地形，並分為空中落下、表面切向出發兩組，合計 270 次優化執行。所有方法有相同時間與各類評估上限，但實際成本及走過的物理時間可能不同。表面組固定總初始動能；另外的 `momentum_energy_match` 控制與純重力版有相同初始總機械能及相同表面位能係數。這仍不代表兩個狀態空間與運動幾何等價。前一版融合模型的表面位能為 `f`，有效障壁不同，報告保留這個限制。

輸出位於 `output/gravity`，包含純量軌跡、碰撞事件、物理守恆檢查、完整成本與四張比較圖。`airborne__time.png`、`surface__time.png` 使用物理時間；`airborne__gradients.png`、`surface__gradients.png` 使用梯度次數。解析路徑與值查詢另行記錄，不能把梯度次數直接當成總成本。圖中不同方法可因預算在不同時間停止。

持續接觸可能很快用完梯度預算。`run_gravity_duration.py` 因此另做二維與十萬維耦合雙井的固定時間診斷：seed 0、相同初始機械能、時間 25，不設評估上限，成本完整保存於 `output/gravity_duration`。這組只檢查讓模型走足時間後的行為，不是等成本的排名。模型與驗證設計見 `docs/gravity_design.md`。

本輪已完成的結果與限制見 [純重力實驗報告](docs/gravity_results.md)。

## 多粒子：共用時鐘的彈性碰撞

```sh
python3 run_population.py
python3 run_population_cost.py
python3 run_population.py --output output/population_preflight --seeds 2 --steps 60 --geometry-seeds 5
```

`src/collision_optim/population.py` 新增 `PopulationState`、`PopulationConfig`、`PopulationGravity`、`PopulationOracle` 與 `run_population`。各粒子維持原本的純重力自由飛行與地形反彈，所有粒子共用事件時鐘與阻尼時刻。粒子間距離首次達到固定 `rho` 且正在靠近時，沿接觸點中心連線施加等質量彈性衝量；不以 loss 或梯度判斷是否碰撞。

空中各粒子的重力加速度相同，相對位置為直線，因此可解析求出二體接觸時間。每次先推進至最近的地形或粒子間接觸，再重新預測。這避免分別跑完一步後才發現中途已穿過另一粒子。粒子中心受地形限制，`rho` 是模型的相互排斥距離，未額外模擬實體球半徑造成的地形偏移。

此版本是**孤立二體衝量的群體原型**。共享粒子的同時多點接觸、需要持續支撐力的貼面狀態，以及二體碰撞需要地形同時提供衝量時，都會明確停止。既有單粒子持續接觸方法沒有刪除，但尚未與此群體排程器聯立。失敗不以投影位置或任意交換順序掩蓋。

預設用二維參數地形（另加高度），比較 4 個獨立粒子與 4 個可碰撞粒子。接觸距離 0.12／0.24、三種地形、各 20 個種子，共 180 條軌跡及 120 組成對比較；每個種子的獨立基線由兩個接觸距離共用。初始位置、速度、總動能 0.6、總機械能與評估上限完全配對。

每個群體共用 10,000 次總 oracle 呼叫上限，值、梯度與解析路徑每次各記一個呼叫，同時分別列出其數量。這是評估記帳規則，並不假定三種操作實際成本相同；粒子對預測檢查與實際耗時另外記錄。結果按共同觀察網格評分；傾斜雙井的已知分界線穿越時間則沿自由飛行解析記錄，與低 loss 目標到達時間分開。

`run_population_cost.py` 再將已保存軌跡切齊至共同評估上限 `3*N*steps`，預設為 6,000 次。此上限由設定與每步最小查詢量決定，不依 loss 選擇；只採用上限內最後完整觀察點，不插值或外推。這讓「固定運動時間」與「相同評估上限」的結果分開查看，完整原始軌跡不會改動。

`output/population` 保存設定、初態、完整軌跡、逐次碰撞、成對改善／惡化／未完成數、守恆及穿透檢查。受控的斜向能量集中與障壁轉移案例另存為 `mechanisms.json`，不混入隨機初態成效統計。另有固定初始雲團大小、總動能與觀察時間的 2／10／100／1,000 維相遇頻率診斷；這是幾何可行性測試，未宣稱已完成高維優化。

設計與驗證範圍見 [多粒子設計](docs/superpowers/specs/2026-11-05-population-collisions-design.md)，已完成結果見 [多粒子實驗報告](docs/population_results.md)。

## 停滯時發球：有限供能與直接擾動對照

```sh
python3 run_strategic.py
python3 run_strategic.py --output output/strategic_check --seeds 2 --steps 100
```

`src/collision_optim/strategic.py` 在純重力群體原型上加入 `StrategicConfig`、`StagnationMonitor`、`run_strategic`。這輪以一個目標粒子出發，最多新增兩個輔助球；**只評分原目標，輔助球不算額外搜尋者**。這是純重力模型的介入策略，尚未接到 momentum SGD 或 NanoGPT。

每個物理時間 1 的視窗同時檢查：最佳 loss 的相對進展不大於 `1e-4`、平均水平動能不大於 `0.002`、正規化路徑活動不大於 `0.05`。連續兩個視窗成立才允許介入，成功後冷卻 4 個時間單位。這個判準偵測低水平動能的停滯，不能判斷是否已到好解，也不以短暫上坡觸發反射。

比較四組：繼續原運動、直接隨機水平速度擾動、沿可行發球方向的直接擾動，以及真實輔助球。三個介入組均為單次供能 `0.20`、最多兩次、總供能上限 `0.40`，但軌跡改變後的觸發時間與實際花費可能不同。方向配對組與發球組在首次可行介入前使用相同的幾何規劃；直接隨機擾動省去這些查詢。

新球總能量包含動能與 `gravity*z`，位能零點固定為 `z=0`。發球規劃會計入隨後的共同阻尼，讓兩球先飛行約 `0.025` 時間單位再真實接觸；不可行方向放棄，失敗規劃仍計費。新增球會一直保留。**停滯觸發的是發球，反彈仍由接觸產生。**

全系統帳本分開記錄外部帶入的能量／動量、阻尼、重力與地形衝量；另記目標粒子得到的二體能量轉移與重力做功。新增質量使供能帳本依賴固定的位能參考，這個原型尚未具備對 loss 垂直平移的不變性。

預設三種二維情境、各 20 個種子與四組策略，共 240 條軌跡，時間上限 25，整個群體共用 12,000 次 oracle 呼叫上限。壞井初始能量低於障壁；好井與二次函數用來檢查不必要的介入。已知極小值只用於初態與評分，沒有提供給發球方向或停滯偵測器。另按事先固定的 `3*steps` 呼叫上限比較，預設 1,500，只採上限內最後一筆已提交紀錄，不插值或外推。

`output/strategic` 保存設定與程式雜湊、共用初態、逐步壓縮軌跡、偵測紀錄、介入與碰撞事件、固定時間／固定成本的配對摘要，以及兩張圖。既有輸出資料夾禁止覆寫；重跑請指定新的 `--output`。群體持續接觸與共享多點接觸仍未聯立求解。設計見 [停滯發球設計](docs/superpowers/specs/2026-11-05-strategic-launch-design.md)，結果見 [停滯發球實驗報告](docs/strategic_results.md)。

## 預演後才發球：規劃成本與首次衝量控制

```sh
python3 run_predictive.py
python3 run_predictive.py --output output/predictive_budget1500_new --oracle-budget 1500
python3 run_predictive.py --output output/predictive_check --seed-start 90 --seeds 2 --steps 220
```

`src/collision_optim/predictive.py` 負責共用查詢帳本與有限預演，`predictive_runner.py` 負責真實執行及物理帳本。原本的停滯條件只允許規劃；先預演不介入，再測兩個相反方向與由小至大的 `0.10 / 0.20 / 0.30` 供能，每個分支觀察時間固定為 8。只有相對於不介入的原目標最佳 loss 改善超過 `1e-4`，才選該層較好的可行方案；優先採用最低通過的供能層。

四組為繼續運動、廉價隨機速度擾動、預演後在預定接觸時刻施加相同首次衝量，以及預演後真正發球。直接衝量只匹配首次接觸對原目標的衝量和時刻，沒有保留輔助球；其供能記為實際外部做功。兩組仍會因後續互動與剩餘能量不同而產生不同決策。

所有預演都使用同一個已計費 oracle。分支中的低 loss 不會算成真實解，虛擬球也不加入真實供能。規劃額度不足則拒絕，不提交部分分支。介入後至少觀察完整預演時間，沒有改善時延長冷卻；不倒帶或刪除真實小球。

預設三種情境（壞井、好井、精確鞍點），主測試使用 seed 100–119，共 240 條，固定時間上限 25、總查詢上限 24,000；每條最多兩次規劃、18,000 次規劃查詢。另把高預算軌跡在 1,500 次處截取。因為政策會看剩餘預算，**另外直接給總上限 1,500 跑同樣 240 條**，保存在 `output/predictive_budget1500`，與高預算前綴分開解讀。

`output/predictive` 保存全部決策、逐分支的原目標預演軌跡、真實軌跡、碰撞與衝量、分項成本、來源雜湊及圖形。已有輸出禁止覆寫。這是確定性合成地形的診斷版，尚未代表可用於黑箱神經網路；精確鞍點案例特意保留不介入時的對稱性。設計見 [預演介入設計](docs/superpowers/specs/2026-11-06-predictive-intervention-design.md)，結果見 [預演介入實驗報告](docs/predictive_results.md)。

## 標準優化器、介入控制與小型 NanoGPT

```sh
python3 run_optimizer_comparison.py
python3 run_nanogpt_comparison.py
python3 audit_optimizer_comparison.py synthetic output/optimizer_comparison
python3 audit_optimizer_comparison.py nanogpt output/nanogpt_comparison --matched-replay
python3 plot_optimizer_comparison.py
```

合成函數仍只需原本依賴；NanoGPT 另需 PyTorch，可安裝 `.[test,benchmark]`。本輪以已安裝的 PyTorch 2.11.0、CPU float32 執行。`third_party/nanogpt/model.py` 是固定提交的未改動官方版本，保留 MIT 授權；文字資料位於 `data/tinyshakespeare.txt`，來源與雜湊均保存。沒有下載預訓練權重。

`optimizer_controls.py` 實作標準 Momentum、Adam 與 AdamW 更新，逐步與 PyTorch 核對，包括非零 weight decay。外部介入使用獨立衰減速度 q；在 Momentum 上可等價為原動量脈衝，在 Adam 上保留 m、v、step 不直接修改，明確視為額外探索控制，沒有 Adam 能量守恆宣稱。零 weight decay 的 Adam／AdamW 身分關係有獨立測試。

`output/optimizer_comparison` 有六種地形、獨立調參種子、每個基底四組設定，介入組繼承選定的基底設定。正式 seed 200–209，共八種方法。主比較 480 條都使用實際總上限 1,500 次查詢；另有 480 條固定最多 500 個實際更新的高預算診斷，後者不是等成本排名。值、梯度及解析路徑分項計費；物理模型的初始高度／垂直儲能另列，不宣稱與 Adam 等初始機械能。

`output/nanogpt_comparison` 使用 108,352 參數的小型字元模型。文字依順序分成 train／tuning／test，比例 80／10／10；每個基底九個設定，以 tuning split 選擇，三個正式種子的 test split 只於最後評估。每組相同 1,200 個 batch-forward 等價單位，完整訓練更新計 3、無梯度 probe 計 1；這是明列的計算近似，另報實際步數與耗時。預演只讀 train，完整複製模型及 optimizer state，額外運算計費。

本輪神經網路介入沒有觸發。診斷使實際更新從 400 降為 386；以原版只跑 386 次的額外核對，得到完全相同權重，避免把較短訓練誤認為介入效果。這項重放只作因果核對，不用來重選設定或 checkpoint。

設計見 [標準優化器比較設計](docs/superpowers/specs/2026-11-06-standard-optimizer-comparison-design.md)，全部結果與限制見 [標準優化器及 NanoGPT 比較報告](docs/optimizer_comparison_results.md)。重跑須指定新的 `--output`，既有資料禁止覆寫。

## 便宜的動量控制：繼續、減速與有限加能

```sh
PYTHONPATH=src python3 run_momentum_routing.py --suite synthetic --output output/momentum_routing_new
PYTHONPATH=src python3 run_momentum_routing.py --suite nanogpt --output output/nanogpt_routing_new
python3 audit_momentum_routing.py synthetic output/momentum_routing_new
python3 audit_momentum_routing.py nanogpt output/nanogpt_routing_new
```

`momentum_routing.py` 只讀已有的 loss、梯度、動量與位移，選擇繼續、減速或有限脈衝；沒有模型預演或額外診斷前向。`nanogpt_routing.py` 在原生 SGD 動量上實作，減速同時修改原生 buffer 與外部速度。脈衝後保留兩個視窗的探索觀察期，容許短暫上坡。這些是控制動作，沒有冒稱自然碰撞。

本輪正式完成 360 次合成與 18 次 NanoGPT；新增 22 個機制測試後，全套 **189 passed**。每條合成軌跡精確使用 1,500 次查詢，每組 NanoGPT 都完成 400 次更新。供能、耗散、梯度做功與離散誤差分開記錄，不以帳本閉合宣稱全程守恆。

結果顯示低成本停滯判斷仍會誤傷好解：加能讓精確鞍點成功 10/10，卻讓好井最後達標由原版 10/10 降到 4/10。NanoGPT 自適應版平均 loss 為 2.56617，原版 Momentum 2.56932，AdamW 2.50131；三個配對只有一個改善，而且加能沒有觸發，自適應與單純減速的權重完全相同。沒有額外模型前向仍有向量診斷耗時，尚未建立效率或穩定優勢。

完整流程、限制與結果見 [動量控制實驗報告](docs/momentum_routing_results.md)，預先固定的規則見 [實驗設計](docs/superpowers/specs/2026-11-06-momentum-routing-design.md)。原始資料在 `output/momentum_routing`、`output/nanogpt_routing`；`plot_momentum_routing.py` 可重建圖形。

## 減速分支：十個新種子的重複驗證

```sh
PYTHONPATH=src python3 run_damping_replication.py --output output/damping_replication_new
PYTHONPATH=src python3 run_damping_replication.py --audit --output output/damping_replication_new
python3 audit_damping_prefixes.py output/damping_replication_new
```

新增固定時刻減速對照：在完成 80／160／240／320 次更新後，將原生動量減半，不做逐步狀態診斷。與原版 Momentum、依狀態重置、依狀態減速及 AdamW，在種子 400–409 上共完成 **50 次訓練**；每組 400 次實際更新，沒有加能或額外模型查詢。全套 **193 個測試通過**。

十個新種子未支持原本的微小平均收益：原版 Momentum 平均 test loss **2.54697**，依狀態減速 **2.54824**，固定減速 **2.54683**，AdamW **2.49770**。依狀態減速相對原版與固定減速，都是 5 個改善、5 個惡化，兩個配對 bootstrap 區間都跨零。目前不把該判準作為已建立的性能優勢。

完整結果、成本與物理／動量係數的關係見 [減速重複驗證報告](docs/damping_replication_results.md)，規則見 [固定設計](docs/superpowers/specs/2026-11-06-damping-replication-design.md)。資料與圖形在 `output/damping_replication`；`plot_damping_replication.py` 可重建圖形。

## 物理碰撞幾何與守恆隨機出射

```sh
PYTHONPATH=src python3 run_scattering.py --output output/collision_scattering_new
python3 audit_scattering.py output/collision_scattering_new
python3 plot_scattering.py output/collision_scattering_new
python3 plot_scattering_mechanism.py output/collision_scattering_new
```

本輪固定初態、四粒子總動能 0.6、共享查詢上限 6,000，比較獨立粒子、真實彈性碰撞，以及同時守恆線性動量／動能的隨機出射消融。後者通常不守恆角動量，不冒稱完整剛體物理。沒有新增供能或 loss 觸發反射。

完成 **300 條正式軌跡、15 條預檢、1,515 筆單次機制紀錄**，全套 **205 個測試通過**。新種子雙井 rho=0.24 的曾達標由獨立 12/20 提升至真實碰撞 16/20，但隨機出射為 17/20；最後仍達標分別為 2/20、4/20、4/20。機制與探索線索成立，目前沒有法向碰撞穩定優於一般守恆重分配的證據。

完整模型、兩種達標指標、配對精度、失敗／成本及執行紀錄見 [物理碰撞消融報告](docs/collision_scattering_results.md)，凍結協議見 [設計](docs/superpowers/specs/2026-11-08-collision-scattering-design.md)。`output/collision_scattering/shards` 與父資料夾是同一批 300 次執行的分批／合併表示，不重複計數；另保留了不採納的 18 條序列中斷紀錄。

## 探索後的物理耗散（E17）

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_cooling.py --output output/collision_cooling_new --workers 4
python3 audit_cooling.py output/collision_cooling_new
python3 diagnose_cooling.py output/collision_cooling_new
python3 plot_cooling.py output/collision_cooling_new
```

保留彈性碰撞，比較固定弱、固定較強與預定漸增的外部速度阻尼；三地形、三種粒子規則、三排程、新種子600–609，共 **270條正式軌跡、27條預檢，215個測試通過**。每條正式實際用完共同6000次查詢，無數值／接觸失敗，移除的熱與外部衝量逐步列帳。E16的300條另作衍生診斷，不重算為新實驗。

雙井真實碰撞最後達標 weak／constant／ramp 為 **3／2／7（各10次）**，漸增耗散的7次歷史成功最後皆保留。但同排程獨立／隨機各6次成功，真實碰撞最後精度仍有8/10個配對劣於獨立粒子；目前支持耗散時機值得復驗，尚不支持碰撞專屬優勢。完整指標、成本、事後幾何探索與限制見 [E17報告](docs/collision_cooling_results.md)及[凍結設計](docs/superpowers/specs/2026-11-08-collision-cooling-design.md)。

## 固定探索／精修交接與獨立復驗（E18）

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_handoff.py --output output/collision_handoff_new --workers 4
python3 audit_handoff.py output/collision_handoff_new
python3 summarize_handoff.py output/collision_handoff_new
MPLCONFIGDIR=/private/tmp/collision-plots-cache XDG_CACHE_HOME=/private/tmp/collision-font-cache python3 plot_handoff.py output/collision_handoff_new
```

新種子720–739、三地形、三種探索前段，在實際查詢約半程後比較保留交互作用、四粒子解除交互作用、當前最佳粒子固定GD精修。完成 **180個共用前段＋540個後段、另180條無交接耗散復驗**，225個測試通過；正式皆無數值／接觸失敗。完整路徑各計6000查詢，前段共享節省的實際執行與每個方法應付成本分開。

雙井相同真實碰撞前段，三個後段最後達標 **14／16／13（各20）**；二次在共同物理時間解除交互作用20/20有更佳精度。相同GD後段，真實／獨立／隨機前段雙井最後 **13／9／10**，但Rosenbrock沒有同樣一致的優勢。原耗散復驗雙井weak／constant／ramp最後 **4／6／14**。目前主候選為碰撞探索後解除耦合，沒有通用優勢或高維性能主張。

首次700–719嘗試的續跑間距檢查錯誤與中止資料保留；修正後用新種子重跑。完整結果、物理／GD介面、成本與900筆交接配對見 [E18報告](docs/collision_handoff_results.md)及[協議與修正](docs/superpowers/specs/2026-11-08-collision-handoff-design.md)。

## 交接曲線、物理訊號與好解保護（E19）

沿用E18模型，先用800–809開發選比例／局部法步長、封存後以840–859確認，再用880–889獨立做保護壓力測試。比較0／25／50／75／100%額度解除交互作用，新增等查詢四起點GD／Momentum，以及只用過去窗口的離線碰撞價值預測。完整234項測試通過，原始物理／成本與過去特徵核對通過；開發9次局部法數值失敗全保留，確認無失敗。

開發選25%，確認雙井最後達標 **14/20**，獨立 **8/20**，同時機守恆隨機出射 **16/20**；25%對50%沒有一致精度優勢。GD／Momentum在二次與Rosenbrock的最後精度較佳，雙井右井初態則未越井。控制已走時間後，加入本輪物理彙總量沒有改善離線預測誤差，暫不據此部署在線交接器。

刻意安排早期接觸的好參數壓力案例顯示，較晚解除交互作用無法撤銷原粒子的早期偏離；案例含離地儲能，不是靜止接觸或一般傷害率估計。5批含預檢共1002策略視圖、1368實際執行片段、6,009,120實際查詢；共享前段／標記不算獨立樣本。完整曲線、基線、預測限制、成本與重現命令見 [E19報告](docs/collision_timing_results.md)及[凍結協議](docs/superpowers/specs/2026-11-09-collision-timing-design.md)。本輪仍為二維合成地形，沒有新高維／NanoGPT／AdamW性能主張。

## 單次碰撞的方向價值與數值敏感性（E20）

用同一自然接觸入射狀態重放真實出射、四次守恆線性P/K隨機出射及無衝量，之後共同解除二體作用，隔離單次效應。開發1200–1209共89接觸／534重放，模型封存後新1240–1259確認172接觸／1032重放，245項測試與原始守恆／成本／重建核對通過，無solver數值／接觸失敗。

主要1000額度下，真實相對隨機平均的改善／惡化為二次29／31、雙井34／26、Rosenbrock29／23，三題平均效益皆負，窗口改變也會改變部分結論。加入方向特徵的確認MAE **.354926→.356970**，僅10/20種子改善，未達預定準則；目前不支持部署即時碰撞判斷器，亦不代表所有方向資訊或多次集體作用無效。

另以同三個Rosenbrock壓力種子、三種h／兩種容差，分開保存時間10與結果驅動追加的時間25診斷。較晚約時間20重現好粒子偏離；縮小h沒有使它消失，收緊本組容差的狀態完全相同。好粒子含離地儲能、三種子本身好點初態相同，不能把它當靜止平衡或一般傷害率估計。

本輪含預檢與兩批數值追查共 **1737實際執行片段、3,105,756 solver查詢**，初態設定值另1405；完整配對／窗口不重算新實驗。所有資料、失敗與版本、條件接觸的適用範圍及重現命令見 [E20報告](docs/collision_event_results.md)與[凍結協議](docs/superpowers/specs/2026-11-09-collision-event-design.md)。

## 證明之前的穩定性機制證據（E21）

完成58條受控Rosenbrock、270個一次反彈返回探針與36條30反彈軌跡，完整253測試及原始物理／成本核對通過。新診斷使用`run_orbit_stability.py`，舊高維`run_stability.py`保持原用途；首次重名工程預檢已保留、改名恢復後另做修正版預檢。

具體線索：單粒子也會重現原高度的較晚偏離；顯式1e-8高方向微擾在單／四粒子排程幾乎重合。將初始高度由.8368降到.02，h=.0125的最大loss上升由約.05降到4.06e-15。簡單碗型跨曲率1/4/16，數值放大轉換均位於所測「曲率×高度」.49與.51之間；30次實際反彈亦呈現小／大放大量差異。

這支持先研究**局部曲率與可用儲能的配合**。降低高度也降低探索能量，尚未驗證越障取捨；一維無阻尼觀察不是一般多維安全閾值，沒有先寫證明、部署新控制器或宣稱優化優勢。所有420已知執行（含28排除工程預檢）與490644 solver查詢均記錄，完整結果／圖形／重現方法見 [E21報告](docs/collision_orbit_stability_results.md)及[協議](docs/superpowers/specs/2026-11-09-collision-stability-design.md)。

## 曲率配合的初始化儲能與探索分工（E22）

完成兩個分開的600條正式批與60條工程預檢，完整264測試通過；能量、外部衝量、原始指標與成本獨立核對通過。五政策比較全高／低高度、全體曲率cap、保護當前最佳並分配剩餘儲能，以及任選一顆保護。後兩者與全高總初始E/P相同，運行中不加能。

原零初速批有保護訊號，但所有壞井探索皆0，不能據此說探索被保留。另固定新種子與總水平動能.6的移動探索者補充：三剛度好粒子途中曾明顯偏離初始解的次數由全高 **19／20／20降到0／0／0**，壞井最後達標為全高 **17／10／8**、保護最佳 **16／10／7**（每格20）。通過本輪描述性篩選，但等時間探索結果較不一致，沒有統計非劣保證。

目前支持繼續研究**保護好點與探索者的能量分工**。本輪為二維合成地形、保護對象零初速、無二體作用，不能稱為碰撞控制器或SGD／AdamW的新性能排名；不同初速批不合併為原預註冊確認。完整版本、所有原始資料、1260實際執行與7,255,996含設定查詢見 [E22報告](docs/collision_curvature_energy_results.md)、[原協議](docs/superpowers/specs/2026-11-09-curvature-energy-design.md)及[結果驅動補充協議](docs/research/2026-11-10-energy-exploration-amendment.md)。

## 帶動量好點與障壁尺度的固定規則復驗（E23）

沿用E22五配置與參數，改三種障壁尺度及原好粒子的初始動能0／1e-5／1e-3，全部保持群體K=.6、P0。新20種子完成1800正式軌跡，另90工程預檢；272測試及完整原始核對通過，無solver失敗。兩個後處理工程錯誤與修正來源另留存。

小動量的分工配置在三尺度皆 **0/20曾明顯偏離初始解、20/20原好粒子最後達標**；較大動量則各 **20/20曾明顯偏離初始解、20/20最後達標**，最大偏離中位數由全高約.26–.29降到.0014–.0015。證據支持抑制放大與恢復，不能稱任意動量下途中未超過偏離門檻。固定低／全cap也都最後20/20；本輪不證明付費曲率比固定同高度必要。

探索最後數接近或高於全高，但最高障壁全高只有3–4/20成功，三動量皆未通過預定整體辨識門檻；等時間探索亦有損失。完整1890執行、10,886,958含設定查詢、個別收支／失敗、五政策與共同時間結果見 [E23報告](docs/collision_energy_generalization_results.md)與[固定協議](docs/superpowers/specs/2026-11-10-energy-generalization-design.md)。本輪無二體碰撞、在線控制、跨地形族或新神經模型排名。

## 等初始E/P的高度與動量配置消融（E24）

新20種子、三剛度、兩初態，比較四高度×保留／轉移選中動量，完成960正式軌跡與48工程預檢，280測試及完整物理／成本核對通過，無solver失敗。八組初始總E／K／線性P相同，角動量變化另記；這是初始化對照，沒有運行中免費重置或二體碰撞。

curve_keep→curve_transfer把好粒子途中曾明顯偏離初始解的次數 **20／20／20降至0／0／0**，但壞井最後成功 **19／12／6變為19／8／4**（依剛度1／8／32，各20次），未通過原定整體篩選。全高下只移走動量仍全部明顯偏離；固定保守高度.00625＋轉移也0／0／0明顯偏離、探索18／7／4，尚未建立付費曲率的必要性。

具體問題轉為「何時值得保護當前最佳點，何時保留其探索動量」。最後精度在curve_keep已20/20恢復，轉移新增收益主要在途中保護，須衡量探索代價。全部1008執行、5,809,352含設定查詢、初始角動量／逐軌跡收支、同高度費差與探索性分組見 [E24報告](docs/collision_initial_budget_results.md)與[固定協議](docs/superpowers/specs/2026-11-10-initial-budget-design.md)。

## 以初始 loss 差距選擇動量轉移（E25）

固定 κ=8 與高度 0.025，只用已付費的四個初始 loss 選擇轉移。20 個新種子下，條件式使好粒子途中曾明顯偏離初始解的次數由一律保留的 20/20 降至 0/20，原好粒子三政策皆 20/20 最後達標。全從壞井出發時，保留／轉移／條件式的最後達標分別為 9/20、11/20、9/20。條件式辨識了兩種起始情境；一律轉移在這批壞井的達標數較高，和 E24 的配對差值方向相反。見 [E25 結果](docs/collision_conditional_transfer_results.md)。

## 初始 loss 差距重疊的壓力測試（E26）

固定門檻 100，讓含好點與全壞井情境共享相對 loss 分數和初始速度。低分含好點時，規則 20/20 次錯過途中保護，最後達標 15/20，轉移為 20/20。高分全壞井時，規則轉移的最後達標為 11/20，保留為 13/20。兩分支在壞井的配對結果只有少數種子會改變，尚無足夠證據支持初始動能或早期地形衝量的排序。見 [E26 結果](docs/collision_overlap_transfer_results.md)。

## 共用十步觀察後的動量轉移（E27）

比較兩分支從同一觀察終點續跑，轉移保持總動能與線動量，角動量變化另記。好點的 20/20 次途中偏離都在第一次反彈之前已被觀察到。壞井最後達標為保留 11/20、轉移 12/20、loss 規則 12/20、衝量規則 11/20。共同時間四種均 11/20，雙向配對種子不足，衝量規則未通過篩選。見 [E27 結果](docs/collision_shared_prefix_transfer_results.md)。

## 反彈前的好粒子預警（E28）

240 條新軌跡上，前兩步 loss 的二次外推在 104 次反彈前偏離中及時警報 102 次，136 次未偏離中誤報 0 次，中位提前三步。其餘兩次在第二步當下已越門檻。只看初始速度抓到 95 次、誤報 25 次。這批測的是預警，後續可從同一觀察狀態配對檢驗實際介入。見 [E28 結果](docs/collision_prebounce_warning_results.md)。

## 第二步預警後的動量轉移（E29）

從 E28 第二步凍結狀態重播。102 個及時警報配對中，保留皆偏離，轉移皆未偏離，差異維持到 (t=1.00)。另兩個在第二步已偏離的案例，轉移後均於 (t=0.35) 首次回到門檻內。全 240 條中，完整警報政策在 0.30 的偏離由 104/240 降至 2/240，至 1.00 由 180/240 降至 78/240。見 [E29 結果](docs/collision_warning_triggered_transfer_results.md)。

## 反彈後預警與配對轉移（E30）

在 E29 的 136 個未警報狀態中，反彈後首次觀察時已有 10 個偏離。付費預測提前抓到其餘 66 次後續偏離、誤報 0/60，平均兩次新增查詢。配對轉移使這 66 組至 (t=1.00) 均未偏離，10 個已偏離者也均恢復至門檻內。零查詢近似同樣抓到 66 次，但誤報 16 次。見 [E30 結果](docs/collision_postbounce_warning_transfer_results.md)。

## 第一次反彈當下的預警與轉移（E31）

從精確反射後狀態配對續跑。四個接觸時尚未越線的案例全部提前警報，保留在下一保存點皆偏離，轉移至 (t=1.00) 都未偏離。六個接觸已偏離的案例，轉移後均比等到 (t=0.35) 介入早恢復，保留到 (t=1.00) 則全未恢復。見 [E31 結果](docs/collision_contact_time_transfer_results.md)。

## 新種子全體首次反彈預警（E32）

40 個新種子的 240 個狀態皆合格。137 個在接觸尚未偏離的狀態中，單步預警抓到下一觀察點的 9/9 次，誤報 0/128；到 (t=1.00) 另有 67 次較晚偏離。9 個及時警報全數可由轉移防止。103 個接觸已偏離的狀態，轉移後 63 個恢復，保留無一恢復。見 [E32 結果](docs/collision_contact_warning_replication_results.md)。

## 首次反彈當下的長窗預警（E33）

40 個新種子、240 個合格首次反彈中，129 個在接觸尚未偏離。以 (t=1.00) 為判準，長窗補回單步的 64 次漏報，抓到 71/71，誤報 0/58，平均花 1.93 次預測查詢。71 個警報配對全部可用當下轉移防止。111 個接觸時已偏離者，轉移後 71 個恢復，保留無一恢復。見 [E33 結果](docs/collision_contact_long_forecast_results.md)。

## 高初始動能的介入時序（E34）

同一高動能的 40 個新種子上，第一步警報後轉移到 (t=2.00) 皆未偏離。首次反彈時才轉移，到 (t=1.00) 全未恢復，之後 40 個都曾回到門檻內，到 (t=2.00) 剩 13 個維持；一律保留則為 0 個。見 [E34 結果](docs/collision_high_kinetic_intervention_timing_results.md)。

## 第一步與首次反彈的兩階段預警（E35）

六動能、40 個新種子共 240 初態。兩階段在 (t=1.00) 把曾偏離從一律保留的 182/240 降到 17/240，第一步與反彈單獨則為 84/240 和 107/240。共 165 次預防轉移，同決策點一路保留也安全的次數為 0；8 次第一步轉移可由等待反彈後的保護替代。兩階段實際走到反彈已偏離的 17 個另配對，診斷轉移後全恢復，至 (t=2.00) 有 2 個再次越線。見 [E35 結果](docs/collision_two_stage_protection_results.md)。

## 第一步未警報時加入第二步預警（E36）

E35 剩下 17 組的實測方向曲率全高於一步先驗，兩步重估能在原窗辨認 13 組。新 240 初態中，新增第二步將到 (t=1.00) 曾偏離由 19 降到 3，16 次新增轉移全部防止一次偏離。剩三例首次在接觸越線，診斷轉移後全恢復，至 (t=2.00) 無再越線。見 [E36 結果](docs/collision_second_step_protection_results.md)。

## 第二步接近門檻時預測首次接觸（E37）

80 個新種子、六動能共 480 配對。接觸窗規則加查 16 次、加轉移 3 次，三次皆防止固定窗在首次接觸才發生的偏離，兩種對照下不必要轉移均為 0。至 (t=1.00) 曾偏離為 3→0，至 (t=2.00) 為 22→19。加上反彈預測，共多 25 次預測查詢。已在接觸時越線的三例從同一反彈後狀態轉移均恢復且至 (t=2.00) 無再越線。見 [E37 結果](docs/collision_contact_window_guard_results.md)。

## 後續反彈的預警與配對轉移（E38）

重用 E37 的 480 條軌跡，第二次反彈抓到全部 19 次晚發偏離，新增預測查詢 939 次。質心轉移只防止 3 次，另有 15 次提早至 (t=1.00) 以前或當下越線。21 個已有越線紀錄的反彈狀態，保留最後均在門檻內，強制轉移最後均在門檻外。見 [E38 結果](docs/collision_later_reflection_warning_results.md)。

## 限制好粒子水平動能的反彈後轉移（E39）

從 E38 同一反彈後狀態配對。19 個及時警報的保留／質心／受限轉移到 (t=2.00) 曾偏離為 19／16／0，受限組無提早越線。20 個分岔時仍越線的診斷狀態，保留全部恢復並維持，受限轉移僅 1 個恢復。新增預警查詢 0。見 [E39 結果](docs/collision_capped_reflection_transfer_results.md)。

## 新種子的第二次反彈受限轉移（E40）

80 個新種子、480 組配對。E37 至 (t=2.00) 有 20 次曾偏離，第二次反彈有警報且尚未越線才轉移可全數防止，無新增或提早越線。新增 478 次預測查詢、20 次轉移、不必要轉移 0。另兩個第二次接觸已越線者照原動量在下一保存點恢復，無再越線。見 [E40 結果](docs/collision_second_reflection_capped_policy_results.md)。

## 延續 E40 配對至 5.00（E41）

從 E40 封存末狀態續跑 500 段。原先已越井的 174 顆探索粒子，兩政策均為同一顆在 4.70 達標，至 5.00 均離開達標範圍。20 組介入的好粒子在延長段皆無越線，全批 15 次晚發越線皆來自共用未介入軌跡。見 [E41 結果](docs/collision_long_horizon_exploration_results.md)。

## 第三次反彈預測與受限轉移（E42）

重用 E41 的 480 組。第三次反彈預測到下一接觸，防止 15 次晚發越線中的 14 次，餘下一次因先前已越線保留。淨增 478 次查詢，無不必要轉移。見 [E42 結果](docs/collision_third_reflection_guard_results.md)。

## 第三次反彈預警的新種子驗證（E43）

80 個新種子、480 組同初態。加第三次反彈預警後，保存點曾越線 6→1，含精確接觸 11→2，無新增或提早越線。九次新增轉移中四次只防止接觸瞬間的越線，因此保存點口徑未通過預定不必要轉移篩選。淨增 478 次政策查詢。見 [E43 結果](docs/collision_third_reflection_replication_results.md)。

## 第三次反彈的保存點預測（E44）

80 個新種子、480 組配對。預測到接觸後第一個保存點，保住原規則防止的 14 次保存點偏離，第三次轉移 16→14，預測查詢 478→1,406。少轉移的兩組只在接觸越線，下一保存點即恢復。見 [E44 結果](docs/collision_third_reflection_saved_window_results.md)。

## 接觸值接近門檻才延伸預測（E45）

80 個新種子、480 組配對。80% 條件式與完整規則同防止 15 次保存點偏離，無漏報或不必要轉移，第三次反彈預測查詢 1,407→487。見 [E45 結果](docs/collision_third_reflection_contact_gate_results.md)。

## 反彈後首個保存點才越線的壓力測試（E46）

48 組構造配對中，36 組在反彈後首個保存點越線。完整規則全數防止，80% 條件式省 48 次預測但漏掉接觸值為 60%、75% 和 79.99% 的 18 組。另有 12 組直到再下一個保存點才越線，超出兩政策共同的預測窗。見 [E46 結果](docs/collision_post_contact_positive_stress_results.md)。

## 用反彈速度、剩餘時間和曲率篩選加查（E47）

E46 的 48 組配對中，曲率上界補回 80% 規則漏掉的 18 組，與完整規則同防止 36 組。預測 132 次，比完整規則少 12 次。第二個保存點才越線的 12 組另列。見 [E47 結果](docs/collision_post_contact_curvature_screen_results.md)。

## 曲率上界延到反彈後第二個保存點（E48）

E46 同 48 初態配對至 5.00。E48 保住 E47 原本防止的 36 組，補回在第二個保存點才越線的 12 組。保存點曾偏離 12/48→0/48，預測查詢 132→144，含設定及續跑的政策查詢 70,356→70,308，無不必要轉移。見 [E48 結果](docs/collision_second_post_contact_curvature_results.md)。

## 新種子的混合案例配對（E49）

24 個新種子的 168 組中，保留原動量有 141 組曾越線、27 組一路安全。E48 比 E47 多防止 32 組，兩政策不必要轉移皆為 0。預測查詢 394→440，政策總查詢 266,588→266,484。見 [E49 結果](docs/collision_mixed_post_contact_curvature_results.md)。

## 跨過預測窗內的第二次接觸（E50）

重播 E49 的 168 組，預測第二次自然反射後補回全部 14 組，27 組安全軌跡無不必要轉移或新增查詢。整批預測 440→468，政策總查詢 266,484→266,442。見 [E50 結果](docs/collision_second_contact_reflection_results.md)。

## 預測到第三個保存點（E51）

重播 E50 的 168 組，新增防止第三個保存點越線的全部 16 組，27 組安全案例無不必要轉移。預測查詢 468→608，政策總查詢 266,442→266,562。見 [E51 結果](docs/collision_third_saved_point_results.md)。

## 在第三點可能越線時才加查（E52）

同 168 初態保住 E51 的 120 組預防，含第三保存點的 16 組。27 組安全案例無不必要轉移。預測 608→516，政策總查詢 266,562→266,470。見 [E52 結果](docs/collision_third_point_bound_results.md)。

## 新種子驗證第三點上界（E53）

24 個新種子的 168 組配對中，E52 保住 E51 的全部 120 組防止，含第三保存點的 16 組。27 組安全案例無不必要轉移，預測 612→533，政策總查詢 265,976→265,897。見 [E53 結果](docs/collision_third_point_bound_replication_results.md)。

## 第四個保存點有風險才加查（E54）

重播 E53 的 168 組，第四點有風險的七組全數防止，保住原 120 組。27 組安全案例無不必要轉移或新增查詢。預測 533→557，政策總查詢 265,897→265,921。見 [E54 結果](docs/collision_fourth_point_bound_results.md)。

## 參考

- [Adaptive Restart for Accelerated Gradient Schemes](https://arxiv.org/abs/1204.3982)
- [Reflection, Refraction, and Hamiltonian Monte Carlo](https://proceedings.neurips.cc/paper_files/paper/2015/hash/8303a79b1e19a194f1875981be5bdb6f-Abstract.html)
- [LAMMPS 反射壁與數值誤差說明](https://docs.lammps.org/fix_wall_reflect.html)
- [Neighbor List Collision-Driven Molecular Dynamics Simulation](https://math.nyu.edu/inmemoriam/donev/Packing/EDMD.pdf)


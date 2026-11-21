# 反彈矩陣的 Lean 核對

**判定。** 報告定理 1 的線性矩陣分析為 **VALID · LEAN-CHECKED**。使用 [mathproof-skill `7afa5e86`](https://github.com/Ztang-Yit-Xiaang/mathproof-skill/commit/7afa5e86449344277a9e0d7cc3418d728f605aba) 的 checker 2.5，以 Lean 4.35.0 和 Mathlib v4.35.0 重驗 [形式證明](bounce_spectrum.lean)。12 個指定宣告均通過，退出碼 0。來源掃描、九個套件來源、編譯、稽核及核心重放皆為 `PASS`。核對結果僅依賴標準公理 `propext`、`Classical.choice`、`Quot.sound`。

初次重驗的暫存專案以連結讀取套件，觸發新版的來源位置檢查。改用獨立安裝的 Lean 專案後，五個階段皆通過。

| 報告敘述 | Lean 宣告 | 對應範圍 |
| --- | --- | --- |
| \(M\) 的特徵多項式為 \(s^2-(2-aT)s+1\) | `characteristic_polynomial` | 以二階行列式展開，參數 \(a,T\) 為實數，\(s\) 為複數 |
| \(-2<\tau<2\) 時有相異的一對單位模長共軛根，所有軌跡有界 | `stable_spectrum`、`stable_bounce` | 共軛以實部相等、虛部互反表示，模長以 `normSq` 表示。用正定二次量證明原座標有界 |
| \(\tau<-2\) 時兩根滿足 \(\lambda_1<-1<\lambda_2<0\)，有一條無界軌跡 | `real_spectrum`、`expanding_bounce` | 保留兩根的次序、乘積及原座標的發散 |
| \(aT=4\) 時，\(N=M+I\ne0\)、\(N^2=0\)，且 \(M^np=(-1)^n(p-nNp)\) | `nilPart_eq`、`nilPart_nonzero`、`nilPart_square`、`boundary_formula`、`boundary_drift` | 任意 \(p\in\mathbb R^2\) 和自然數 \(n\)，其中 \(N\ne0\) 用 \(T>0\)。對每個 \(Np\ne0\) 證明無界 |
| \(T=2\sqrt{2gH}/g\)、\(a=2\kappa\sqrt{2gH}\) 時，全部擾動有界當且僅當 \(0<\kappa H<1/2\)。等號時 \(\lvert x_n\rvert=nT\) 對 \(p=(0,1)\) 成立 | `physical_bounded_iff`、`physical_boundary_growth` | 使用報告的物理參數與原始 \((x,u)\) 座標。兩個座標皆有界等價於向量範數有界 |

## 從飛行到返回 Jacobian

[飛行與反射的形式證明](bounce_jacobian.lean)從固定能量分支與精確的反射式開始。[報告第 2 節](../ee5239_final_report.pdf#page=2)依相同順序推導。

**判定。** 這段推導為 **VALID · LEAN-CHECKED**。新版 checker 2.5 的[原始 JSON](bounce_jacobian_check.json)回傳 `PASS`，15 個指定宣告皆在核對結果中，五階段包含核心重放全通過。列印摘要的外層程式在讀取 JSON 已省略的欄位時出錯，checker 的原始結果完整保留。

| 報告步驟 | Lean 宣告 | 對應範圍 |
| --- | --- | --- |
| \(f'(x)=\kappa x\)、\(w=\sqrt{2g(H-f(x))-u^2}\) | `wall_deriv`、`physical_radicand` | 二次牆面法線斜率與固定能量根號中的量 |
| \(t_*=2(w-\kappa xu)/(g+\kappa u^2)\) 是第一個正接觸時間 | `gap_factor`、`first_contact` | 在原點的鄰域內，根號內為正、\(t_*>0\)、間隙在此前為正並在 \(t_*\) 為零，接觸速度朝向牆面 |
| \(D_0w=D_0t_*=D_0W=0\)，\(D_0X=dx+T\,du\)，\(W(0)=-v\) | `launch_deriv`、`flight_deriv`、`flight_zero`、`contact_deriv`、`incoming_deriv`、`incoming_zero` | 對精確函數求 Fréchet 導數 |
| \(U=u-2((u,W)\cdot n)n_x=u+s\,2(W-su)/(1+s^2)\) | `reflection_rational` | 從正規化法線直接推得的精確等式 |
| \(R(0)=0,\ D_0R=M\) | `return_zero`、`physical_return_deriv`、`jacobian_apply`、`physical_fderiv_apply` | \(R=(X,U)\)。對所有 \((dx,du)\in\mathbb R^2\)，導數作用等於報告中的矩陣乘法 |

譜分析檔使用同一矩陣在 \((x,u)\) 座標中的作用 \((x+Tu,-ax+(1-aT)u)\)，其中 \(a=2v\kappa\)。

## 假設核對

- \(g,H>0\) 給出 \(T>0\)，使 \(u=(y-x)/T\) 有定義，也使邊界的 \(N(0,1)=(T,-2)\ne0\)。
- \(v^2>0\) 和 \(g>0\) 分別確保原點的平方根與接觸時間商式可微。飛行與反射的局部推導允許任意實數 \(\kappa\)，譜定理採用報告的 \(\kappa>0\)。
- 乘積 \(aT=8\kappa H\) 由物理參數推出。取 \(g=\kappa=1\) 及 \(H=1/4,1/2,1\)，三種譜情形的假設皆可實現。
- 穩定區間的座標界依賴 \(1-\tau^2/4\)。這個正下界在接近 \(\kappa H=1/2\) 時趨於零。

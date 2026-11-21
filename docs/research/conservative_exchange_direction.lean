import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic.Abel
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

open scoped RealInnerProductSpace

namespace ConservativeExchangeDirection

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def UpperAt (f : E → ℝ) (L : ℝ) (y g : E) : Prop :=
  ∀ d : E, f (y + d) - f y ≤ ⟪g, d⟫ + L / 2 * ‖d‖ ^ 2

def pairChange (f : E → ℝ) (μ : ℝ) (y₁ y₂ w v : E) : ℝ :=
  let d := μ • (v - w)
  f (y₁ + d) + f (y₂ - d) - (f y₁ + f y₂)

def score (L μ : ℝ) (g₁ g₂ w : E) : E :=
  g₁ - g₂ - (2 * L * μ) • w

def model (L μ : ℝ) (g₁ g₂ w v : E) : ℝ :=
  μ * ⟪g₁ - g₂, v - w⟫ + L * μ ^ 2 * ‖v - w‖ ^ 2

theorem pair_upper {f : E → ℝ} {L μ : ℝ} {y₁ y₂ g₁ g₂ w v : E}
    (h₁ : UpperAt f L y₁ g₁) (h₂ : UpperAt f L y₂ g₂) :
    pairChange f μ y₁ y₂ w v ≤ model L μ g₁ g₂ w v := by
  have hleft := h₁ (μ • (v - w))
  have hright := h₂ (-(μ • (v - w)))
  simp only [← sub_eq_add_neg, inner_neg_right, norm_neg, norm_smul, Real.norm_eq_abs,
    mul_pow, sq_abs, real_inner_smul_right] at hright
  simp only [norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, real_inner_smul_right] at hleft
  unfold pairChange model
  dsimp only
  rw [inner_sub_left]
  linarith only [hleft, hright]

theorem model_on_sphere (L μ : ℝ) (g₁ g₂ w v : E) (hv : ‖v‖ = ‖w‖) :
    model L μ g₁ g₂ w v = μ * ⟪score L μ g₁ g₂ w, v - w⟫ := by
  unfold model score
  rw [norm_sub_sq_real, hv]
  simp only [inner_sub_left, real_inner_smul_left, inner_sub_right, real_inner_self_eq_norm_sq,
    real_inner_comm v w]
  ring

theorem optimal_model_exchange {f : E → ℝ} {L μ : ℝ} {y₁ y₂ g₁ g₂ w : E}
    (hμ : 0 < μ) (h₁ : UpperAt f L y₁ g₁) (h₂ : UpperAt f L y₂ g₂)
    (hs : score L μ g₁ g₂ w ≠ 0) :
    let s := score L μ g₁ g₂ w
    let v := (-(‖w‖ / ‖s‖)) • s
    let margin := ‖w‖ * ‖s‖ + ⟪s, w⟫
    ‖v‖ = ‖w‖ ∧
      (∀ u : E, ‖u‖ = ‖w‖ → model L μ g₁ g₂ w v ≤ model L μ g₁ g₂ w u) ∧
      0 ≤ margin ∧ pairChange f μ y₁ y₂ w v ≤ -μ * margin := by
  let s := score L μ g₁ g₂ w
  let v := (-(‖w‖ / ‖s‖)) • s
  change ‖v‖ = ‖w‖ ∧
      (∀ u : E, ‖u‖ = ‖w‖ → model L μ g₁ g₂ w v ≤ model L μ g₁ g₂ w u) ∧
      0 ≤ ‖w‖ * ‖s‖ + ⟪s, w⟫ ∧
      pairChange f μ y₁ y₂ w v ≤ -μ * (‖w‖ * ‖s‖ + ⟪s, w⟫)
  have hn : 0 < ‖s‖ := norm_pos_iff.mpr hs
  have hv : ‖v‖ = ‖w‖ := by
    simp only [v, norm_smul, Real.norm_eq_abs, abs_neg,
      abs_of_nonneg (div_nonneg (norm_nonneg w) hn.le)]
    field_simp
  have hs_v : ⟪s, v⟫ = -(‖w‖ * ‖s‖) := by
    rw [show v = (-(‖w‖ / ‖s‖)) • s from rfl, real_inner_smul_right,
      real_inner_self_eq_norm_sq]
    field_simp
  have hs_min (u : E) (hu : ‖u‖ = ‖w‖) : -(‖w‖ * ‖s‖) ≤ ⟪s, u⟫ := by
    have bound := neg_le_of_abs_le (abs_real_inner_le_norm s u)
    rwa [hu, mul_comm ‖s‖ ‖w‖] at bound
  have model_v : model L μ g₁ g₂ w v = -μ * (‖w‖ * ‖s‖ + ⟪s, w⟫) := by
    rw [model_on_sphere L μ g₁ g₂ w v hv]
    change μ * ⟪s, v - w⟫ = _
    rw [inner_sub_right, hs_v]
    ring
  refine ⟨hv, ?_, ?_, (pair_upper h₁ h₂).trans_eq model_v⟩
  · intro u hu
    rw [model_v, model_on_sphere L μ g₁ g₂ w u hu]
    change -μ * (‖w‖ * ‖s‖ + ⟪s, w⟫) ≤ μ * ⟪s, u - w⟫
    rw [inner_sub_right]
    linarith only [mul_le_mul_of_nonneg_left (hs_min u hu) hμ.le]
  · linarith only [hs_min w rfl]

theorem invariants {c w v : E} (hv : ‖v‖ = ‖w‖) :
    (c + v) + (c - v) = (c + w) + (c - w) ∧
      ‖c + v‖ ^ 2 + ‖c - v‖ ^ 2 = ‖c + w‖ ^ 2 + ‖c - w‖ ^ 2 := by
  constructor
  · abel
  · rw [norm_add_sq_real, norm_sub_sq_real, norm_add_sq_real, norm_sub_sq_real, hv]
    ring

theorem quadratic_witness :
    let f : ℝ → ℝ := fun x => x ^ 2 / 2
    UpperAt f 1 4 4 ∧ UpperAt f 1 0 0 ∧
      0 < ‖(1 : ℝ)‖ * ‖score 1 1 (4 : ℝ) 0 1‖ + ⟪score 1 1 (4 : ℝ) 0 1, 1⟫ ∧
      pairChange f 1 4 0 1 (-1) = -4 ∧ f 2 > f 0 := by
  dsimp only
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro d
    rw [Real.inner_apply, Real.norm_eq_abs, sq_abs]
    ring_nf <;> exact le_rfl
  · intro d
    rw [Real.inner_apply, Real.norm_eq_abs, sq_abs]
    ring_nf <;> exact le_rfl
  · norm_num [score, Real.inner_apply, Real.norm_eq_abs, smul_eq_mul]
  · norm_num [pairChange, smul_eq_mul]
  · norm_num

end

end ConservativeExchangeDirection

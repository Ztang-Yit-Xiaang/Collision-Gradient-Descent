import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

set_option autoImplicit false
open scoped BigOperators

namespace NanoGPTDiagnosis

/-- Compare the selected exchange state to the independent arm's selected index. -/
theorem selection_split {ι : Type} (exchange control : ι → ℝ) (i j : ι) :
    exchange i - control j = (exchange j - control j) + (exchange i - exchange j) := by
  ring

/-- Minimizing validation loss can only reduce the exchange arm's validation loss. -/
theorem validation_reselection {ι : Type} (exchange : ι → ℝ) (i : ι)
    (selected : ∀ k, exchange i ≤ exchange k) (j : ι) :
    exchange i - exchange j ≤ 0 :=
  sub_nonpos.mpr (selected j)

/-- For distinct selected indices, the missing test score permits either sign. -/
theorem missing_test_penalty {ι : Type} [DecidableEq ι] {i j : ι} (hd : i ≠ j)
    (score : ℝ) (hs : 0 < score) :
    ∃ low high : ι → ℝ,
      (∀ k, 0 ≤ low k ∧ 0 ≤ high k) ∧
      low i = score ∧ high i = score ∧
      0 < score - low j ∧ score - high j < 0 := by
  refine ⟨fun k => if k = i then score else score / 2,
          fun k => if k = i then score else score + 1, ?_, ?_, ?_, ?_, ?_⟩
  · intro k
    by_cases hk : k = i
    · simp [hk, le_of_lt hs]
    · simp only [if_neg hk]
      constructor <;> linarith
  · simp
  · simp
  · simp [Ne.symm hd]
    linarith
  · simp [Ne.symm hd]

/-- At a shared state and minibatch, the common gradient increment cancels. -/
theorem alignment_difference {d : ℕ} (retained incoming clipped raw : Fin d → ℝ)
    (μ η : ℝ) :
    (∑ k, raw k * ((μ * incoming k - η * clipped k) -
                    (μ * retained k - η * clipped k))) =
      μ * ∑ k, raw k * (incoming k - retained k) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- Equal gradient norms allow either sign for the alignment term. -/
theorem opposite_alignment (μ : ℝ) (hμ : 0 < μ) :
    |(1 : ℝ)| = |(-1 : ℝ)| ∧
      0 < μ * 1 * (1 - 0) ∧ μ * (-1) * (1 - 0) < 0 := by
  constructor
  · exact (abs_neg (1 : ℝ)).symm
  · constructor <;> linarith

/-- A stationary particle that receives nonzero velocity leaves the quadratic minimum. -/
theorem quadratic_loss_rise (μ b : ℝ) (hμ : 0 < μ) (hb : b ≠ 0) :
    (0 : ℝ)^2 / 2 < (μ * b)^2 / 2 := by
  have hprod : μ * b ≠ 0 := mul_ne_zero (ne_of_gt hμ) hb
  have hsq : 0 < (μ * b)^2 := sq_pos_of_ne_zero hprod
  linarith

/-- Exact gap along a flight from a parabolic wall. -/
theorem wall_gap_factor (κ x u w g t : ℝ) :
    κ * x^2 / 2 + w * t - g * t^2 / 2 - κ * (x + u * t)^2 / 2 =
      t * (w - κ * x * u - (g + κ * u^2) * t / 2) := by
  ring

/-- The return Jacobian is the product of the flight and reflection shears. -/
theorem bounce_determinant (v κ T : ℝ) :
    1 * (1 - 2 * v * κ * T) - T * (-2 * v * κ) = 1 := by
  ring

/-- The exact trace range for the linearized bounce, with q = κH. -/
theorem bounce_trace_range (q : ℝ) :
    (-2 < 2 - 8 * q ∧ 2 - 8 * q < 2) ↔ (0 < q ∧ q < 1 / 2) := by
  constructor <;> intro h <;> constructor <;> linarith [h.1, h.2]

end NanoGPTDiagnosis

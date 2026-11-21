import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OptimizerRate

theorem momentum_identity (μ η a v g ξ : ℝ) :
    let s := μ * a * v - η * (g + ξ)
    2 * η * g * s + s ^ 2 =
      (μ * a * v) ^ 2 - η ^ 2 * g ^ 2 + η ^ 2 * ξ ^ 2
        - 2 * η * μ * a * v * ξ := by
  dsimp
  ring

theorem guard_error (m n ε : ℝ) (hn : m ^ 2 ≤ 2 * n ^ 2 + 2 * ε ^ 2) :
    -(n ^ 2) / 2 + 2 * ε * n ≤ -(m ^ 2) / 8 + (17 / 4) * ε ^ 2 := by
  nlinarith [sq_nonneg (n / 2 - 2 * ε)]

theorem auxiliary_square (ρ q j next : ℝ) (hr : 0 ≤ ρ) (hr' : ρ < 1)
    (hn : 0 ≤ next) (hqj : next ≤ ρ * (q + j)) :
    (1 - ρ) * next ^ 2 ≤ (1 - ρ) * ρ * q ^ 2 + ρ ^ 2 * j ^ 2 := by
  have hs := mul_le_mul_of_nonneg_right hqj
    (show 0 ≤ ρ * (q + j) + next by linarith)
  have hsq : next ^ 2 ≤ (ρ * (q + j)) ^ 2 := by nlinarith only [hs]
  have hweighted := mul_le_mul_of_nonneg_left hsq (show 0 ≤ 1 - ρ by linarith)
  calc
    (1 - ρ) * next ^ 2 ≤ (1 - ρ) * (ρ * (q + j)) ^ 2 := hweighted
    _ = (1 - ρ) * ρ * q ^ 2 + ρ ^ 2 * j ^ 2 -
        ρ * ((1 - ρ) * q - ρ * j) ^ 2 := by ring
    _ ≤ (1 - ρ) * ρ * q ^ 2 + ρ ^ 2 * j ^ 2 := by
      linarith [mul_nonneg hr (sq_nonneg ((1 - ρ) * q - ρ * j))]

theorem sum_decrease (z c : ℕ → ℝ) (h : ∀ t, z (t + 1) + c t ≤ z t) (n : ℕ) :
    z n + (∑ t ∈ Finset.range n, c t) ≤ z 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ]
    linarith [h n]

end OptimizerRate

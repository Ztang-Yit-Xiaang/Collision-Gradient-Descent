import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring

open scoped BigOperators

namespace NanoGPTExchange

noncomputable def kinetic {d : ℕ} (v : Fin d → ℝ) : ℝ :=
  (∑ k, (v k)^2) / 2

/-- Momentum is conserved componentwise and kinetic energy is conserved in a full swap. -/
theorem swap_conserves {d : ℕ} (a b : Fin d → ℝ) :
    (∀ k, b k + a k = a k + b k) ∧
    kinetic b + kinetic a = kinetic a + kinetic b := by
  exact ⟨fun k => add_comm (b k) (a k), add_comm _ _⟩

/-- Energy received by one replica leaves its partner. -/
theorem swap_pair_transfer {d : ℕ} (a b : Fin d → ℝ) :
    kinetic b - kinetic a = -(kinetic a - kinetic b) := by
  ring

/-- Summing this component identity gives the aggregate kinetic balance. -/
theorem update_kinetic_component (u g μ η : ℝ) :
    (μ * u - η * g)^2 / 2 - u^2 / 2 =
    -(1 - μ^2) * (u^2 / 2) - μ * η * (u * g) + η^2 / 2 * g^2 := by
  ring

/-- Sum of two displacement updates, componentwise. -/
theorem update_momentum_component (a b ga gb μ η : ℝ) :
    (μ * a - η * ga) + (μ * b - η * gb) - (a + b) =
    (μ - 1) * (a + b) - η * (ga + gb) := by
  ring

end NanoGPTExchange

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.GroupWithZero.Action
import Mathlib.Basic.Real.Basic
import Mathlib.LinearAlgebra.BilinearForm.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

open scoped BigOperators

namespace QuadraticExchange

noncomputable section

variable {E : Type*} [AddCommGroup E] [Module ℝ E]
variable {ι : Type*} [Fintype ι]

def trial (A : E →ₗ[ℝ] E) (b : E) (μ η : ℝ) (x u : E) : E :=
  μ • u - η • (A x + b)

def total (x : ι → E) : E := ∑ i, x i

def mean (x : ι → E) : E := ((Fintype.card ι : ℝ)⁻¹) • total x

theorem step_totals (A : E →ₗ[ℝ] E) (b : E) (μ η : ℝ)
    (x u : ι → E) (π : Equiv.Perm ι) :
    let s := fun i => trial A b μ η (x i) (u i)
    let v := μ • total u - η • (A (total x) + Fintype.card ι • b)
    total (fun i => s (π i)) = v ∧ total (fun i => x i + s i) = total x + v := by
  dsimp
  have hs : total (fun i => trial A b μ η (x i) (u i)) =
      μ • total u - η • (A (total x) + Fintype.card ι • b) := by
    simp only [total, trial, Finset.sum_sub_distrib, ← Finset.smul_sum,
      Finset.sum_add_distrib, ← map_sum, Finset.sum_const, Finset.card_univ]
  constructor
  · rw [← hs]
    exact Equiv.sum_comp π (fun i => trial A b μ η (x i) (u i))
  · rw [← hs]
    exact Finset.sum_add_distrib

theorem mean_trajectory (A : E →ₗ[ℝ] E) (b : E) (μ η : ℝ)
    (x u y v : ℕ → ι → E) (π σ : ℕ → Equiv.Perm ι)
    (hx : ∀ t i, x (t+1) i = x t i + trial A b μ η (x t i) (u t i))
    (hy : ∀ t i, y (t+1) i = y t i + trial A b μ η (y t i) (v t i))
    (hu : ∀ t i, u (t+1) i = trial A b μ η (x t (π t i)) (u t (π t i)))
    (hv : ∀ t i, v (t+1) i = trial A b μ η (y t (σ t i)) (v t (σ t i)))
    (hP : 0 < Fintype.card ι) (h₀ : mean (x 0) = mean (y 0))
    (h₁ : mean (u 0) = mean (v 0)) (t : ℕ) :
    mean (x t) = mean (y t) ∧ mean (u t) = mean (v t) := by
  have hc : (Fintype.card ι : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hP)
  have same : ∀ n, total (x n) = total (y n) ∧ total (u n) = total (v n) := by
    intro n
    induction n with
    | zero =>
      have hx₀ := congrArg (fun z : E => (Fintype.card ι : ℝ) • z) h₀
      have hu₀ := congrArg (fun z : E => (Fintype.card ι : ℝ) • z) h₁
      exact ⟨by simpa [mean, smul_smul, hc] using hx₀, by simpa [mean, smul_smul, hc] using hu₀⟩
    | succ t ih =>
      rcases step_totals A b μ η (x t) (u t) (π t) with ⟨hU, hX⟩
      rcases step_totals A b μ η (y t) (v t) (σ t) with ⟨hV, hY⟩
      have hx' : x (t+1) = fun i => x t i + trial A b μ η (x t i) (u t i) := funext (hx t)
      have hy' : y (t+1) = fun i => y t i + trial A b μ η (y t i) (v t i) := funext (hy t)
      have hu' : u (t+1) = fun i => trial A b μ η (x t (π t i)) (u t (π t i)) := funext (hu t)
      have hv' : v (t+1) = fun i => trial A b μ η (y t (σ t i)) (v t (σ t i)) := funext (hv t)
      rw [hx', hy', hu', hv', hX, hY, hU, hV, ih.1, ih.2]
      exact ⟨rfl, rfl⟩
  exact ⟨congrArg (fun z : E => ((Fintype.card ι : ℝ)⁻¹) • z) (same t).1,
    congrArg (fun z : E => ((Fintype.card ι : ℝ)⁻¹) • z) (same t).2⟩

def quadratic (B : E →ₗ[ℝ] E →ₗ[ℝ] ℝ) (b : E →ₗ[ℝ] ℝ) (c : ℝ) (x : E) : ℝ :=
  B x x / 2 + b x + c

theorem quadratic_step_expansion (B : E →ₗ[ℝ] E →ₗ[ℝ] ℝ)
    (b : E →ₗ[ℝ] ℝ) (c μ : ℝ) (z u : E)
    (symmetric : ∀ x y, B x y = B y x) :
    quadratic B b c (z + μ • u) = quadratic B b c z + μ * (B z u + b u)
      + μ^2/2 * B u u := by
  simp only [quadratic, map_add, map_smul, LinearMap.add_apply, LinearMap.smul_apply, smul_eq_mul]
  rw [symmetric u z]
  ring

theorem pair_loss_difference (B : E →ₗ[ℝ] E →ₗ[ℝ] ℝ)
    (b : E →ₗ[ℝ] ℝ) (c μ : ℝ) (z₁ z₂ u₁ u₂ : E)
    (symmetric : ∀ x y, B x y = B y x) :
    quadratic B b c (z₁ + μ • u₂) + quadratic B b c (z₂ + μ • u₁) -
      (quadratic B b c (z₁ + μ • u₁) + quadratic B b c (z₂ + μ • u₂)) =
    μ * B (z₁-z₂) (u₂-u₁) := by
  simp only [quadratic_step_expansion B b c μ _ _ symmetric,
    map_sub, LinearMap.sub_apply]
  ring

def halfStep (x u : ℝ) : ℝ × ℝ :=
  let w := u/2-x/2
  (x+w,w)

def scalarLoss (x : ℝ) : ℝ := x^2/2

theorem reachable_loss_rise (a : ℝ) (ha : a ≠ 0) :
    let q₁ := halfStep a 0
    let q₂ := halfStep (-a) 0
    let keep₁ := halfStep q₁.1 q₁.2
    let keep₂ := halfStep q₂.1 q₂.2
    let swap₁ := halfStep q₁.1 q₂.2
    let swap₂ := halfStep q₂.1 q₁.2
    keep₁.1 = 0 ∧ keep₂.1 = 0 ∧ swap₁.1 = a/2 ∧ swap₂.1 = -a/2 ∧
      scalarLoss swap₁.1 = a^2/8 ∧ scalarLoss swap₂.1 = a^2/8 ∧ 0 < a^2/8 := by
  dsimp [halfStep, scalarLoss]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals first | exact div_pos (sq_pos_of_ne_zero ha) (by norm_num) | ring

end
end QuadraticExchange

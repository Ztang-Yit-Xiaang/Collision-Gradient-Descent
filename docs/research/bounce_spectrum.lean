import Mathlib.Analysis.Real.Sqrt
import Mathlib.Data.Complex.Basic
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Logic.Function.Iterate
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

set_option autoImplicit false

namespace BounceSpectrum

abbrev State := ℝ × ℝ

/-- The report's Jacobian in outgoing position and horizontal velocity coordinates. -/
noncomputable def bounce (T a : ℝ) (p : State) : State :=
  (p.1 + T * p.2, -a * p.1 + (1 - a * T) * p.2)

/-- The determinant of \(sI-M\), expanded in dimension two. -/
theorem characteristic_polynomial (T a : ℝ) (s : ℂ) :
    (s - 1) * (s - (1 - (a : ℂ) * (T : ℂ))) + (T : ℂ) * (a : ℂ) =
      s^2 - (2 - (a : ℂ) * (T : ℂ)) * s + 1 := by
  ring

def BoundedOrbits (F : State → State) : Prop :=
  ∀ p, ∃ B : ℝ, ∀ n : ℕ, |(F^[n] p).1| ≤ B ∧ |(F^[n] p).2| ≤ B

noncomputable def coords (T : ℝ) (p : State) : State := (p.1, p.1 + T * p.2)

noncomputable def step (τ : ℝ) (p : State) : State := (p.2, τ * p.2 - p.1)

theorem coords_bounce (T a : ℝ) (p : State) :
    coords T (bounce T a p) = step (2 - a * T) (coords T p) := by
  apply Prod.ext
  · rfl
  · simp [coords, bounce, step]
    ring

theorem coords_iterate (T a : ℝ) (p : State) (n : ℕ) :
    coords T ((bounce T a)^[n] p) = (step (2 - a * T))^[n] (coords T p) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', coords_bounce, ih]

noncomputable def energy (τ : ℝ) (p : State) : ℝ :=
  p.1^2 + p.2^2 - τ * p.1 * p.2

theorem step_energy (τ : ℝ) (p : State) : energy τ (step τ p) = energy τ p := by
  simp [energy, step]
  ring

theorem iterate_energy (τ : ℝ) (p : State) (n : ℕ) :
    energy τ ((step τ)^[n] p) = energy τ p := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Function.iterate_succ_apply', step_energy, ih]

theorem stable_step (τ : ℝ) (hlo : -2 < τ) (hhi : τ < 2) : BoundedOrbits (step τ) := by
  let c := 1 - τ^2 / 4
  have hc : 0 < c := by
    dsimp [c]
    nlinarith [mul_pos (sub_pos.mpr hhi) (show 0 < 2 + τ by linarith)]
  have bound (p : State) :
      |p.1| ≤ energy τ p / c + 1 ∧ |p.2| ≤ energy τ p / c + 1 := by
    have h₁ : p.1^2 ≤ energy τ p / c := (le_div_iff₀ hc).mpr (by
      dsimp [c, energy]
      nlinarith [sq_nonneg (p.2 - τ * p.1 / 2)])
    have h₂ : p.2^2 ≤ energy τ p / c := (le_div_iff₀ hc).mpr (by
      dsimp [c, energy]
      nlinarith [sq_nonneg (p.1 - τ * p.2 / 2)])
    have habs (x : ℝ) : |x| ≤ x^2 + 1 := by
      nlinarith [sq_abs x, sq_nonneg (|x| - 1)]
    exact ⟨(habs _).trans (add_le_add h₁ le_rfl), (habs _).trans (add_le_add h₂ le_rfl)⟩
  intro p
  refine ⟨energy τ p / c + 1, fun n => ?_⟩
  simpa only [iterate_energy] using bound ((step τ)^[n] p)

theorem stable_bounce (T a : ℝ) (hT : 0 < T)
    (hlo : -2 < 2 - a * T) (hhi : 2 - a * T < 2) : BoundedOrbits (bounce T a) := by
  intro p
  obtain ⟨B, hB⟩ := stable_step (2 - a * T) hlo hhi (coords T p)
  have hB0 : 0 ≤ B := (abs_nonneg _).trans (hB 0).1
  refine ⟨B + (B + B) / T, fun n => ?_⟩
  have bounds := hB n
  rw [← coords_iterate T a p n] at bounds
  let w := (bounce T a)^[n] p
  change |w.1| ≤ B ∧ |w.1 + T * w.2| ≤ B at bounds
  have hy : |w.2| ≤ (B + B) / T := by
    apply (le_div_iff₀ hT).mpr
    calc
      |w.2| * T = |(w.1 + T * w.2) - w.1| := by
        rw [show w.1 + T * w.2 - w.1 = T * w.2 by ring, abs_mul, abs_of_pos hT, mul_comm]
      _ ≤ |w.1 + T * w.2| + |w.1| := by
        simpa using (abs_sub_le (w.1 + T * w.2) 0 w.1)
      _ ≤ B + B := add_le_add bounds.2 bounds.1
  exact ⟨bounds.1.trans (le_add_of_nonneg_right (div_nonneg (add_nonneg hB0 hB0) hT.le)),
    hy.trans (le_add_of_nonneg_left hB0)⟩

/-- The map represented by \(N=M+I\) when \(aT=4\). -/
noncomputable def nilPart (T a : ℝ) (p : State) : State :=
  (2 * p.1 + T * p.2, -a * p.1 - 2 * p.2)

theorem nilPart_eq (T a : ℝ) (h : a * T = 4) (p : State) :
    nilPart T a p = ((bounce T a p).1 + p.1, (bounce T a p).2 + p.2) := by
  apply Prod.ext <;> simp [nilPart, bounce, h] <;> ring

theorem nilPart_square (T a : ℝ) (h : a * T = 4) (p : State) :
    nilPart T a (nilPart T a p) = (0, 0) := by
  apply Prod.ext <;> simp only [nilPart]
  · linear_combination -p.1 * h
  · linear_combination -p.2 * h

theorem nilPart_nonzero (T a : ℝ) (hT : 0 < T) :
    nilPart T a (0, 1) ≠ (0, 0) := by
  intro h
  have h₁ := congrArg Prod.fst h
  simp [nilPart] at h₁
  exact hT.ne' h₁

/-- The exact boundary power formula, in the report's original coordinates. -/
theorem boundary_formula (T a : ℝ) (h : a * T = 4) (p : State) (n : ℕ) :
    (bounce T a)^[n] p =
      ((-1 : ℝ)^n * (p.1 - (n : ℝ) * (nilPart T a p).1),
       (-1 : ℝ)^n * (p.2 - (n : ℝ) * (nilPart T a p).2)) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply', ih]
    have hdiag : 1 - a * T = -3 := by linarith
    apply Prod.ext
    · simp only [bounce, nilPart, Nat.cast_succ, pow_succ]
      linear_combination (-1 : ℝ)^n * (n : ℝ) * p.1 * h
    · simp only [bounce, nilPart, Nat.cast_succ, pow_succ, hdiag]
      linear_combination (-1 : ℝ)^n * (n : ℝ) * p.2 * h

theorem boundary_linear_growth (T a : ℝ) (hT : 0 < T) (h : a * T = 4) (n : ℕ) :
    |((bounce T a)^[n] (0, 1)).1| = (n : ℝ) * T := by
  rw [boundary_formula T a h]
  simp [nilPart, abs_mul, abs_pow, abs_of_nonneg (Nat.cast_nonneg n : 0 ≤ (n : ℝ)), abs_of_pos hT]

private theorem drift_gt (x d B : ℝ) (hd : d ≠ 0) :
    ∃ n : ℕ, B < |x - (n : ℝ) * d| := by
  have hdabs := abs_pos.mpr hd
  obtain ⟨n, hn⟩ := exists_nat_gt ((B + |x|) / |d|)
  have hg : B + |x| < (n : ℝ) * |d| := (div_lt_iff₀ hdabs).mp hn
  have ht : (n : ℝ) * |d| ≤ |x| + |x - (n : ℝ) * d| := calc
    (n : ℝ) * |d| = |x - (x - (n : ℝ) * d)| := by
      rw [show x - (x - (n : ℝ) * d) = (n : ℝ) * d by ring, abs_mul,
        abs_of_nonneg (show 0 ≤ (n : ℝ) from Nat.cast_nonneg n)]
    _ ≤ |x - 0| + |0 - (x - (n : ℝ) * d)| := abs_sub_le _ _ _
    _ = _ := by rw [sub_zero, zero_sub, abs_neg]
  exact ⟨n, by linarith⟩

theorem boundary_drift (T a : ℝ) (h : a * T = 4) (p : State)
    (hp : nilPart T a p ≠ (0, 0)) (B : ℝ) :
    ∃ n : ℕ, B < |((bounce T a)^[n] p).1| ∨ B < |((bounce T a)^[n] p).2| := by
  by_cases hx : (nilPart T a p).1 = 0
  · have hy : (nilPart T a p).2 ≠ 0 := fun h₂ => hp (Prod.ext hx h₂)
    obtain ⟨n, hn⟩ := drift_gt p.2 (nilPart T a p).2 B hy
    use n
    right
    rw [boundary_formula T a h]
    simpa [abs_mul, abs_pow] using hn
  · obtain ⟨n, hn⟩ := drift_gt p.1 (nilPart T a p).1 B hx
    use n
    left
    rw [boundary_formula T a h]
    simpa [abs_mul, abs_pow] using hn

theorem boundary_not_bounded (T a : ℝ) (hT : 0 < T) (h : a * T = 4) :
    ¬ BoundedOrbits (bounce T a) := by
  intro bounded
  obtain ⟨B, hB⟩ := bounded (0, 1)
  obtain ⟨n, hn⟩ := exists_nat_gt (B / T)
  have hg : B < (n : ℝ) * T := (div_lt_iff₀ hT).mp hn
  exact (not_lt_of_ge ((boundary_linear_growth T a hT h n) ▸ (hB n).1)) hg

theorem expanding_root (τ : ℝ) (hτ : τ < -2) :
    ∃ lam : ℝ, lam^2 - τ * lam + 1 = 0 ∧ lam < -1 := by
  have hd : 0 ≤ τ^2 - 4 := by nlinarith [sq_nonneg (τ + 2)]
  have hs := Real.sq_sqrt hd
  refine ⟨(τ - Real.sqrt (τ^2 - 4)) / 2, ?_, ?_⟩
  · nlinarith
  · linarith [Real.sqrt_nonneg (τ^2 - 4)]

theorem real_spectrum (τ : ℝ) (hτ : τ < -2) :
    ∃ lam1 lam2 : ℝ,
      lam1^2 - τ * lam1 + 1 = 0 ∧ lam2^2 - τ * lam2 + 1 = 0 ∧
      lam1 < -1 ∧ -1 < lam2 ∧ lam2 < 0 ∧ lam1 * lam2 = 1 := by
  obtain ⟨lam, hroot, h₁⟩ := expanding_root τ hτ
  let r := τ - lam
  have hprod : lam * r = 1 := by dsimp [r]; nlinarith
  have h₂ : r < 0 := by
    by_contra! hge
    have hle := mul_nonpos_of_nonpos_of_nonneg (show lam ≤ 0 by linarith) hge
    linarith
  have h₃ : -1 < r := by
    by_contra! hle
    have hmul := mul_nonneg_of_nonpos_of_nonpos (show lam + 1 ≤ 0 by linarith)
      (show r + 1 ≤ 0 by linarith)
    dsimp [r] at hmul hprod
    nlinarith
  exact ⟨lam, r, hroot, by dsimp [r]; linear_combination hroot, h₁, h₃, h₂, hprod⟩

theorem stable_spectrum (τ : ℝ) (hlo : -2 < τ) (hhi : τ < 2) :
    ∃ z w : ℂ,
      z^2 - (τ : ℂ) * z + 1 = 0 ∧ w^2 - (τ : ℂ) * w + 1 = 0 ∧
      z ≠ w ∧ z.re = w.re ∧ z.im = -w.im ∧ Complex.normSq z = 1 ∧ Complex.normSq w = 1 := by
  have hd : 0 < 4 - τ^2 := by
    nlinarith [mul_pos (sub_pos.mpr hhi) (show 0 < 2 + τ by linarith)]
  let a := τ / 2
  let b := Real.sqrt (4 - τ^2) / 2
  have hb : 0 < b := div_pos (Real.sqrt_pos.2 hd) (by norm_num)
  have hab : a^2 + b^2 = 1 := by
    dsimp [a, b]
    nlinarith [Real.sq_sqrt hd.le]
  have hτa : τ = 2 * a := by dsimp [a]; ring
  have root (u : ℝ) (hu : u^2 = b^2) :
      (⟨a, u⟩ : ℂ)^2 - (τ : ℂ) * (⟨a, u⟩ : ℂ) + 1 = 0 := by
    apply Complex.ext <;>
      simp [pow_two, Complex.mul_re, Complex.mul_im] <;> nlinarith [congrArg (· * u) hτa, congrArg (· * a) hτa]
  refine ⟨⟨a, b⟩, ⟨a, -b⟩, root b rfl, root (-b) (by ring), ?_, rfl, by simp, ?_, ?_⟩
  · intro h
    have him := congrArg Complex.im h
    change b = -b at him
    linarith
  · change a * a + b * b = 1
    nlinarith
  · change a * a + (-b) * (-b) = 1
    nlinarith

theorem eigen_orbit (τ lam : ℝ) (hroot : lam^2 - τ * lam + 1 = 0) (n : ℕ) :
    (step τ)^[n] (1, lam) = (lam^n, lam^(n+1)) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply', ih]
    apply Prod.ext
    · rfl
    · change τ * lam^(n+1) - lam^n = lam^(n+1+1)
      rw [pow_succ lam (n+1), pow_succ lam n]
      linear_combination -(lam^n) * hroot

theorem expanding_bounce (T a : ℝ) (hT : 0 < T) (hτ : 2 - a * T < -2) :
    ¬ BoundedOrbits (bounce T a) := by
  obtain ⟨lam, hroot, hneg⟩ := expanding_root (2 - a * T) hτ
  let p : State := (1, (lam - 1) / T)
  have hp : coords T p = (1, lam) := by
    apply Prod.ext
    · rfl
    · change 1 + T * ((lam - 1) / T) = lam
      field_simp [hT.ne']
      ring
  intro bounded
  obtain ⟨B, hB⟩ := bounded p
  have habs : 1 < |lam| := by rw [abs_of_neg (by linarith : lam < 0)]; linarith
  obtain ⟨n, hn⟩ := pow_unbounded_of_one_lt B habs
  have hx : ((bounce T a)^[n] p).1 = lam^n := by
    have hh := congrArg Prod.fst (coords_iterate T a p n)
    rw [hp, eigen_orbit _ _ hroot] at hh
    exact hh
  rw [← abs_pow, ← hx] at hn
  exact (not_lt_of_ge (hB n).1) hn

theorem bounce_bounded_iff (T a q : ℝ) (hT : 0 < T) (hq : 0 < q) (ha : a * T = 8 * q) :
    BoundedOrbits (bounce T a) ↔ 0 < q ∧ q < 1 / 2 := by
  constructor
  · intro hb
    refine ⟨hq, ?_⟩
    rcases lt_trichotomy q (1 / 2) with h | h | h
    · exact h
    · exact False.elim (boundary_not_bounded T a hT (by linarith) hb)
    · exact False.elim (expanding_bounce T a hT (by linarith) hb)
  · intro h
    exact stable_bounce T a hT (by linarith [h.2]) (by linarith [h.1])

private theorem physical_scales (g κ H : ℝ) (hg : 0 < g) (hH : 0 < H) :
    let v := Real.sqrt (2 * g * H)
    0 < 2 * v / g ∧ (2 * v * κ) * (2 * v / g) = 8 * (κ * H) := by
  let v := Real.sqrt (2 * g * H)
  have hvpos : 0 < v := Real.sqrt_pos.2 (by positivity)
  have hvsq : v^2 = 2 * g * H := Real.sq_sqrt (by positivity)
  change 0 < 2 * v / g ∧ _
  refine ⟨by positivity, ?_⟩
  calc
    (2 * v * κ) * (2 * v / g) = (4 * κ * v^2) / g := by ring
    _ = 8 * (κ * H) := by
      rw [hvsq]
      field_simp [hg.ne']
      ring

/-- Specialization to \(v=\sqrt{2gH}\), \(T=2v/g\), \(a=2v\kappa\). -/
theorem physical_bounded_iff (g κ H : ℝ) (hg : 0 < g) (hκ : 0 < κ) (hH : 0 < H) :
    let v := Real.sqrt (2 * g * H)
    BoundedOrbits (bounce (2 * v / g) (2 * v * κ)) ↔ 0 < κ * H ∧ κ * H < 1 / 2 := by
  obtain ⟨hT, ha⟩ := physical_scales g κ H hg hH
  exact bounce_bounded_iff _ _ _ hT (mul_pos hκ hH) ha

theorem physical_boundary_growth (g κ H : ℝ) (hg : 0 < g) (hH : 0 < H)
    (hcrit : κ * H = 1 / 2) (n : ℕ) :
    let v := Real.sqrt (2 * g * H)
    let T := 2 * v / g
    |((bounce T (2 * v * κ))^[n] (0, 1)).1| = (n : ℝ) * T := by
  obtain ⟨hT, ha⟩ := physical_scales g κ H hg hH
  exact boundary_linear_growth _ _ hT (by rw [hcrit] at ha; norm_num at ha; exact ha) n

end BounceSpectrum

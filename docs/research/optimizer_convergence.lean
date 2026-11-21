import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
Appendices C and D. Time is discrete, replica indices are finite, and the
parameter space is a real inner product space, including every Euclidean space.
-/

namespace OptimizerConvergence

noncomputable section

open scoped BigOperators RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {ι : Type*} [Fintype ι]

def UpperModel (f : E → ℝ) (g : E → E) (L : ℝ) : Prop :=
  ∀ x d, f (x + d) ≤ f x + ⟪g x, d⟫ + L / 2 * ‖d‖ ^ 2

def GradDominance (f : E → ℝ) (g : E → E) (star m : ℝ) : Prop :=
  ∀ x, 2 * m * (f x - star) ≤ ‖g x‖ ^ 2

def SqMass (v : ι → E) : ℝ := ∑ i, ‖v i‖ ^ 2

def Gap (f : E → ℝ) (star : ℝ) (x : ι → E) : ℝ :=
  ∑ i, (f (x i) - star)

def Energy (f : E → ℝ) (star L η : ℝ) (x u : ι → E) : ℝ :=
  Gap f star x + (1 - η * L) / (2 * η) * SqMass u

def Trial (μ η a : ℝ) (u g ξ : E) : E :=
  (μ * a) • u - η • (g + ξ)

def Compatible (g d s : E) : Prop :=
  ⟪g, d⟫ ≤ ⟪g, s⟫ ∧ ‖d‖ ^ 2 ≤ ‖s‖ ^ 2

def Guard (L η : ℝ) (G d : E) : Prop :=
  ⟪G, d⟫ + L / 2 * ‖d‖ ^ 2 ≤ -(η / 2) * ‖G‖ ^ 2 ∧
  ‖d‖ ≤ 2 * η * ‖G‖

noncomputable def GuardedStep (L η : ℝ) (G candidate : E) : E := by
  classical
  exact if Guard L η G candidate then candidate else (-η) • G

theorem sum_descent {z a b : ℕ → ℝ}
    (step : ∀ t, z (t+1) + a t ≤ z t + b t) (T : ℕ) :
    z T + (∑ t ∈ Finset.range T, a t) ≤ z 0 + (∑ t ∈ Finset.range T, b t) := by
  induction T with
  | zero => simp
  | succ t ih =>
    rw [Finset.sum_range_succ, Finset.sum_range_succ]
    linarith [step t]

theorem geometric_descent {z : ℕ → ℝ} {r b : ℝ} (hr' : r ≤ 1)
    (step : ∀ t, z (t+1) ≤ (1-r) * z t + r * b) (T : ℕ) :
    z T ≤ (1-r)^T * z 0 + (1-(1-r)^T)*b := by
  induction T with
  | zero => simp
  | succ t ih =>
    have h := mul_le_mul_of_nonneg_left ih (show 0 ≤ 1-r by linarith)
    rw [pow_succ]
    nlinarith only [step t, h]

theorem some_le_average {a : ℕ → ℝ} {T : ℕ} {B : ℝ} (hT : 0 < T)
    (hs : (∑ t ∈ Finset.range T, a t) ≤ (T : ℝ) * B) :
    ∃ t < T, a t ≤ B := by
  by_contra! h
  have hn : (Finset.range T).Nonempty := ⟨0, Finset.mem_range.mpr hT⟩
  have hlt := Finset.sum_lt_sum_of_nonempty hn
    (fun t ht => h t (Finset.mem_range.mp ht))
  simp at hlt
  linarith

theorem norm_pair_sq (a b : E) : ‖a+b‖^2 ≤ 2*‖a‖^2 + 2*‖b‖^2 := by
  rw [norm_add_sq_real]
  nlinarith only [sq_nonneg ‖a-b‖, norm_sub_sq_real a b]

theorem gradient_upper_model [CompleteSpace E] {f : E → ℝ} {g : E → E} {L : ℝ}
    (hgrad : ∀ x, HasGradientAt f (g x) x)
    (hlip : ∀ x y, ‖g x-g y‖ ≤ L*‖x-y‖) : UpperModel f g L := by
  intro x d
  let q : ℝ → ℝ := fun t => f (x+t • d) - t*⟪g x,d⟫ - L/2*t^2*‖d‖^2
  have hq (t : ℝ) : HasDerivAt q
      (⟪g (x+t • d),d⟫ - ⟪g x,d⟫ - L*t*‖d‖^2) t := by
    have hpath : HasDerivAt (fun r : ℝ => x + r • d) d t := by
      simpa using ((hasDerivAt_id t).smul_const d).const_add x
    have hf : HasDerivAt (fun r : ℝ => f (x+r • d)) ⟪g (x+t • d), d⟫ t := by
      exact (hgrad (x+t • d)).hasFDerivAt.comp_hasDerivAt t hpath
    have hlin := (hasDerivAt_id t).mul_const ⟪g x,d⟫
    have hquad := (((hasDerivAt_id t).pow 2).const_mul (L/2)).mul_const (‖d‖^2)
    have h := (hf.sub hlin).sub hquad
    convert h using 1
    · funext r
      rfl
    · norm_num only [id_eq, Nat.reduceSub, Nat.cast_ofNat, pow_one, mul_one, one_mul]
      ring
  have hanti : AntitoneOn q (Set.Icc 0 1) := by
    apply antitoneOn_of_deriv_nonpos (convex_Icc 0 1)
      (fun t _ => (hq t).continuousAt.continuousWithinAt)
      (fun t _ => (hq t).differentiableAt.differentiableWithinAt)
    intro t ht
    rw [(hq t).deriv]
    have ht0 : 0 ≤ t := (interior_subset ht).1
    have hb := hlip (x+t • d) x
    rw [add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_nonneg ht0] at hb
    rw [← inner_sub_left]
    have hle := le_trans (le_abs_self _) (abs_real_inner_le_norm (g (x+t • d)-g x) d)
    have hh := mul_le_mul_of_nonneg_right hb (norm_nonneg d)
    nlinarith only [hle, hh]
  have h := hanti (show (0:ℝ) ∈ Set.Icc 0 1 by norm_num)
    (show (1:ℝ) ∈ Set.Icc 0 1 by norm_num) (by norm_num)
  simp [q] at h
  linarith

theorem reflection_compatible {g s : E} {e : ℝ}
    (hup : 0 < ⟪g,s⟫) (he : 0 ≤ e) (he' : e ≤ 1) :
    let n := ‖g‖⁻¹ • g
    Compatible g (s - ((1+e)*⟪s,n⟫) • n) s := by
  let n := ‖g‖⁻¹ • g
  change Compatible g (s - ((1+e)*⟪s,n⟫) • n) s
  have hg : g ≠ 0 := by
    intro h
    simp [h] at hup
  have hgn : 0 < ‖g‖ := norm_pos_iff.mpr hg
  have hn : ‖n‖ = 1 := by
    simp [n, norm_smul, hgn.ne']
  have hs : 0 ≤ ⟪s,n⟫ := by
    rw [show n = ‖g‖⁻¹ • g from rfl, real_inner_smul_right, real_inner_comm g s]
    exact mul_nonneg (inv_nonneg.mpr hgn.le) hup.le
  have hgn' : 0 ≤ ⟪g,n⟫ := by
    rw [show n = ‖g‖⁻¹ • g from rfl, real_inner_smul_right, real_inner_self_eq_norm_sq]
    positivity
  constructor
  · rw [inner_sub_right, real_inner_smul_right]
    exact sub_le_self _ (mul_nonneg (mul_nonneg (by linarith) hs) hgn')
  · rw [norm_sub_sq_real, norm_smul, hn, mul_one, real_inner_smul_right,
      Real.norm_eq_abs, sq_abs]
    have h := mul_nonneg (mul_nonneg (by linarith : 0 ≤ 1-e)
      (by linarith : 0 ≤ 1+e)) (sq_nonneg ⟪s,n⟫)
    nlinarith only [h]

theorem trial_identity (p g ξ : E) {η : ℝ} (hη : 0 < η) :
    let s := p - η • (g+ξ)
    ⟪g,s⟫ + ‖s‖^2/(2*η) =
      ‖p‖^2/(2*η) - η/2*‖g‖^2 + η/2*‖ξ‖^2 - ⟪p,ξ⟫ := by
  dsimp
  rw [inner_sub_right, real_inner_smul_right, inner_add_right, real_inner_self_eq_norm_sq,
    norm_sub_sq_real, norm_smul, Real.norm_eq_abs, abs_of_pos hη, mul_pow, norm_add_sq_real,
    real_inner_smul_right, inner_add_right, real_inner_comm p g]
  field_simp [hη.ne']
  <;> ring

theorem momentum_descent {f : E → ℝ} {g : E → E} {star L η μ : ℝ}
    (model : UpperModel f g L) (hη : 0 < η)
    (small : η*L < 1-μ^2) (hμ : 0 ≤ μ)
    (x u x' u' d ξ : ι → E) (a : ι → ℝ)
    (ha : ∀ i, 0 ≤ a i ∧ a i ≤ 1)
    (motion : ∀ i, x' i = x i + d i)
    (direction : ∀ i, Compatible (g (x i)) (d i) (Trial μ η (a i) (u i) (g (x i)) (ξ i)))
    (exchange : SqMass u' ≤ SqMass d) :
    Energy f star L η x' u' + (1-η*L-μ^2)/(2*η)*SqMass u +
        η/2*SqMass (fun i => g (x i)) ≤
      Energy f star L η x u + η/2*SqMass ξ -
        (∑ i, ⟪(μ*a i) • u i, ξ i⟫) := by
  let c := (1-η*L)/(2*η)
  have hc : 0 ≤ c := div_nonneg (by nlinarith [sq_nonneg μ]) (by positivity)
  have hcancel : L/2+c = 1/(2*η) := by
    dsimp [c]
    field_simp
    <;> ring
  have hcoef : c = (1-η*L-μ^2)/(2*η) + μ^2/(2*η) := by
    dsimp [c]
    ring
  have bounds (i : ι) :
      (f (x' i)-star) + c*‖d i‖^2 + η/2*‖g (x i)‖^2 +
          ⟪(μ*a i) • u i, ξ i⟫ ≤
        (f (x i)-star) + μ^2/(2*η)*‖u i‖^2 + η/2*‖ξ i‖^2 := by
    let s := Trial μ η (a i) (u i) (g (x i)) (ξ i)
    have hf := model (x i) (d i)
    rw [← motion i] at hf
    have hs := (direction i).1
    have hn := mul_le_mul_of_nonneg_left (direction i).2 (show 0 ≤ 1/(2*η) by positivity)
    have hi := trial_identity ((μ*a i) • u i) (g (x i)) (ξ i) hη
    change ⟪g (x i), s⟫ + ‖s‖^2/(2*η) =
      ‖(μ*a i) • u i‖^2/(2*η) - η/2*‖g (x i)‖^2 + η/2*‖ξ i‖^2 -
      ⟪(μ*a i) • u i, ξ i⟫ at hi
    have ha0 : 0 ≤ μ*a i := mul_nonneg hμ (ha i).1
    have ha1 : μ*a i ≤ μ := by simpa using mul_le_mul_of_nonneg_left (ha i).2 hμ
    have hh : ‖(μ*a i) • u i‖^2/(2*η) ≤ μ^2/(2*η)*‖u i‖^2 := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ha0, mul_pow, ← div_mul_eq_mul_div]
      apply mul_le_mul_of_nonneg_right _ (sq_nonneg _)
      have hsq : (μ*a i)^2 ≤ μ^2 := by
        nlinarith [mul_nonneg (add_nonneg hμ ha0) (sub_nonneg.mpr ha1)]
      exact div_le_div_of_nonneg_right hsq (by positivity)
    have hcombine : L/2*‖d i‖^2 + c*‖d i‖^2 = ‖d i‖^2/(2*η) := by
      rw [← add_mul, hcancel, one_div, div_eq_mul_inv, mul_comm]
    change ⟪g (x i), d i⟫ ≤ ⟪g (x i), s⟫ at hs
    change 1/(2*η)*‖d i‖^2 ≤ 1/(2*η)*‖s‖^2 at hn
    simp only [div_eq_mul_inv] at hn hi hh hcombine ⊢
    nlinarith only [hf, hs, hn, hi, hh, hcombine]
  have hsum := Finset.sum_le_sum (s := Finset.univ) (fun i _ => bounds i)
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum] at hsum
  change Gap f star x' + c*SqMass d + η/2*SqMass (fun i => g (x i)) +
    (∑ i, ⟪(μ*a i) • u i, ξ i⟫) ≤
    Gap f star x + μ^2/(2*η)*SqMass u + η/2*SqMass ξ at hsum
  have hx := mul_le_mul_of_nonneg_left exchange hc
  change Gap f star x' + c*SqMass u' + (1-η*L-μ^2)/(2*η)*SqMass u +
    η/2*SqMass (fun i => g (x i)) ≤
    Gap f star x + c*SqMass u + η/2*SqMass ξ - _ 
  linarith only [hsum, hx, congrArg (fun b => b*SqMass u) hcoef]

theorem fallback_passes {L η : ℝ} (hη : 0 < η) (hL : η*L ≤ 1) (G : E) :
    Guard L η G ((-η) • G) := by
  rw [Guard, real_inner_smul_right, real_inner_self_eq_norm_sq, norm_smul,
    Real.norm_eq_abs, abs_neg, abs_of_pos hη]
  constructor
  · have h := mul_nonneg (show 0 ≤ 1-η*L by linarith)
      (mul_nonneg hη.le (sq_nonneg ‖G‖))
    nlinarith only [h]
  · nlinarith only [mul_nonneg hη.le (norm_nonneg G)]

theorem guarded_passes {L η : ℝ} (hη : 0 < η) (hL : η*L ≤ 1) (G candidate : E) :
    Guard L η G (GuardedStep L η G candidate) := by
  classical
  unfold GuardedStep
  split
  · assumption
  · exact fallback_passes hη hL G

theorem guard_descent {f : E → ℝ} {g : E → E} {L η ε : ℝ}
    {x G d : E}
    (model : UpperModel f g L) (hε : ‖G-g x‖ ≤ ε) (accept : Guard L η G d) :
    f (x+d)-f x ≤ -(η/2)*‖G‖^2 + 2*η*ε*‖G‖ := by
  have hw : 0 ≤ ε := le_trans (norm_nonneg _) hε
  have hb := mul_le_mul_of_nonneg_right hε (norm_nonneg d)
  have hi := le_trans (le_abs_self _) (abs_real_inner_le_norm (g x-G) d)
  rw [norm_sub_rev, inner_sub_left] at hi
  have hs := mul_le_mul_of_nonneg_left accept.2 hw
  linarith only [model x d, accept.1, hi, hb, hs]

theorem guard_error_descent {f : E → ℝ} {g : E → E} {L η ε : ℝ}
    {x G d : E} (model : UpperModel f g L) (hη : 0 < η)
    (hε : ‖G-g x‖ ≤ ε) (accept : Guard L η G d) :
    f (x+d)-f x + η/8*‖g x‖^2 ≤ (17*η/4)*ε^2 := by
  have hd := guard_descent model hε accept
  have hg := norm_pair_sq G (g x-G)
  rw [add_sub_cancel, norm_sub_rev] at hg
  have hn : ‖G-g x‖^2 ≤ ε^2 := by
    nlinarith only [mul_nonneg (sub_nonneg.mpr hε)
      (show 0 ≤ ε+‖G-g x‖ by
        linarith [norm_nonneg (G-g x)])]
  have hw : -(‖G‖^2)/2+2*ε*‖G‖ ≤ -(‖g x‖^2)/8+17/4*ε^2 := by
    nlinarith only [hg, hn, sq_nonneg (‖G‖/2-2*ε)]
  have hh := mul_le_mul_of_nonneg_left hw hη.le
  nlinarith only [hd, hh]

structure MomentumRun (g : E → E) (μ η : ℝ)
    (x u d ξ : ℕ → ι → E) (a : ℕ → ι → ℝ) : Prop where
  attenuation : ∀ t i, 0 ≤ a t i ∧ a t i ≤ 1
  motion : ∀ t i, x (t+1) i = x t i + d t i
  direction : ∀ t i, Compatible (g (x t i)) (d t i)
    (Trial μ η (a t i) (u t i) (g (x t i)) (ξ t i))
  exchange : ∀ t, SqMass (u (t+1)) ≤ SqMass (d t)

theorem energy_nonneg {f : E → ℝ} {star L η μ : ℝ}
    (lower : ∀ x, star ≤ f x) (hη : 0 < η) (small : η*L < 1-μ^2)
    (x u : ι → E) : 0 ≤ Energy f star L η x u := by
  have hc : 0 ≤ (1-η*L)/(2*η) := div_nonneg (by nlinarith [sq_nonneg μ]) (by positivity)
  exact add_nonneg (Finset.sum_nonneg (fun i _ => sub_nonneg.mpr (lower (x i))))
    (mul_nonneg hc (Finset.sum_nonneg (fun i _ => sq_nonneg ‖u i‖)))

theorem normalize_sum {A B k η n : ℝ} (hη : 0 < η) (hn : 0 < n)
    (bound : η/2*A ≤ B+η/2*k) : A/n ≤ 2*B/(η*n)+k/n := by
  apply (div_le_iff₀ hn).mpr
  have h : A ≤ 2*B/η+k := by
    have hh : η*(A-k) ≤ 2*B := by nlinarith only [bound]
    have hh' : A-k ≤ 2*B/η := (le_div_iff₀ hη).mpr (by simpa [mul_comm] using hh)
    linarith only [hh']
  have he : (2*B/(η*n)+k/n)*n = 2*B/η+k := by
    field_simp
  rwa [he]

theorem momentum_exact_rate {f : E → ℝ} {g : E → E} {star L η μ : ℝ}
    (model : UpperModel f g L) (lower : ∀ x, star ≤ f x)
    (hη : 0 < η) (small : η*L < 1-μ^2) (hμ : 0 ≤ μ)
    (x u d : ℕ → ι → E) (a : ℕ → ι → ℝ)
    (run : MomentumRun g μ η x u d (fun _ _ => 0) a)
    (T : ℕ) (hT : 0 < T) (hP : 0 < Fintype.card ι) :
    (∑ t ∈ Finset.range T, SqMass (fun i => g (x t i))) /
        ((Fintype.card ι : ℝ)*(T : ℝ)) ≤
      2*Energy f star L η (x 0) (u 0)/(η*((Fintype.card ι : ℝ)*(T : ℝ))) := by
  have step (t : ℕ) :
      Energy f star L η (x (t+1)) (u (t+1)) +
        η/2*SqMass (fun i => g (x t i)) ≤ Energy f star L η (x t) (u t) + 0 := by
    have hs := momentum_descent (star := star) model hη small hμ (x t) (u t) (x (t+1)) (u (t+1))
      (d t) (fun _ => 0) (a t) (run.attenuation t) (run.motion t) (run.direction t)
      (run.exchange t)
    simp [SqMass] at hs
    have hn : 0 ≤ (1-η*L-μ^2)/(2*η) * (∑ i, ‖u t i‖^2) :=
      mul_nonneg (div_nonneg (by linarith) (by positivity))
        (Finset.sum_nonneg (fun i _ => sq_nonneg ‖u t i‖))
    unfold SqMass
    linarith
  have h := sum_descent (z := fun t => Energy f star L η (x t) (u t))
    (a := fun t => η/2*SqMass (fun i => g (x t i))) (b := fun _ => 0) step T
  simp only [Finset.sum_const_zero, ← Finset.mul_sum, add_zero] at h
  have hb : η/2*(∑ t ∈ Finset.range T, SqMass (fun i => g (x t i))) ≤
      Energy f star L η (x 0) (u 0)+η/2*0 :=
    by linarith [energy_nonneg lower hη small (x T) (u T)]
  have hn : 0 < (Fintype.card ι : ℝ)*(T : ℝ) := by
    exact mul_pos (Nat.cast_pos.mpr hP) (Nat.cast_pos.mpr hT)
  simpa using normalize_sum hη hn hb

theorem noise_cross_bound (u ξ : E) {μ a η δ : ℝ}
    (hμ : 0 ≤ μ) (ha : 0 ≤ a ∧ a ≤ 1) (hη : 0 < η) (hδ : 0 < δ) :
    -⟪(μ*a) • u, ξ⟫ ≤ δ/(4*η)*‖u‖^2 + η*μ^2/δ*‖ξ‖^2 := by
  have H : -⟪u,ξ⟫ ≤ ‖u‖*‖ξ‖ := le_trans (neg_le_abs _) (abs_real_inner_le_norm _ _)
  have hm := mul_le_mul_of_nonneg_left H (mul_nonneg hμ ha.1)
  have hμa : μ*a ≤ μ := by simpa using mul_le_mul_of_nonneg_left ha.2 hμ
  have hn := mul_le_mul_of_nonneg_right hμa (mul_nonneg (norm_nonneg u) (norm_nonneg ξ))
  have hy : μ*(‖u‖*‖ξ‖) ≤ δ/(4*η)*‖u‖^2 + η*μ^2/δ*‖ξ‖^2 := by
    apply (mul_le_mul_iff_of_pos_left (show 0 < 4*η*δ by positivity)).mp
    have hcalc : (4*η*δ)*(δ/(4*η)*‖u‖^2+η*μ^2/δ*‖ξ‖^2) -
        (4*η*δ)*(μ*(‖u‖*‖ξ‖)) = (δ*‖u‖-2*η*μ*‖ξ‖)^2 := by
      field_simp
      <;> ring
    linarith only [hcalc, sq_nonneg (δ*‖u‖-2*η*μ*‖ξ‖)]
  rw [real_inner_smul_left]
  linarith only [hm, hn, hy]

theorem momentum_pathwise_rate {f : E → ℝ} {g : E → E} {star L η μ : ℝ}
    (model : UpperModel f g L) (lower : ∀ x, star ≤ f x)
    (hη : 0 < η) (small : η*L < 1-μ^2) (hμ : 0 ≤ μ)
    (x u d ξ : ℕ → ι → E) (a : ℕ → ι → ℝ)
    (run : MomentumRun g μ η x u d ξ a)
    (T : ℕ) (hT : 0 < T) (hP : 0 < Fintype.card ι) :
    let n := (Fintype.card ι : ℝ)*(T : ℝ)
    (∑ t ∈ Finset.range T, SqMass (fun i => g (x t i))) / n ≤
      2*Energy f star L η (x 0) (u 0)/(η*n) +
      (1+2*μ^2/(1-η*L-μ^2))*(∑ t ∈ Finset.range T, SqMass (ξ t))/n := by
  let δ := 1-η*L-μ^2
  have hδ : 0 < δ := by dsimp [δ]; linarith
  have step (t : ℕ) :
      Energy f star L η (x (t+1)) (u (t+1)) +
        η/2*SqMass (fun i => g (x t i)) ≤ Energy f star L η (x t) (u t) +
          η/2*(1+2*μ^2/δ)*SqMass (ξ t) := by
    have hs := momentum_descent (star := star) model hη small hμ (x t) (u t) (x (t+1)) (u (t+1))
      (d t) (ξ t) (a t) (run.attenuation t) (run.motion t) (run.direction t) (run.exchange t)
    have hc := Finset.sum_le_sum (s := Finset.univ) (fun i _ =>
      noise_cross_bound (u t i) (ξ t i) hμ (run.attenuation t i) hη hδ)
    simp only [Finset.sum_neg_distrib, Finset.sum_add_distrib, ← Finset.mul_sum] at hc
    change -(∑ i, ⟪(μ*a t i) • u t i, ξ t i⟫) ≤
      δ/(4*η)*SqMass (u t) + η*μ^2/δ*SqMass (ξ t) at hc
    have hn : 0 ≤ δ/(4*η)*SqMass (u t) :=
      mul_nonneg (by positivity) (Finset.sum_nonneg (fun i _ => sq_nonneg ‖u t i‖))
    have hh : δ/(4*η)*SqMass (u t)+δ/(4*η)*SqMass (u t) =
        (1-η*L-μ^2)/(2*η)*SqMass (u t) := by dsimp [δ]; ring
    have he : η/2*(1+2*μ^2/δ)*SqMass (ξ t) =
        η/2*SqMass (ξ t)+η*μ^2/δ*SqMass (ξ t) := by ring
    linarith only [hs, hc, hn, hh, he]
  have h := sum_descent (z := fun t => Energy f star L η (x t) (u t))
    (a := fun t => η/2*SqMass (fun i => g (x t i)))
    (b := fun t => η/2*(1+2*μ^2/δ)*SqMass (ξ t)) step T
  simp only [← Finset.mul_sum] at h
  have hb : η/2*(∑ t ∈ Finset.range T, SqMass (fun i => g (x t i))) ≤
      Energy f star L η (x 0) (u 0)+η/2*((1+2*μ^2/δ)*(∑ t ∈ Finset.range T, SqMass (ξ t))) :=
    by linarith [energy_nonneg lower hη small (x T) (u T)]
  exact normalize_sum hη (mul_pos (Nat.cast_pos.mpr hP) (Nat.cast_pos.mpr hT)) hb

theorem guarded_rates {f : E → ℝ} {g : E → E} {star L η : ℝ}
    (model : UpperModel f g L) (lower : ∀ x, star ≤ f x)
    (hη : 0 < η) (small : η*L ≤ 1) (x G candidate : ℕ → E) (ε : ℕ → ℝ)
    (motion : ∀ t, x (t+1) = x t + GuardedStep L η (G t) (candidate t))
    (err : ∀ t, ‖G t-g (x t)‖ ≤ ε t) (T : ℕ) (hT : 0 < T) :
    (∃ t < T, ‖g (x t)‖^2 ≤ 8*(f (x 0)-star)/(η*(T : ℝ)) +
      34/(T : ℝ) * (∑ s ∈ Finset.range T, (ε s)^2)) ∧
    ((∀ t, G t = g (x t)) →
      ∃ t < T, ‖g (x t)‖^2 ≤ 2*(f (x 0)-star)/(η*(T : ℝ))) := by
  have hTR : 0 < (T : ℝ) := Nat.cast_pos.mpr hT
  constructor
  · have step (t : ℕ) : (f (x (t+1))-star) + η/8*‖g (x t)‖^2 ≤
        (f (x t)-star)+(17*η/4)*(ε t)^2 := by
      rw [motion t]
      have h := guard_error_descent model hη (err t)
        (guarded_passes hη small (G t) (candidate t))
      linarith only [h]
    have h := sum_descent (z := fun t => f (x t)-star)
      (a := fun t => η/8*‖g (x t)‖^2) (b := fun t => (17*η/4)*(ε t)^2) step T
    simp only [← Finset.mul_sum] at h
    apply some_le_average hT
    have hh := lower (x T)
    have hb : (∑ t ∈ Finset.range T, ‖g (x t)‖^2) ≤
        8*(f (x 0)-star)/η + 34*(∑ t ∈ Finset.range T, (ε t)^2) := by
      apply (mul_le_mul_iff_of_pos_left hη).mp
      have he : η*(8*(f (x 0)-star)/η+34*(∑ t ∈ Finset.range T, (ε t)^2)) =
          8*(f (x 0)-star)+34*η*(∑ t ∈ Finset.range T, (ε t)^2) := by
        field_simp
        <;> ring
      rw [he]
      linarith only [h, hh]
    have he : (T : ℝ)*(8*(f (x 0)-star)/(η*(T : ℝ)) +
        34/(T : ℝ)*(∑ t ∈ Finset.range T, (ε t)^2)) =
          8*(f (x 0)-star)/η + 34*(∑ t ∈ Finset.range T, (ε t)^2) := by
      field_simp
    rwa [he]
  · intro exactGrad
    have step (t : ℕ) : (f (x (t+1))-star) + η/2*‖g (x t)‖^2 ≤
        (f (x t)-star)+0 := by
      have h := (guarded_passes hη small (G t) (candidate t)).1
      rw [exactGrad t] at h
      rw [motion t, exactGrad t]
      linarith only [h, model (x t) (GuardedStep L η (g (x t)) (candidate t))]
    have h := sum_descent (z := fun t => f (x t)-star)
      (a := fun t => η/2*‖g (x t)‖^2) (b := fun _ => 0) step T
    simp only [Finset.sum_const_zero, add_zero, ← Finset.mul_sum] at h
    apply some_le_average hT
    have hb : (∑ t ∈ Finset.range T, ‖g (x t)‖^2) ≤ 2*(f (x 0)-star)/η := by
      apply (le_div_iff₀ hη).mpr
      linarith [lower (x T)]
    have he : (T : ℝ)*(2*(f (x 0)-star)/(η*(T : ℝ))) = 2*(f (x 0)-star)/η := by
      field_simp
    rwa [he]

def Contraction (L η μ m : ℝ) : ℝ :=
  min (η*m) ((1-η*L-μ^2)/(1-η*L))

theorem contraction_pos {L η μ m : ℝ} (hη : 0 < η)
    (small : η*L < 1-μ^2) (hm : 0 < m) : 0 < Contraction L η μ m := by
  unfold Contraction
  exact lt_min (mul_pos hη hm)
    (div_pos (by linarith) (by nlinarith [sq_nonneg μ]))

theorem contraction_le_one {L η μ m : ℝ} (hη : 0 < η)
    (small : η*L < 1-μ^2) (hm : m ≤ L) : Contraction L η μ m ≤ 1 := by
  unfold Contraction
  have hh := mul_le_mul_of_nonneg_left hm hη.le
  exact min_le_left _ _ |>.trans (by nlinarith [sq_nonneg μ])

theorem momentum_PL_step {f : E → ℝ} {g : E → E} {star L η μ m : ℝ}
    (lower : ∀ x, star ≤ f x) (pl : GradDominance f g star m)
    (hη : 0 < η) (small : η*L < 1-μ^2)
    (x u : ι → E) {next error : ℝ}
    (step : next + (1-η*L-μ^2)/(2*η)*SqMass u +
      η/2*SqMass (fun i => g (x i)) ≤ Energy f star L η x u + error) :
    next ≤ (1-Contraction L η μ m)*Energy f star L η x u + error := by
  have hq : 0 < 1-η*L := by nlinarith [sq_nonneg μ]
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun i _ => pl (x i))
  simp only [← Finset.mul_sum] at hs
  change 2*m*Gap f star x ≤ SqMass (fun i => g (x i)) at hs
  have hg := mul_le_mul_of_nonneg_left hs (show 0 ≤ η/2 by positivity)
  have hn : 0 ≤ Gap f star x := Finset.sum_nonneg (fun i _ => sub_nonneg.mpr (lower (x i)))
  have ha := mul_le_mul_of_nonneg_right (min_le_left (η*m) ((1-η*L-μ^2)/(1-η*L))) hn
  change Contraction L η μ m*Gap f star x ≤ η*m*Gap f star x at ha
  have hb : Contraction L η μ m*(1-η*L) ≤ 1-η*L-μ^2 :=
    (le_div_iff₀ hq).mp (min_le_right _ _)
  have hc := div_le_div_of_nonneg_right hb (show 0 ≤ 2*η by positivity)
  have hu : 0 ≤ SqMass u := Finset.sum_nonneg (fun i _ => sq_nonneg ‖u i‖)
  have hh := mul_le_mul_of_nonneg_right hc hu
  dsimp [Energy] at step ⊢
  simp only [div_eq_mul_inv] at hh step ⊢
  nlinarith only [hg, ha, hh, step]

theorem momentum_geometric_rate {f : E → ℝ} {g : E → E} {star L η μ m : ℝ}
    (model : UpperModel f g L) (lower : ∀ x, star ≤ f x) (pl : GradDominance f g star m)
    (hη : 0 < η) (small : η*L < 1-μ^2) (hμ : 0 ≤ μ) (hm : m ≤ L)
    (x u d : ℕ → ι → E) (a : ℕ → ι → ℝ)
    (run : MomentumRun g μ η x u d (fun _ _ => 0) a) (T : ℕ) :
    Energy f star L η (x T) (u T) ≤
      (1-Contraction L η μ m)^T*Energy f star L η (x 0) (u 0) := by
  have step (t : ℕ) : Energy f star L η (x (t+1)) (u (t+1)) ≤
      (1-Contraction L η μ m)*Energy f star L η (x t) (u t) +
      Contraction L η μ m*0 := by
    have hs := momentum_descent (star := star) model hη small hμ (x t) (u t) (x (t+1)) (u (t+1))
      (d t) (fun _ => 0) (a t) (run.attenuation t) (run.motion t) (run.direction t) (run.exchange t)
    apply momentum_PL_step lower pl hη small (x t) (u t)
    simpa [SqMass] using hs
  simpa using geometric_descent (z := fun t => Energy f star L η (x t) (u t))
    (contraction_le_one hη small hm) step T

theorem guarded_geometric_rates {f : E → ℝ} {g : E → E} {star L η m : ℝ}
    (model : UpperModel f g L) (pl : GradDominance f g star m)
    (hη : 0 < η) (small : η*L ≤ 1) (hm : 0 < m) (hmL : m ≤ L)
    (x G candidate : ℕ → E)
    (motion : ∀ t, x (t+1) = x t + GuardedStep L η (G t) (candidate t)) (T : ℕ) :
    ((∀ t, G t = g (x t)) → f (x T)-star ≤ (1-η*m)^T*(f (x 0)-star)) ∧
    (∀ ε : ℝ, (∀ t, ‖G t-g (x t)‖ ≤ ε) →
      f (x T)-star ≤ (1-η*m/4)^T*(f (x 0)-star) +
        (1-(1-η*m/4)^T)*(17*ε^2/m)) := by
  have hmη := (mul_le_mul_of_nonneg_left hmL hη.le).trans small
  constructor
  · intro exactGrad
    have step (t : ℕ) : f (x (t+1))-star ≤ (1-η*m)*(f (x t)-star)+η*m*0 := by
      have h := (guarded_passes hη small (G t) (candidate t)).1
      rw [exactGrad t] at h
      rw [motion t, exactGrad t]
      have hpl := mul_le_mul_of_nonneg_left (pl (x t)) (show 0 ≤ η/2 by positivity)
      nlinarith only [h, hpl, model (x t) (GuardedStep L η (g (x t)) (candidate t))]
    simpa using geometric_descent (z := fun t => f (x t)-star) hmη step T
  · intro ε herr
    have hr : η*m/4 ≤ 1 := by linarith
    have hn : (η*m/4)*(17*ε^2/m) = (17*η/4)*ε^2 := by
      field_simp [hm.ne']
    have step (t : ℕ) : f (x (t+1))-star ≤
        (1-η*m/4)*(f (x t)-star)+(η*m/4)*(17*ε^2/m) := by
      rw [motion t, hn]
      have h := guard_error_descent model hη (herr t)
        (guarded_passes hη small (G t) (candidate t))
      have hpl := mul_le_mul_of_nonneg_left (pl (x t)) (show 0 ≤ η/8 by positivity)
      nlinarith only [h, hpl]
    exact geometric_descent (z := fun t => f (x t)-star) hr step T

section FiniteSampling

variable {Ω H : Type*} [Fintype Ω] [Fintype H]

def Mean (prob : Ω → ℝ) (value : Ω → ℝ) : ℝ :=
  ∑ w, prob w * value w

theorem history_cross_zero [DecidableEq H]
    (prob : Ω → ℝ) (hist : Ω → H) (p : H → E) (ξ : Ω → E)
    (centered : ∀ h, (∑ w, if hist w = h then prob w • ξ w else 0) = 0) :
    Mean prob (fun w => ⟪p (hist w), ξ w⟫) = 0 := by
  classical
  calc
    Mean prob (fun w => ⟪p (hist w), ξ w⟫) =
        ∑ w, ∑ h, if hist w = h then prob w * ⟪p h, ξ w⟫ else 0 := by
      unfold Mean
      apply Finset.sum_congr rfl
      intro w _
      simp
    _ = ∑ h, ⟪p h, ∑ w, if hist w = h then prob w • ξ w else 0⟫ := by
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro h _
      rw [inner_sum]
      apply Finset.sum_congr rfl
      intro w _
      split <;> simp_all [real_inner_smul_right]
    _ = 0 := by simp [centered]

theorem mean_mono {prob a b : Ω → ℝ} (hp : ∀ w, 0 ≤ prob w)
    (h : ∀ w, a w ≤ b w) : Mean prob a ≤ Mean prob b :=
  Finset.sum_le_sum (fun w _ => mul_le_mul_of_nonneg_left (h w) (hp w))

theorem mean_add (prob a b : Ω → ℝ) :
    Mean prob (fun w => a w+b w) = Mean prob a+Mean prob b := by
  simp only [Mean, mul_add, Finset.sum_add_distrib]

theorem mean_sub (prob a b : Ω → ℝ) :
    Mean prob (fun w => a w-b w) = Mean prob a-Mean prob b := by
  simp only [Mean, mul_sub, Finset.sum_sub_distrib]

theorem mean_mul (prob : Ω → ℝ) (c : ℝ) (a : Ω → ℝ) :
    Mean prob (fun w => c*a w) = c*Mean prob a := by
  simp only [Mean, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro w _
  ring

theorem mean_centered_step {prob z z' k n cross : Ω → ℝ} {b c σ : ℝ}
    (prob_nonneg : ∀ w, 0 ≤ prob w) (hc : 0 ≤ c)
    (noise : Mean prob n ≤ σ) (centered : Mean prob cross = 0)
    (step : ∀ w, z' w + c*k w ≤ b*z w+c*n w-cross w) :
    Mean prob z' + c*Mean prob k ≤ b*Mean prob z+c*σ := by
  have h := mean_mono prob_nonneg step
  simp only [mean_sub, mean_add, mean_mul, centered, sub_zero] at h
  linarith only [h, mul_le_mul_of_nonneg_left noise hc]

theorem momentum_expected_rates {f : E → ℝ} {g : E → E} {star L η μ varianceBound : ℝ}
    (model : UpperModel f g L) (lower : ∀ x, star ≤ f x)
    (hη : 0 < η) (small : η*L < 1-μ^2) (hμ : 0 ≤ μ)
    (prob : Ω → ℝ) (prob_nonneg : ∀ w, 0 ≤ prob w) (_prob_total : (∑ w, prob w) = 1)
    (x u d ξ : Ω → ℕ → ι → E) (a : Ω → ℕ → ι → ℝ)
    (run : ∀ w, MomentumRun g μ η (x w) (u w) (d w) (ξ w) (a w))
    (variance : ∀ t, Mean prob (fun w => SqMass (ξ w t)) ≤ (Fintype.card ι : ℝ)*varianceBound)
    (centered : ∀ t, Mean prob (fun w => ∑ i, ⟪(μ*a w t i) • u w t i, ξ w t i⟫) = 0)
    (T : ℕ) (hT : 0 < T) (hP : 0 < Fintype.card ι) :
    let z := fun t => Mean prob (fun w => Energy f star L η (x w t) (u w t))
    let A := fun t => Mean prob (fun w => SqMass (fun i => g (x w t i)))
    let n := (Fintype.card ι : ℝ)*(T : ℝ)
    (∑ t ∈ Finset.range T, A t)/n ≤ 2*z 0/(η*n)+varianceBound ∧
    (∀ m, 0 < m → m ≤ L → GradDominance f g star m →
      let r := Contraction L η μ m
      z T ≤ (1-r)^T*z 0 + (1-(1-r)^T)*(η*(Fintype.card ι : ℝ)*varianceBound/(2*r))) := by
  let z := fun t => Mean prob (fun w => Energy f star L η (x w t) (u w t))
  let A := fun t => Mean prob (fun w => SqMass (fun i => g (x w t i)))
  let n := (Fintype.card ι : ℝ)*(T : ℝ)
  have hmass (t : ℕ) (w : Ω) := momentum_descent (star := star) model hη small hμ
    (x w t) (u w t) (x w (t+1)) (u w (t+1)) (d w t) (ξ w t) (a w t)
    ((run w).attenuation t) ((run w).motion t) ((run w).direction t) ((run w).exchange t)
  have hc : 0 ≤ η/2 := by positivity
  have hδ : 0 ≤ (1-η*L-μ^2)/(2*η) := div_nonneg (by linarith) (by positivity)
  have hn : 0 < n := mul_pos (Nat.cast_pos.mpr hP) (Nat.cast_pos.mpr hT)
  dsimp only
  constructor
  · have step (t : ℕ) : z (t+1)+η/2*A t ≤ z t+η/2*((Fintype.card ι : ℝ)*varianceBound) := by
      have hs := mean_centered_step (b := 1)
        (z := fun w => Energy f star L η (x w t) (u w t))
        (z' := fun w => Energy f star L η (x w (t+1)) (u w (t+1)))
        (k := fun w => SqMass (fun i => g (x w t i)))
        prob_nonneg hc (variance t) (centered t)
        (fun w => by
          have hi := hmass t w
          have hn' := mul_nonneg hδ (Finset.sum_nonneg (s := Finset.univ)
            (fun i _ => sq_nonneg ‖u w t i‖))
          change 0 ≤ (1-η*L-μ^2)/(2*η)*SqMass (u w t) at hn'
          simp only [one_mul]
          linarith only [hi, hn'])
      simpa only [one_mul, z, A] using hs
    have h := sum_descent (z := z) (a := fun t => η/2*A t)
      (b := fun _ => η/2*((Fintype.card ι : ℝ)*varianceBound)) step T
    simp only [← Finset.mul_sum, Finset.sum_const, Finset.card_range, nsmul_eq_mul] at h
    have hnonneg : 0 ≤ z T := by
      unfold z Mean
      exact Finset.sum_nonneg (fun w _ =>
        mul_nonneg (prob_nonneg w) (energy_nonneg lower hη small (x w T) (u w T)))
    have hb : η/2*(∑ t ∈ Finset.range T, A t) ≤ z 0+η/2*(n*varianceBound) := by
      dsimp [n]
      nlinarith only [h, hnonneg]
    have ha := normalize_sum hη hn hb
    rw [mul_div_cancel_left₀ varianceBound hn.ne'] at ha
    exact ha
  · intro m hm hmL pl
    let r := Contraction L η μ m
    have hr : 0 < r := contraction_pos hη small hm
    have hcoef : r*(η*(Fintype.card ι : ℝ)*varianceBound/(2*r)) =
        η/2*((Fintype.card ι : ℝ)*varianceBound) := by
      field_simp [hr.ne']
    have step (t : ℕ) : z (t+1) ≤ (1-r)*z t+r*(η*(Fintype.card ι : ℝ)*varianceBound/(2*r)) := by
      have hs := mean_centered_step (b := 1-r)
        (z := fun w => Energy f star L η (x w t) (u w t))
        (z' := fun w => Energy f star L η (x w (t+1)) (u w (t+1)))
        (k := fun _ => 0) prob_nonneg hc (variance t) (centered t) (fun w => by
          have h := momentum_PL_step lower pl hη small (x w t) (u w t)
            (next := Energy f star L η (x w (t+1)) (u w (t+1)))
            (error := η/2*SqMass (ξ w t) - ∑ i, ⟪(μ*a w t i) • u w t i, ξ w t i⟫)
            (by linarith only [hmass t w])
          simpa [r, add_sub_assoc] using h)
      have hz : Mean prob (fun _ => 0) = 0 := by simp [Mean]
      simpa only [hz, mul_zero, add_zero, hcoef, z] using hs
    exact geometric_descent (z := z) (contraction_le_one hη small hmL) step T

end FiniteSampling

theorem gravity_flight_energy (v n : E) (gravity z star t : ℝ) (unit : ‖n‖ = 1) :
    ‖v-(gravity*t) • n‖^2/2 +
      gravity*(z+t*⟪v,n⟫-gravity/2*t^2-star) =
      ‖v‖^2/2 + gravity*(z-star) := by
  rw [norm_sub_sq_real, real_inner_smul_right, norm_smul, unit, mul_one, Real.norm_eq_abs, sq_abs]
  ring

theorem elastic_reflection_energy (v n : E) (unit : ‖n‖ = 1) :
    ‖v-(2*⟪v,n⟫) • n‖^2 = ‖v‖^2 := by
  rw [norm_sub_sq_real, real_inner_smul_right, norm_smul, unit, mul_one, Real.norm_eq_abs, sq_abs]
  ring

theorem gravity_damping_rate (energy K β : ℕ → ℝ) {cap : ℝ}
    (nonneg : ∀ t, 0 ≤ energy t) (kinetic : ∀ t, 0 ≤ K t)
    (cap0 : 0 ≤ cap) (cap1 : cap < 1)
    (damping : ∀ t, 0 ≤ β t ∧ β t ≤ cap)
    (balance : ∀ t, energy (t+1)+(1-(β t)^2)*K t = energy t)
    (T : ℕ) (hT : 0 < T) :
    (energy T+(∑ t ∈ Finset.range T, (1-(β t)^2)*K t) = energy 0) ∧
    (∃ t < T, K t ≤ energy 0/((T : ℝ)*(1-cap^2))) := by
  have telescopes (n : ℕ) :
      energy n+(∑ t ∈ Finset.range n, (1-(β t)^2)*K t) = energy 0 := by
    induction n with
    | zero => simp
    | succ n ih =>
      rw [Finset.sum_range_succ]
      linarith only [balance n, ih]
  refine ⟨telescopes T, some_le_average hT ?_⟩
  have hcap : 0 < 1-cap^2 := by
    nlinarith only [mul_pos (sub_pos.mpr cap1) (show 0 < 1+cap by linarith)]
  have hd (t : ℕ) : (1-cap^2)*K t ≤ (1-(β t)^2)*K t := by
    apply mul_le_mul_of_nonneg_right _ (kinetic t)
    nlinarith only [mul_nonneg (sub_nonneg.mpr (damping t).2)
      (add_nonneg cap0 (damping t).1)]
  have hb := Finset.sum_le_sum (s := Finset.range T) (fun t _ => hd t)
  simp only [← Finset.mul_sum] at hb
  have h := telescopes T
  have hc : (∑ t ∈ Finset.range T, K t) ≤ energy 0/(1-cap^2) :=
    (le_div_iff₀ hcap).mpr (by nlinarith only [hb, h, nonneg T])
  have hTR : 0 < (T : ℝ) := Nat.cast_pos.mpr hT
  have he : (T : ℝ)*(energy 0/((T : ℝ)*(1-cap^2))) = energy 0/(1-cap^2) := by
    field_simp [hTR.ne', hcap.ne']
  rwa [he]

end
end OptimizerConvergence

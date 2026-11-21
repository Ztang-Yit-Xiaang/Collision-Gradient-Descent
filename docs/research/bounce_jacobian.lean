import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.FDeriv.Pow
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

set_option autoImplicit false
open Filter
open scoped Topology

namespace BounceJacobian

abbrev Phase := ℝ × ℝ
abbrev dx : Phase →L[ℝ] ℝ := ContinuousLinearMap.fst ℝ ℝ ℝ
abbrev du : Phase →L[ℝ] ℝ := ContinuousLinearMap.snd ℝ ℝ ℝ

noncomputable def wall (κ x : ℝ) : ℝ := κ * x^2 / 2

theorem wall_deriv (κ x : ℝ) : HasDerivAt (wall κ) (κ * x) x := by
  convert ((hasDerivAt_id x).pow 2).const_mul (κ / 2) using 1
  · funext y
    simp [wall, Pi.pow_apply]
    ring
  · simp
    ring

noncomputable def radicand (g κ v : ℝ) (p : Phase) : ℝ :=
  v^2 - g * κ * p.1^2 - p.2^2

noncomputable def launchSpeed (g κ v : ℝ) (p : Phase) : ℝ :=
  Real.sqrt (radicand g κ v p)

noncomputable def flightNumerator (g κ v : ℝ) (p : Phase) : ℝ :=
  launchSpeed g κ v p - κ * p.1 * p.2

noncomputable def flightDenominator (g κ : ℝ) (p : Phase) : ℝ :=
  g + κ * p.2^2

noncomputable def flightTime (g κ v : ℝ) (p : Phase) : ℝ :=
  2 * flightNumerator g κ v p / flightDenominator g κ p

noncomputable def contactX (g κ v : ℝ) (p : Phase) : ℝ :=
  p.1 + flightTime g κ v p * p.2

noncomputable def incomingZ (g κ v : ℝ) (p : Phase) : ℝ :=
  launchSpeed g κ v p - g * flightTime g κ v p

noncomputable def slope (g κ v : ℝ) (p : Phase) : ℝ :=
  κ * contactX g κ v p

noncomputable def gap (g κ v : ℝ) (p : Phase) (t : ℝ) : ℝ :=
  wall κ p.1 + launchSpeed g κ v p * t - g * t^2 / 2 - wall κ (p.1 + t * p.2)

noncomputable def horizontalReflection (s u w : ℝ) : ℝ :=
  let n₁ := -s / Real.sqrt (1 + s^2)
  let n₂ := 1 / Real.sqrt (1 + s^2)
  u - 2 * (u * n₁ + w * n₂) * n₁

noncomputable def returnMap (g κ v : ℝ) (p : Phase) : Phase :=
  (contactX g κ v p, horizontalReflection (slope g κ v p) p.2 (incomingZ g κ v p))

noncomputable def flightLinear (g v : ℝ) : Phase →L[ℝ] ℝ :=
  dx + (2 * v / g) • du

noncomputable def jacobian (g κ v : ℝ) : Phase →L[ℝ] Phase :=
  (flightLinear g v).prod (du - (2 * v * κ) • flightLinear g v)

private theorem dx_at_zero : HasFDerivAt (@Prod.fst ℝ ℝ) dx 0 :=
  dx.hasFDerivAt

private theorem du_at_zero : HasFDerivAt (@Prod.snd ℝ ℝ) du 0 :=
  du.hasFDerivAt

private theorem quot_at_zero {f d : Phase → ℝ}
    (hf : HasFDerivAt f (0 : Phase →L[ℝ] ℝ) 0) (hd : HasFDerivAt d (0 : Phase →L[ℝ] ℝ) 0) (hne : d 0 ≠ 0) :
    HasFDerivAt (fun p => f p / d p) (0 : Phase →L[ℝ] ℝ) 0 := by
  have hi : HasFDerivAt (fun p => (d p)⁻¹) (0 : Phase →L[ℝ] ℝ) 0 := by
    simpa [Function.comp_def] using (hasDerivAt_inv hne).comp_hasFDerivAt (0 : Phase) hd
  simpa [div_eq_mul_inv, Pi.mul_def] using hf.mul hi

private theorem radicand_deriv (g κ v : ℝ) :
    HasFDerivAt (radicand g κ v) (0 : Phase →L[ℝ] ℝ) 0 := by
  have hx2 : HasFDerivAt (fun p : Phase => p.1^2) (0 : Phase →L[ℝ] ℝ) 0 := by
    simpa using dx_at_zero.pow 2
  have hu2 : HasFDerivAt (fun p : Phase => p.2^2) (0 : Phase →L[ℝ] ℝ) 0 := by
    simpa using du_at_zero.pow 2
  change HasFDerivAt (fun p : Phase => v^2 - g * κ * p.1^2 - p.2^2) _ _
  simpa [Pi.sub_def] using
    ((hasFDerivAt_const (v^2) (0 : Phase)).sub (hx2.const_mul (g*κ))).sub hu2

@[simp] private theorem launch_zero (g κ v : ℝ) (hv : 0 < v) :
    launchSpeed g κ v 0 = v := by
  simp [launchSpeed, radicand, Real.sqrt_sq hv.le]

theorem launch_deriv (g κ v : ℝ) (hv : 0 < v) :
    HasFDerivAt (launchSpeed g κ v) (0 : Phase →L[ℝ] ℝ) 0 := by
  have h : radicand g κ v 0 ≠ 0 := by simp [radicand, hv.ne']
  change HasFDerivAt (fun p => Real.sqrt (radicand g κ v p)) _ _
  simpa using (radicand_deriv g κ v).sqrt h

private theorem numerator_deriv (g κ v : ℝ) (hv : 0 < v) :
    HasFDerivAt (flightNumerator g κ v) (0 : Phase →L[ℝ] ℝ) 0 := by
  have hx : HasFDerivAt (fun p : Phase => κ * p.1) (κ • dx) 0 :=
    dx_at_zero.const_mul κ
  have hxu : HasFDerivAt (fun p : Phase => κ * p.1 * p.2) (0 : Phase →L[ℝ] ℝ) 0 := by
    simpa [Pi.mul_def] using hx.mul du_at_zero
  change HasFDerivAt (fun p : Phase => launchSpeed g κ v p - κ * p.1 * p.2) _ _
  simpa [Pi.sub_def] using (launch_deriv g κ v hv).sub hxu

private theorem denominator_deriv (g κ : ℝ) : HasFDerivAt (flightDenominator g κ) (0 : Phase →L[ℝ] ℝ) 0 := by
  have h : HasFDerivAt (fun p : Phase => p.2^2) (0 : Phase →L[ℝ] ℝ) 0 := by
    simpa using du_at_zero.pow 2
  change HasFDerivAt (fun p : Phase => g + κ * p.2^2) _ _
  simpa [Pi.add_def] using (hasFDerivAt_const g (0 : Phase)).add (h.const_mul κ)

theorem flight_deriv (g κ v : ℝ) (hg : 0 < g) (hv : 0 < v) :
    HasFDerivAt (flightTime g κ v) (0 : Phase →L[ℝ] ℝ) 0 := by
  have hn : HasFDerivAt (fun p => 2 * flightNumerator g κ v p) (0 : Phase →L[ℝ] ℝ) 0 := by
    simpa using (numerator_deriv g κ v hv).const_mul (2 : ℝ)
  exact quot_at_zero hn (denominator_deriv g κ) (by simp [flightDenominator, hg.ne'])

@[simp] theorem flight_zero (g κ v : ℝ) (hv : 0 < v) :
    flightTime g κ v 0 = 2 * v / g := by
  simp [flightTime, flightNumerator, flightDenominator, launch_zero g κ v hv]

theorem gap_factor (g κ v : ℝ) (p : Phase) (t : ℝ) :
    gap g κ v p t =
      t * (flightNumerator g κ v p - flightDenominator g κ p * t / 2) := by
  unfold gap wall flightNumerator flightDenominator
  ring

/-- In a neighborhood of the vertical orbit the formula selects the first positive contact
    and the velocity there points into the wall. -/
theorem first_contact (g κ v : ℝ) (hg : 0 < g) (hv : 0 < v) :
    ∀ᶠ p in 𝓝 (0 : Phase),
      0 < radicand g κ v p ∧
      0 < flightTime g κ v p ∧
      gap g κ v p (flightTime g κ v p) = 0 ∧
      (∀ t : ℝ, 0 < t → t < flightTime g κ v p → 0 < gap g κ v p t) ∧
      incomingZ g κ v p - slope g κ v p * p.2 < 0 := by
  have hr : ∀ᶠ p in 𝓝 (0 : Phase), 0 < radicand g κ v p :=
    continuousAt_const.eventually_lt (radicand_deriv g κ v).continuousAt
      (by simpa [radicand] using pow_pos hv 2)
  have hn : ∀ᶠ p in 𝓝 (0 : Phase), 0 < flightNumerator g κ v p :=
    continuousAt_const.eventually_lt (numerator_deriv g κ v hv).continuousAt
      (by simpa [flightNumerator, launch_zero g κ v hv] using hv)
  have hd : ∀ᶠ p in 𝓝 (0 : Phase), 0 < flightDenominator g κ p :=
    continuousAt_const.eventually_lt (denominator_deriv g κ).continuousAt
      (by simpa [flightDenominator] using hg)
  filter_upwards [hr, hn, hd] with p hp hn hd
  have ht : 0 < flightTime g κ v p := div_pos (by positivity) hd
  have he : flightDenominator g κ p * flightTime g κ v p =
      2 * flightNumerator g κ v p := by
    unfold flightTime
    field_simp [hd.ne']
  refine ⟨hp, ht, ?_, ?_, ?_⟩
  · rw [gap_factor, he]
    ring
  · intro t ht0 htlt
    rw [gap_factor]
    apply mul_pos ht0
    linarith [mul_lt_mul_of_pos_left htlt hd]
  · have hnormal :
        incomingZ g κ v p - slope g κ v p * p.2 =
          flightNumerator g κ v p - flightDenominator g κ p * flightTime g κ v p := by
      unfold incomingZ slope contactX flightNumerator flightDenominator
      ring
    rw [hnormal, he]
    linarith

theorem reflection_rational (s u w : ℝ) :
    horizontalReflection s u w = u + s * (2 * (w - s*u) / (1 + s^2)) := by
  let r := Real.sqrt (1 + s^2)
  have hp : 0 < 1 + s^2 := by positivity
  have hr : 0 < r := Real.sqrt_pos.2 hp
  have hrsq : r^2 = 1 + s^2 := Real.sq_sqrt hp.le
  change u - 2 * (u * (-s/r) + w * (1/r)) * (-s/r) = _
  calc
    _ = u + s * (2 * (w - s*u) / r^2) := by field_simp [hr.ne']; ring
    _ = _ := by rw [hrsq]

theorem contact_deriv (g κ v : ℝ) (hg : 0 < g) (hv : 0 < v) :
    HasFDerivAt (contactX g κ v) (flightLinear g v) 0 := by
  change HasFDerivAt (fun p : Phase => p.1 + flightTime g κ v p * p.2) _ _
  simpa [Pi.add_def, Pi.mul_def, flightLinear, flight_zero g κ v hv] using
    dx_at_zero.add ((flight_deriv g κ v hg hv).mul du_at_zero)

private theorem contact_zero (g κ v : ℝ) : contactX g κ v 0 = 0 := by
  simp [contactX]

theorem incoming_deriv (g κ v : ℝ) (hg : 0 < g) (hv : 0 < v) :
    HasFDerivAt (incomingZ g κ v) (0 : Phase →L[ℝ] ℝ) 0 := by
  change HasFDerivAt (fun p => launchSpeed g κ v p - g * flightTime g κ v p) _ _
  simpa [Pi.sub_def] using
    (launch_deriv g κ v hv).sub ((flight_deriv g κ v hg hv).const_mul g)

theorem incoming_zero (g κ v : ℝ) (hg : 0 < g) (hv : 0 < v) :
    incomingZ g κ v 0 = -v := by
  simp [incomingZ, launch_zero g κ v hv, flight_zero g κ v hv]
  field_simp [hg.ne']
  ring

private noncomputable def kickFactor (g κ v : ℝ) (p : Phase) : ℝ :=
  2 * (incomingZ g κ v p - slope g κ v p * p.2) / (1 + (slope g κ v p)^2)

theorem return_zero (g κ v : ℝ) : returnMap g κ v 0 = 0 := by
  apply Prod.ext <;> simp [returnMap, reflection_rational, slope, contact_zero]

theorem return_deriv (g κ v : ℝ) (hg : 0 < g) (hv : 0 < v) :
    HasFDerivAt (returnMap g κ v) (jacobian g κ v) 0 := by
  have hs : HasFDerivAt (slope g κ v) (κ • flightLinear g v) 0 :=
    (contact_deriv g κ v hg hv).const_mul κ
  have hs0 : slope g κ v 0 = 0 := by simp [slope, contact_zero]
  have hsu : HasFDerivAt (fun p : Phase => slope g κ v p * p.2) (0 : Phase →L[ℝ] ℝ) 0 := by
    simpa [Pi.mul_def, hs0] using hs.mul du_at_zero
  have hnum : HasFDerivAt (fun p : Phase => 2 * (incomingZ g κ v p - slope g κ v p * p.2)) (0 : Phase →L[ℝ] ℝ) 0 := by
    simpa [Pi.sub_def] using ((incoming_deriv g κ v hg hv).sub hsu).const_mul (2 : ℝ)
  have hden : HasFDerivAt (fun p => 1 + (slope g κ v p)^2) (0 : Phase →L[ℝ] ℝ) 0 := by
    have hs2 : HasFDerivAt (fun p => (slope g κ v p)^2) (0 : Phase →L[ℝ] ℝ) 0 := by
      simpa [hs0] using hs.pow 2
    simpa [Pi.add_def] using (hasFDerivAt_const (1 : ℝ) (0 : Phase)).add hs2
  have hk : HasFDerivAt (kickFactor g κ v) (0 : Phase →L[ℝ] ℝ) 0 := quot_at_zero hnum hden (by simp [hs0])
  have hk0 : kickFactor g κ v 0 = -(2*v) := by simp [kickFactor, hs0, incoming_zero g κ v hg hv]
  have hu : HasFDerivAt (fun p : Phase => p.2 + slope g κ v p * kickFactor g κ v p)
      (du - (2*v*κ) • flightLinear g v) 0 := by
    simpa [Pi.add_def, Pi.mul_def, hs0, hk0, smul_smul, neg_mul, sub_eq_add_neg] using
      du_at_zero.add (hs.mul hk)
  have heq : returnMap g κ v =
      fun p => (contactX g κ v p, p.2 + slope g κ v p * kickFactor g κ v p) := by
    funext p
    simp [returnMap, reflection_rational, kickFactor]
  rw [heq]
  exact (contact_deriv g κ v hg hv).prodMk hu

theorem jacobian_apply (g κ v : ℝ) (p : Phase) :
    let T := 2*v/g
    jacobian g κ v p = (p.1 + T*p.2, -(2*v*κ)*p.1 + (1-2*v*κ*T)*p.2) := by
  apply Prod.ext
  · simp [jacobian, flightLinear]
  · simp [jacobian, flightLinear]
    ring

theorem physical_radicand (g κ H : ℝ) (hg : 0 < g) (hH : 0 < H) (p : Phase) :
    radicand g κ (Real.sqrt (2*g*H)) p = 2*g*(H - wall κ p.1) - p.2^2 := by
  rw [radicand, Real.sq_sqrt (by positivity)]
  unfold wall
  ring

theorem physical_return_deriv (g κ H : ℝ) (hg : 0 < g) (hH : 0 < H) :
    let v := Real.sqrt (2*g*H)
    HasFDerivAt (returnMap g κ v) (jacobian g κ v) 0 :=
  return_deriv g κ _ hg (Real.sqrt_pos.2 (by positivity))

theorem physical_fderiv_apply (g κ H : ℝ) (hg : 0 < g) (hH : 0 < H) (p : Phase) :
    let v := Real.sqrt (2*g*H)
    let T := 2*v/g
    fderiv ℝ (returnMap g κ v) 0 p =
      (p.1 + T*p.2, -(2*v*κ)*p.1 + (1-2*v*κ*T)*p.2) := by
  dsimp only
  rw [(physical_return_deriv g κ H hg hH).fderiv]
  exact jacobian_apply g κ _ p

end BounceJacobian

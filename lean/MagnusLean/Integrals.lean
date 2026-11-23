/-
Copyright (c) 2026 Pragyaan Gaur. All rights reserved.
Released under the MIT license as described in the file LICENSE.
-/
import MagnusLean.Vertical
import Mathlib.MeasureTheory.Integral.IntervalIntegral.IntegrationByParts

/-!
# Closure and chord as integrals over the heading

## i. Overview

The paper writes the height change and horizontal displacement as integrals over the
heading `ψ` rather than over time (Eqs. (closure) and (chord)):
`z(Ψ) - z(0) = L_L ∫₀^Ψ sin γ dψ`, `x(Ψ) - x(0) = L_L ∫₀^Ψ cos γ cos ψ dψ`,
`y(Ψ) - y(0) = L_L ∫₀^Ψ cos γ sin ψ dψ`.

Here we prove these by substituting `ψ = κ_L s(t)` in the time integrals. We assume the
climb angle is a continuous function of the heading, `sin γ(t) = S(ψ(t))` and
`cos γ(t) = C(ψ(t))`, which is how the paper uses `γ(ψ)`. No monotonicity of `ψ` is needed
for the substitution formula itself. For the chord we take the launch along `+x`, so
`v_y(0) = 0` and `v_x(0) > 0`, as in the paper.
-/

open Complex MeasureTheory intervalIntegral

namespace MagnusFlight3

variable (F : MagnusFlight3)

lemma continuous_vx : Continuous F.vx :=
  continuous_iff_continuousAt.mpr fun t => (F.hvx t).continuousAt
lemma continuous_vy : Continuous F.vy :=
  continuous_iff_continuousAt.mpr fun t => (F.hvy t).continuousAt
lemma continuous_vz : Continuous F.vz :=
  continuous_iff_continuousAt.mpr fun t => (F.hvz t).continuousAt

lemma continuous_speed : Continuous F.speed := by
  have : F.speed = fun t => √(F.vx t ^ 2 + F.vy t ^ 2 + F.vz t ^ 2) := funext F.speed_eq
  rw [this]
  exact (((F.continuous_vx.pow 2).add (F.continuous_vy.pow 2)).add
    (F.continuous_vz.pow 2)).sqrt

lemma hasDerivAt_ψ (t : ℝ) : HasDerivAt F.ψ (F.κL * F.speed t) t :=
  (F.hs t).const_mul F.κL

lemma ψ_zero : F.ψ 0 = 0 := by simp [ψ, F.s_zero]

/-- Generic substitution: if `w(t) = L_L · P(ψ(t)) · dψ/dt` with `P` continuous, then
`∫₀^T w dt = L_L ∫₀^{ψ(T)} P dψ`. -/
lemma integral_subst (_hL : F.κL ≠ 0) {P : ℝ → ℝ} (hP : Continuous P) (T : ℝ) :
    ∫ t in (0 : ℝ)..T, (1 / F.κL) * P (F.ψ t) * (F.κL * F.speed t) =
      (1 / F.κL) * ∫ u in (0 : ℝ)..F.ψ T, P u := by
  have h := integral_comp_mul_deriv (a := 0) (b := T) (f := F.ψ)
    (f' := fun t => F.κL * F.speed t) (g := P) (fun t _ => F.hasDerivAt_ψ t)
    ((continuous_const.mul F.continuous_speed).continuousOn) hP
  rw [F.ψ_zero] at h
  rw [← h, ← intervalIntegral.integral_const_mul]
  congr 1; funext t; simp only [Function.comp]; ring

/-- Eq. (closure): `z(T) - z(0) = L_L ∫₀^{ψ(T)} sin γ dψ`. The ball is back at launch
height after the turn `Ψ = ψ(T)` exactly when `∫₀^Ψ sin γ dψ = 0`. -/
theorem closure (hL : F.κL ≠ 0) (h0 : F.zeta 0 ≠ 0) {z : ℝ → ℝ}
    (hz : ∀ t, HasDerivAt z (F.vz t) t) {S : ℝ → ℝ} (hS : Continuous S)
    (hγ : ∀ t, F.vz t / F.speed t = S (F.ψ t)) (T : ℝ) :
    z T - z 0 = (1 / F.κL) * ∫ u in (0 : ℝ)..F.ψ T, S u := by
  rw [← F.integral_subst hL hS T, ← integral_eq_sub_of_hasDerivAt (fun t _ => hz t)
    (F.continuous_vz.intervalIntegrable _ _)]
  congr 1; funext t
  rw [← hγ t]
  exact F.height_rate hL h0 t

/-- With launch along `+x`, the horizontal velocity is `v_h (cos ψ, sin ψ)`. -/
lemma vx_vy_of_launch_x (hy0 : F.vy 0 = 0) (hx0 : 0 < F.vx 0) (t : ℝ) :
    F.vx t = F.vh t * Real.cos (F.ψ t) ∧ F.vy t = F.vh t * Real.sin (F.ψ t) := by
  have h := F.zeta_eq_decay_rotation t
  have hz0 : F.zeta 0 = (F.vx 0 : ℂ) := by simp [MagnusFlight.zeta, hy0]
  have hn : ‖F.zeta 0‖ = F.vx 0 := by rw [hz0, norm_real, Real.norm_eq_abs, abs_of_pos hx0]
  rw [hz0, exp_mul_I, ← ofReal_cos, ← ofReal_sin] at h
  have hre := congrArg re h
  have him := congrArg im h
  simp only [MagnusFlight.zeta, add_re, add_im, ofReal_re, ofReal_im, mul_re, mul_im, I_re,
    I_im] at hre him
  simp only [mul_zero, sub_zero, zero_mul, add_zero, mul_one, zero_add] at hre him
  refine ⟨?_, ?_⟩
  · rw [hre, vh, hn, ψ]
  · rw [him, vh, hn, ψ]

/-- Real part of Eq. (chord): `x(T) - x(0) = L_L ∫₀^{ψ(T)} cos γ cos ψ dψ`. -/
theorem chord_x (hL : F.κL ≠ 0) (h0 : F.zeta 0 ≠ 0) (hy0 : F.vy 0 = 0) (hx0 : 0 < F.vx 0)
    {x : ℝ → ℝ} (hx : ∀ t, HasDerivAt x (F.vx t) t) {C : ℝ → ℝ} (hC : Continuous C)
    (hγ : ∀ t, F.vh t / F.speed t = C (F.ψ t)) (T : ℝ) :
    x T - x 0 = (1 / F.κL) * ∫ u in (0 : ℝ)..F.ψ T, C u * Real.cos u := by
  rw [← F.integral_subst (P := fun u => C u * Real.cos u) hL (hC.mul Real.continuous_cos) T,
    ← integral_eq_sub_of_hasDerivAt (fun t _ => hx t) (F.continuous_vx.intervalIntegrable _ _)]
  congr 1; funext t
  rw [← hγ t, (F.vx_vy_of_launch_x hy0 hx0 t).1]
  have := (F.speed_pos h0 t).ne'
  field_simp

/-- Imaginary part of Eq. (chord): `y(T) - y(0) = L_L ∫₀^{ψ(T)} cos γ sin ψ dψ`. -/
theorem chord_y (hL : F.κL ≠ 0) (h0 : F.zeta 0 ≠ 0) (hy0 : F.vy 0 = 0) (hx0 : 0 < F.vx 0)
    {y : ℝ → ℝ} (hy : ∀ t, HasDerivAt y (F.vy t) t) {C : ℝ → ℝ} (hC : Continuous C)
    (hγ : ∀ t, F.vh t / F.speed t = C (F.ψ t)) (T : ℝ) :
    y T - y 0 = (1 / F.κL) * ∫ u in (0 : ℝ)..F.ψ T, C u * Real.sin u := by
  rw [← F.integral_subst (P := fun u => C u * Real.sin u) hL (hC.mul Real.continuous_sin) T,
    ← integral_eq_sub_of_hasDerivAt (fun t _ => hy t) (F.continuous_vy.intervalIntegrable _ _)]
  congr 1; funext t
  rw [← hγ t, (F.vx_vy_of_launch_x hy0 hx0 t).2]
  have := (F.speed_pos h0 t).ne'
  field_simp

end MagnusFlight3

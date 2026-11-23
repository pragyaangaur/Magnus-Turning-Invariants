/-
Copyright (c) 2026 Pragyaan Gaur. All rights reserved.
Released under the MIT license as described in the file LICENSE.
-/
import Mathlib.Analysis.SpecialFunctions.Complex.Log
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.MeanValue

/-!
# Horizontal motion of a ball with quadratic drag and a vertical-axis Magnus force

## i. Overview

A ball moves under gravity, quadratic drag and a Magnus force whose spin axis is
vertical (`ω̂ = +ẑ`). Newton's second law reads
`dv/dt = -g ẑ - κ_D |v| v + κ_L |v| (ẑ × v)`.
Gravity is vertical and `ẑ × v = (-v_y, v_x, 0)` is horizontal, so the horizontal
components are
`dv_x/dt = -κ_D |v| v_x - κ_L |v| v_y`,
`dv_y/dt = -κ_D |v| v_y + κ_L |v| v_x`.
Write `ζ = v_x + i v_y` and let `s` be the arc length, so `ds/dt = |v|`.
The main theorem is that, for every time `t`,
`ζ(t) = ζ(0) exp((i κ_L - κ_D) s(t))`,
however the vertical motion behaves. This is Eq. (master) of the paper. The proof never
divides by `|v|`, so it does not need the speed to stay positive and does not change the
independent variable from `t` to `s`. It shows `ζ(t) exp(-(i κ_L - κ_D) s(t))` has zero
time derivative.

Consequences proved here: the horizontal speed decays as `exp(-κ_D s)`, the heading turns
by exactly `κ_L s` (as a continuous angle), the horizontal velocity lies on a logarithmic
spiral, a reversal of heading needs path length `(2k+1) π / κ_L`, and the ground track has
signed curvature `κ_L |v| / v_h`, i.e. radius `L_L cos γ`.

## ii. Key results

- `MagnusFlight.zeta_eq` : the exact solution for the horizontal velocity.
- `MagnusFlight.norm_zeta` : horizontal speed `v_h = v_h(0) exp(-κ_D s)`.
- `MagnusFlight.reversal_path_length` : turning by `π` needs `κ_L s = π + 2πk`.
- `MagnusFlight.track_curvature` : `Im(conj ζ · dζ/dt) = κ_L |v| v_h²`.

## iii. References

- P. Gaur, *Exact turning invariants for projectile motion under quadratic drag and a
  vertical-axis Magnus force*, arXiv:2609.30305.
-/

open Complex

/-- A trajectory of a ball under gravity, quadratic drag and a Magnus force about `+ẑ`.
Only the horizontal equations and the definition of arc length are recorded, because the
vertical equation is not needed for the horizontal results. `speed t` stands for `|v(t)|`
and is left abstract, so the results hold whatever the vertical velocity does. -/
structure MagnusFlight where
  /-- Lift inverse length `κ_L = ρ C_L A / (2m)`. -/
  κL : ℝ
  /-- Drag inverse length `κ_D = ρ C_D A / (2m)`. -/
  κD : ℝ
  /-- The `x` component of velocity. -/
  vx : ℝ → ℝ
  /-- The `y` component of velocity. -/
  vy : ℝ → ℝ
  /-- The total speed `|v|`. -/
  speed : ℝ → ℝ
  /-- The arc length travelled. -/
  s : ℝ → ℝ
  s_zero : s 0 = 0
  hs : ∀ t, HasDerivAt s (speed t) t
  hvx : ∀ t, HasDerivAt vx (-κD * speed t * vx t - κL * speed t * vy t) t
  hvy : ∀ t, HasDerivAt vy (-κD * speed t * vy t + κL * speed t * vx t) t

namespace MagnusFlight

variable (F : MagnusFlight)

/-- The horizontal velocity as a complex number, `ζ = v_x + i v_y`. -/
noncomputable def zeta (t : ℝ) : ℂ := (F.vx t : ℂ) + (F.vy t : ℂ) * I

/-- The complex rate `i κ_L - κ_D`. -/
noncomputable def rate : ℂ := (F.κL : ℂ) * I - (F.κD : ℂ)

/-- `dζ/dt = (i κ_L - κ_D) |v| ζ`, Eq. (zdot) of the paper. -/
lemma hasDerivAt_zeta (t : ℝ) :
    HasDerivAt F.zeta (F.rate * (F.speed t : ℂ) * F.zeta t) t := by
  have hx := (F.hvx t).ofReal_comp
  have hy := ((F.hvy t).ofReal_comp).mul_const I
  convert hx.add hy using 1
  · rfl
  unfold rate zeta
  push_cast
  linear_combination (F.κL * F.speed t * F.vy t : ℂ) * I_sq

/-- The exact horizontal solution `ζ(t) = ζ(0) exp((i κ_L - κ_D) s(t))`. -/
theorem zeta_eq (t : ℝ) :
    F.zeta t = F.zeta 0 * exp (F.rate * (F.s t : ℂ)) := by
  set f : ℝ → ℂ := fun u => F.zeta u * exp (-(F.rate * (F.s u : ℂ))) with hf
  have hderiv : ∀ u, HasDerivAt f 0 u := by
    intro u
    have hs' : HasDerivAt (fun u => -(F.rate * (F.s u : ℂ))) (-(F.rate * (F.speed u : ℂ))) u :=
      (((F.hs u).ofReal_comp).const_mul F.rate).neg
    convert (F.hasDerivAt_zeta u).mul hs'.cexp using 1
    ring
  have hconst : f t = f 0 :=
    is_const_of_deriv_eq_zero (fun u => (hderiv u).differentiableAt)
      (fun u => (hderiv u).deriv) t 0
  simp only [hf, F.s_zero, ofReal_zero, mul_zero, neg_zero, exp_zero, mul_one] at hconst
  calc F.zeta t = F.zeta t * exp (-(F.rate * (F.s t : ℂ))) * exp (F.rate * (F.s t : ℂ)) := by
        rw [mul_assoc, ← exp_add, neg_add_cancel, exp_zero, mul_one]
    _ = F.zeta 0 * exp (F.rate * (F.s t : ℂ)) := by rw [hconst]

/-- The solution split into a decay factor and a pure rotation by the angle `κ_L s`.
The heading has turned by exactly `κ_L s(t)`, independent of gravity, launch conditions
and `C_D`. -/
theorem zeta_eq_decay_rotation (t : ℝ) :
    F.zeta t = F.zeta 0 * (Real.exp (-F.κD * F.s t) : ℂ) *
      exp (((F.κL * F.s t : ℝ) : ℂ) * I) := by
  rw [F.zeta_eq, mul_assoc, ofReal_exp, ← exp_add]
  congr 2
  simp only [rate, ofReal_mul, ofReal_neg]
  ring

/-- The horizontal speed decays as `v_h(t) = v_h(0) exp(-κ_D s(t))`. -/
theorem norm_zeta (t : ℝ) :
    ‖F.zeta t‖ = ‖F.zeta 0‖ * Real.exp (-F.κD * F.s t) := by
  rw [F.zeta_eq_decay_rotation, norm_mul, norm_mul, norm_exp_ofReal_mul_I, mul_one,
    norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]

/-- The logarithmic spiral: with `ψ = κ_L s` the heading turned and `μ = κ_D / κ_L`,
`v_h = v_h(0) exp(-μ ψ)`. -/
theorem log_spiral (hL : F.κL ≠ 0) (t : ℝ) :
    ‖F.zeta t‖ = ‖F.zeta 0‖ * Real.exp (-(F.κD / F.κL) * (F.κL * F.s t)) := by
  rw [F.norm_zeta]
  congr 2
  field_simp

/-- A reversal of heading needs a fixed path length. If at time `t` the horizontal
velocity points exactly opposite to its launch direction (a negative real multiple of
`ζ(0) ≠ 0`), then `κ_L s(t) = π + 2πk` for an integer `k`. With `L_L = 1/κ_L` and `k = 0`
this is the invariant `S = π L_L`. -/
theorem reversal_path_length (h0 : F.zeta 0 ≠ 0) (t : ℝ) (r : ℝ) (hr : 0 < r)
    (hrev : F.zeta t = -(r : ℂ) * F.zeta 0) :
    ∃ k : ℤ, F.κL * F.s t = Real.pi + 2 * Real.pi * k := by
  set θ : ℝ := F.κL * F.s t
  set e : ℝ := Real.exp (-F.κD * F.s t)
  have he : 0 < e := Real.exp_pos _
  have key : (e : ℂ) * exp ((θ : ℂ) * I) = -(r : ℂ) := by
    have := F.zeta_eq_decay_rotation t
    rw [hrev] at this
    apply mul_left_cancel₀ h0
    linear_combination -this
  -- take norms: e = r, so exp(iθ) = -1
  have hnorm : e = r := by
    have := congrArg norm key
    rw [norm_mul, norm_exp_ofReal_mul_I, mul_one, norm_neg, norm_real, norm_real,
      Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos he, abs_of_pos hr] at this
    exact this
  have hm1 : exp ((θ : ℂ) * I) = -1 := by
    rw [hnorm] at key
    have hr0 : (r : ℂ) ≠ 0 := by exact_mod_cast hr.ne'
    apply mul_left_cancel₀ hr0
    rw [key]; ring
  have h1 : exp (((θ - Real.pi : ℝ) : ℂ) * I) = 1 := by
    rw [ofReal_sub, sub_mul, exp_sub, hm1, exp_pi_mul_I]
    norm_num
  obtain ⟨n, hn⟩ := exp_eq_one_iff.mp h1
  refine ⟨n, ?_⟩
  have := congrArg im hn
  simp at this
  linarith

/-- The ground track bends at the rate set by `κ_L` alone. For a plane curve with velocity
`ζ`, `Im(conj ζ · dζ/dt) / |ζ|³` is its signed curvature; here it equals `κ_L |v| / v_h`,
so the radius of curvature is `v_h / (κ_L |v|) = L_L cos γ`, Eq. (curvature). -/
theorem track_curvature (t : ℝ) :
    (starRingEnd ℂ (F.zeta t) * (F.rate * (F.speed t : ℂ) * F.zeta t)).im =
      F.κL * F.speed t * normSq (F.zeta t) := by
  simp only [rate, zeta, normSq_apply, mul_im, mul_re, conj_re, conj_im, add_re, add_im,
    ofReal_re, ofReal_im, mul_re, mul_im, I_re, I_im, sub_re, sub_im]
  ring

end MagnusFlight

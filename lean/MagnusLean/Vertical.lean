/-
Copyright (c) 2026 Pragyaan Gaur. All rights reserved.
Released under the MIT license as described in the file LICENSE.
-/
import MagnusLean.Horizontal
import Mathlib.Analysis.Calculus.Deriv.Inv

/-!
# Vertical motion and the reduced equation

## i. Overview

The vertical part of Newton's law is `dv_z/dt = -g - κ_D |v| v_z`, because the Magnus force
is horizontal. Write `v_h = |ζ|` for the horizontal speed, `Q = tan γ = v_z / v_h`,
`ψ = κ_L s` for the heading turned, `μ = κ_D / κ_L` and `λ = g / (κ_L v_h(0)²)`
(this is `g L_L / (v_0 cos θ)²`). The paper's Eq. (reduced) says
`dQ/dψ = -λ e^{2μψ} / √(1 + Q²)`.

We prove it in time-derivative form, `dQ/dt = (-λ e^{2μψ} / √(1+Q²)) · dψ/dt`, which is
the chain rule version of Eq. (reduced) and needs no change of independent variable.
On the way we prove the simpler identity `dQ/dt = -g / v_h`.

We also prove the rate forms of Eqs. (closure) and (chord): `dz/dt = L_L sin γ · dψ/dt`
and `d(x + i y)/dt = L_L cos γ e^{iψ} u₀ · dψ/dt`, where `u₀ = ζ(0)/|ζ(0)|` is the launch
direction.

## ii. Key results

- `MagnusFlight3.hasDerivAt_Q` : `dQ/dt = -g / v_h`.
- `MagnusFlight3.reduced` : Eq. (reduced) in chain-rule form.
- `MagnusFlight3.chord_rate` : Eq. (chord) in rate form.
-/

open Complex

/-- A full three-dimensional flight: the horizontal data of `MagnusFlight` plus the vertical
velocity, gravity and the definition of the speed. -/
structure MagnusFlight3 extends MagnusFlight where
  /-- Gravitational acceleration. -/
  g : ℝ
  /-- The vertical velocity. -/
  vz : ℝ → ℝ
  hvz : ∀ t, HasDerivAt vz (-g - κD * speed t * vz t) t
  speed_eq : ∀ t, speed t = √(vx t ^ 2 + vy t ^ 2 + vz t ^ 2)

namespace MagnusFlight3

variable (F : MagnusFlight3)

/-- Horizontal speed, `v_h(t) = v_h(0) e^{-κ_D s(t)}` (see `MagnusFlight.norm_zeta`). -/
noncomputable def vh (t : ℝ) : ℝ := ‖F.zeta 0‖ * Real.exp (-F.κD * F.s t)

/-- `Q = tan γ = v_z / v_h`. -/
noncomputable def Q (t : ℝ) : ℝ := F.vz t / F.vh t

/-- Heading turned, `ψ = κ_L s`. -/
noncomputable def ψ (t : ℝ) : ℝ := F.κL * F.s t

/-- `μ = κ_D / κ_L = C_D / C_L`. -/
noncomputable def μ : ℝ := F.κD / F.κL

/-- `λ = g / (κ_L v_h(0)²) = g L_L / (v₀ cos θ)²`. -/
noncomputable def lam : ℝ := F.g / (F.κL * ‖F.zeta 0‖ ^ 2)

lemma vh_eq_norm (t : ℝ) : F.vh t = ‖F.zeta t‖ := (F.norm_zeta t).symm

lemma vh_pos (h0 : F.zeta 0 ≠ 0) (t : ℝ) : 0 < F.vh t :=
  mul_pos (norm_pos_iff.mpr h0) (Real.exp_pos _)

lemma vh_sq (t : ℝ) : F.vh t ^ 2 = F.vx t ^ 2 + F.vy t ^ 2 := by
  rw [vh_eq_norm, Complex.sq_norm, normSq_apply]
  simp [MagnusFlight.zeta]
  ring

lemma speed_eq_vh (t : ℝ) : F.speed t = √(F.vh t ^ 2 + F.vz t ^ 2) := by
  rw [F.speed_eq, vh_sq]

lemma hasDerivAt_vh (t : ℝ) :
    HasDerivAt F.vh (-F.κD * F.speed t * F.vh t) t := by
  have h := ((((F.hs t).const_mul (-F.κD)).exp).const_mul ‖F.zeta 0‖)
  convert h using 1
  · rfl
  simp only [vh]
  ring

/-- `dQ/dt = -g / v_h`: drag cancels out of the climb-angle equation. -/
theorem hasDerivAt_Q (h0 : F.zeta 0 ≠ 0) (t : ℝ) :
    HasDerivAt F.Q (-F.g / F.vh t) t := by
  have hv := F.vh_pos h0 t
  convert HasDerivAt.div (F.hvz t) (F.hasDerivAt_vh t) hv.ne' using 1
  · rfl
  field_simp
  ring

lemma sqrt_one_add_Q_sq (h0 : F.zeta 0 ≠ 0) (t : ℝ) :
    √(1 + F.Q t ^ 2) = F.speed t / F.vh t := by
  have hv := F.vh_pos h0 t
  rw [speed_eq_vh, Q, show 1 + (F.vz t / F.vh t) ^ 2 = (F.vh t ^ 2 + F.vz t ^ 2) / F.vh t ^ 2 by
    field_simp, Real.sqrt_div' _ (by positivity), Real.sqrt_sq hv.le]

lemma speed_pos (h0 : F.zeta 0 ≠ 0) (t : ℝ) : 0 < F.speed t := by
  rw [speed_eq_vh]
  apply Real.sqrt_pos.mpr
  have := F.vh_pos h0 t
  positivity

/-- Eq. (reduced) of the paper in chain-rule form:
`dQ/dt = (-λ e^{2μψ} / √(1+Q²)) · dψ/dt`, with `dψ/dt = κ_L |v|`. -/
theorem reduced (hL : F.κL ≠ 0) (h0 : F.zeta 0 ≠ 0) (t : ℝ) :
    HasDerivAt F.Q
      (-F.lam * Real.exp (2 * F.μ * F.ψ t) / √(1 + F.Q t ^ 2) * (F.κL * F.speed t)) t := by
  convert F.hasDerivAt_Q h0 t using 1
  have hv := F.vh_pos h0 t
  have hsp := F.speed_pos h0 t
  have hn : 0 < ‖F.zeta 0‖ := norm_pos_iff.mpr h0
  rw [F.sqrt_one_add_Q_sq h0]
  have hE : Real.exp (2 * F.μ * F.ψ t) = Real.exp (F.κD * F.s t) ^ 2 := by
    rw [← Real.exp_nat_mul]; congr 1; simp only [μ, ψ]; field_simp; ring
  have hvh : F.vh t = ‖F.zeta 0‖ / Real.exp (F.κD * F.s t) := by
    rw [vh, neg_mul, Real.exp_neg, div_eq_mul_inv]
  have hEp := Real.exp_pos (F.κD * F.s t)
  rw [hE, hvh, lam]
  field_simp

/-- Rate form of Eq. (closure): `dz/dt = v_z = L_L sin γ · dψ/dt` with `sin γ = v_z/|v|`. -/
theorem height_rate (hL : F.κL ≠ 0) (h0 : F.zeta 0 ≠ 0) (t : ℝ) :
    F.vz t = (1 / F.κL) * (F.vz t / F.speed t) * (F.κL * F.speed t) := by
  have := (F.speed_pos h0 t).ne'
  field_simp

/-- Rate form of Eq. (chord): `d(x+iy)/dt = ζ = L_L cos γ e^{iψ} u₀ · dψ/dt`, where
`cos γ = v_h/|v|` and `u₀ = ζ(0)/|ζ(0)|` is the launch direction. -/
theorem chord_rate (hL : F.κL ≠ 0) (h0 : F.zeta 0 ≠ 0) (t : ℝ) :
    F.zeta t = ((1 / F.κL) * (F.vh t / F.speed t) : ℝ) * exp ((F.ψ t : ℂ) * I) *
      (F.zeta 0 / (‖F.zeta 0‖ : ℂ)) * ((F.κL * F.speed t : ℝ) : ℂ) := by
  have hsp : (F.speed t : ℂ) ≠ 0 := by exact_mod_cast (F.speed_pos h0 t).ne'
  have hn : ((‖F.zeta 0‖ : ℝ) : ℂ) ≠ 0 := by exact_mod_cast (norm_pos_iff.mpr h0).ne'
  have hL' : (F.κL : ℂ) ≠ 0 := by exact_mod_cast hL
  rw [F.zeta_eq_decay_rotation, vh, ψ]
  push_cast
  field_simp

end MagnusFlight3

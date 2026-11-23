/-
Copyright (c) 2026 Pragyaan Gaur. All rights reserved.
Released under the MIT license as described in the file LICENSE.
-/
import Mathlib.Analysis.SpecialFunctions.Arsinh
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

/-!
# The two-ball rendezvous

## i. Overview

Two balls are thrown in opposite directions with opposite vertical spins. They meet
head-on at launch height after each turns by `π` exactly when
`I = ∫₀^π cos γ(ψ) cos ψ dψ = 0` (Eq. (Icond)), where `Q = tan γ` solves the reduced
equation `dQ/dψ = -λ e^{2μψ} / √(1+Q²)` with `cos γ = 1/√(1+Q²)`.

*Without drag* (`μ = 0`) the equation is separable with first integral
`G(Q) = ½ (Q √(1+Q²) + arsinh Q)`, so `G(Q(ψ)) = G(Q(0)) - λψ` (Eq. (sep)). When
`λπ = 2 G(Q(0))`, which is Eq. (lamfree) with `Q(0) = tan θ`, we prove that
`Q(π - ψ) = -Q(ψ)`, that `Q(π) = -Q(0)`, and that `I = 0`, so the balls meet.

*With drag* we prove that the symmetry `Q(π - ψ) = -Q(ψ)`, which is what made `I` vanish,
is impossible for any `μ ≠ 0` (with `λ ≠ 0`). The full statement that `I ≠ 0` for every
`μ > 0` is open: the paper proves it only to lowest order in `μ θ²` and checks it
numerically, and it is not formalised here.

## ii. Key results

- `Rendezvous.G_comp` : the first integral, Eq. (sep).
- `Rendezvous.symm` : `Q(π - ψ) = -Q(ψ)` without drag.
- `Rendezvous.meeting_integral_eq_zero` : `I = 0` without drag.
- `Rendezvous.drag_breaks_symmetry` : with drag, the symmetry forces `μ = 0`.
-/

namespace Rendezvous

open Real

/-- The first integral `G(Q) = ½ (Q √(1+Q²) + arsinh Q)`. -/
noncomputable def G (x : ℝ) : ℝ := (x * √(1 + x ^ 2) + arsinh x) / 2

lemma one_add_sq_pos (x : ℝ) : 0 < 1 + x ^ 2 := by positivity

lemma hasDerivAt_G (x : ℝ) : HasDerivAt G (√(1 + x ^ 2)) x := by
  have hpos := one_add_sq_pos x
  have hq : HasDerivAt (fun x : ℝ => 1 + x ^ 2) (2 * x) x := by
    simpa using ((hasDerivAt_id x).pow 2).const_add 1
  have hs := hq.sqrt hpos.ne'
  have h := (((hasDerivAt_id x).mul hs).add (hasDerivAt_arsinh x)).div_const 2
  convert h using 1
  · funext y; simp [G]
  have hr : 0 < √(1 + x ^ 2) := Real.sqrt_pos.mpr hpos
  have hsq : √(1 + x ^ 2) ^ 2 = 1 + x ^ 2 := Real.sq_sqrt hpos.le
  simp only [id]
  field_simp
  nlinarith [hsq]

lemma G_strictMono : StrictMono G :=
  strictMono_of_deriv_pos fun x => by
    rw [(hasDerivAt_G x).deriv]; exact Real.sqrt_pos.mpr (one_add_sq_pos x)

lemma G_neg (x : ℝ) : G (-x) = -G x := by
  simp only [G, arsinh_neg, neg_sq]; ring

variable {Q : ℝ → ℝ} {lam : ℝ}

/-- Eq. (sep): for the drag-free reduced equation, `G(Q(ψ)) = G(Q(0)) - λψ`. -/
theorem G_comp (hQ : ∀ ψ, HasDerivAt Q (-lam / √(1 + Q ψ ^ 2)) ψ) (ψ : ℝ) :
    G (Q ψ) = G (Q 0) - lam * ψ := by
  set f : ℝ → ℝ := fun ψ => G (Q ψ) + lam * ψ
  have hf : ∀ ψ, HasDerivAt f 0 ψ := by
    intro ψ
    have h := ((hasDerivAt_G (Q ψ)).comp ψ (hQ ψ)).add ((hasDerivAt_id ψ).const_mul lam)
    convert h using 1
    · rfl
    have hr : 0 < √(1 + Q ψ ^ 2) := Real.sqrt_pos.mpr (one_add_sq_pos _)
    have : √(1 + Q ψ ^ 2) * (-lam / √(1 + Q ψ ^ 2)) = -lam := by field_simp
    rw [this]; ring
  have := is_const_of_deriv_eq_zero (fun u => (hf u).differentiableAt)
    (fun u => (hf u).deriv) ψ 0
  simp only [f, mul_zero, add_zero] at this
  linarith

/-- Without drag and with `λπ = 2 G(Q(0))` (Eq. (lamfree)), the climb angle is
antisymmetric about the half turn: `Q(π - ψ) = -Q(ψ)`. -/
theorem symm (hQ : ∀ ψ, HasDerivAt Q (-lam / √(1 + Q ψ ^ 2)) ψ)
    (hlam : lam * π = 2 * G (Q 0)) (ψ : ℝ) : Q (π - ψ) = -Q ψ := by
  apply G_strictMono.injective
  rw [G_neg, G_comp hQ (π - ψ), G_comp hQ ψ]
  linarith

/-- The ball returns to launch height descending at the launch angle: `Q(π) = -Q(0)`. -/
theorem Q_pi (hQ : ∀ ψ, HasDerivAt Q (-lam / √(1 + Q ψ ^ 2)) ψ)
    (hlam : lam * π = 2 * G (Q 0)) : Q π = -Q 0 := by
  simpa using symm hQ hlam 0

/-- Without drag the meeting condition Eq. (Icond) holds:
`∫₀^π cos γ(ψ) cos ψ dψ = 0` with `cos γ = 1/√(1+Q²)`. So the two balls meet. -/
theorem meeting_integral_eq_zero (hQ : ∀ ψ, HasDerivAt Q (-lam / √(1 + Q ψ ^ 2)) ψ)
    (hlam : lam * π = 2 * G (Q 0)) :
    ∫ ψ in (0 : ℝ)..π, (1 / √(1 + Q ψ ^ 2)) * Real.cos ψ = 0 := by
  set f : ℝ → ℝ := fun ψ => (1 / √(1 + Q ψ ^ 2)) * Real.cos ψ
  have hodd : ∀ ψ, f (π - ψ) = -f ψ := by
    intro ψ
    simp only [f, symm hQ hlam ψ, neg_sq, Real.cos_pi_sub]
    ring
  have h := intervalIntegral.integral_comp_sub_left (a := 0) (b := π) f π
  simp only [sub_self, sub_zero, hodd, intervalIntegral.integral_neg] at h
  change ∫ ψ in (0 : ℝ)..π, f ψ = 0
  linarith

/-- With drag the symmetry is lost. If `Q` solves the reduced equation
`dQ/dψ = -λ e^{2μψ}/√(1+Q²)` with `λ ≠ 0` and `Q(π - ψ) = -Q(ψ)` for all `ψ`, then `μ = 0`.
This is the mechanism behind the failure of the rendezvous with drag; it is not by itself
a proof that `I ≠ 0`. -/
theorem drag_breaks_symmetry {μ : ℝ} (hlam0 : lam ≠ 0)
    (hQ : ∀ ψ, HasDerivAt Q (-lam * Real.exp (2 * μ * ψ) / √(1 + Q ψ ^ 2)) ψ)
    (hsym : ∀ ψ, Q (π - ψ) = -Q ψ) : μ = 0 := by
  -- differentiate both sides of the symmetry at ψ = 0
  have hL : HasDerivAt (fun ψ => Q (π - ψ))
      (-lam * Real.exp (2 * μ * π) / √(1 + Q π ^ 2) * (-1)) 0 := by
    have := (hQ (π - 0)).comp 0 ((hasDerivAt_id (0 : ℝ)).const_sub π)
    simpa [Function.comp_def] using this
  have hR : HasDerivAt (fun ψ => -Q ψ) (-(-lam * Real.exp (2 * μ * 0) / √(1 + Q 0 ^ 2))) 0 :=
    (hQ 0).neg
  have heq : (fun ψ => Q (π - ψ)) = fun ψ => -Q ψ := funext hsym
  rw [heq] at hL
  have h := hL.unique hR
  have hQπ : Q π ^ 2 = Q 0 ^ 2 := by
    have := hsym 0; simp at this; rw [this, neg_sq]
  rw [hQπ] at h
  have hr : 0 < √(1 + Q 0 ^ 2) := Real.sqrt_pos.mpr (one_add_sq_pos _)
  simp only [mul_zero, Real.exp_zero] at h
  field_simp at h
  have he := h
  rw [Real.exp_eq_one_iff] at he
  have hπ := Real.pi_pos
  nlinarith [mul_pos (by norm_num : (0:ℝ) < 2) hπ]

end Rendezvous

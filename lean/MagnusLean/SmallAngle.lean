/-
Copyright (c) 2026 Pragyaan Gaur. All rights reserved.
Released under the MIT license as described in the file LICENSE.
-/
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Real.Pi.Bounds

/-!
# The rendezvous fails for every drag in the small-angle limit

## i. Overview

For a shallow throw, `Q = tan γ` stays small, so `√(1+Q²) ≈ 1`, `sin γ ≈ Q` and
`cos γ ≈ 1 - Q²/2`. The reduced equation `dQ/dψ = -λ e^{2μψ}/√(1+Q²)` becomes
`dQ/dψ = -λ e^{2μψ}`, whose solution is `Q(ψ) = A + B e^{kψ}` with `k = 2μ` and
`B = -λ/(2μ)`. The closure condition `∫₀^π sin γ = 0` becomes `∫₀^π Q = 0`, and the
meeting integral becomes `I = ∫₀^π (1 - Q²/2) cos ψ dψ = -½ ∫₀^π Q² cos ψ dψ`.

We prove that, for every `k > 0` and `B ≠ 0`, closure forces `∫₀^π Q² cos ψ dψ < 0`,
so `I > 0` and the balls miss. This holds for **all** drag-to-lift ratios `μ > 0` in the
small-angle model, not only to lowest order in `μ`. The paper's coefficient
`8/3 - 24/π²` is the `μ → 0` slope of this quantity.

The proof reduces to `h(a) > 0` for `a = kπ > 0`, where
`h(a) = a(π² + a²)(e^{2a} + 1) - (π² + 4a²)(e^{2a} - 1)`.
We have `h(0) = h'(0) = h''(0) = 0`, `h'''(0) = 4π² - 36 > 0`, and
`h''''(a) = e^{2a}(16a³ + 32a² + (16π² - 112)a + 16π² - 144) > 0`. Both signs need
only `π² > 9`. Four applications of the mean value theorem then give `h > 0`.

This is a statement about the small-angle model, not the full nonlinear equation. The
full statement for all `μ` and `θ` remains open.

## ii. Key results

- `SmallAngle.h_pos` : `h(a) > 0` for `a > 0`.
- `SmallAngle.integral_Q_sq_cos_neg` : closure implies `∫₀^π Q² cos ψ dψ < 0`.
- `SmallAngle.meeting_integral_pos` : closure implies `I > 0`, so no rendezvous.
-/

namespace SmallAngle

open Real intervalIntegral

lemma pi_sq_gt_nine : 9 < π ^ 2 := by nlinarith [pi_gt_three]

/-- If `f(0) ≥ 0` and `f' > 0` on `(0, ∞)`, then `f > 0` on `(0, ∞)`. -/
lemma pos_of_deriv_pos {f f' : ℝ → ℝ} (hf : ∀ x, HasDerivAt f (f' x) x) (h0 : 0 ≤ f 0)
    (hd : ∀ x, 0 < x → 0 < f' x) {x : ℝ} (hx : 0 < x) : 0 < f x := by
  obtain ⟨c, hc, hc'⟩ := exists_hasDerivAt_eq_slope f f' hx
    (fun y _ => (hf y).continuousAt.continuousWithinAt) (fun y _ => hf y)
  have := hd c hc.1
  rw [hc', sub_zero, lt_div_iff₀ hx, zero_mul] at this
  linarith

/-- A generic derivative: `d/da (P(a) e^{2a} + R(a)) = (P' + 2P) e^{2a} + R'`. -/
lemma hasDerivAt_pe {P R : ℝ → ℝ} {p' r' a : ℝ} (hP : HasDerivAt P p' a)
    (hR : HasDerivAt R r' a) :
    HasDerivAt (fun a => P a * exp (2 * a) + R a)
      ((p' + 2 * P a) * exp (2 * a) + r') a := by
  have he : HasDerivAt (fun a => exp (2 * a)) (exp (2 * a) * 2) a := by
    simpa using ((hasDerivAt_id a).const_mul 2).exp
  convert (hP.mul he).add hR using 1
  ring

noncomputable def h0 (a : ℝ) : ℝ :=
  (a ^ 3 - 4 * a ^ 2 + π ^ 2 * a - π ^ 2) * exp (2 * a) + (a ^ 3 + 4 * a ^ 2 + π ^ 2 * a + π ^ 2)
noncomputable def h1 (a : ℝ) : ℝ :=
  (2 * a ^ 3 - 5 * a ^ 2 + (2 * π ^ 2 - 8) * a - π ^ 2) * exp (2 * a) +
    (3 * a ^ 2 + 8 * a + π ^ 2)
noncomputable def h2 (a : ℝ) : ℝ :=
  (4 * a ^ 3 - 4 * a ^ 2 + (4 * π ^ 2 - 26) * a - 8) * exp (2 * a) + (6 * a + 8)
noncomputable def h3 (a : ℝ) : ℝ :=
  (8 * a ^ 3 + 4 * a ^ 2 + (8 * π ^ 2 - 60) * a + 4 * π ^ 2 - 42) * exp (2 * a) + 6
noncomputable def h4 (a : ℝ) : ℝ :=
  (16 * a ^ 3 + 32 * a ^ 2 + (16 * π ^ 2 - 112) * a + 16 * π ^ 2 - 144) * exp (2 * a) + 0

/-- Polynomial derivatives, packaged for `hasDerivAt_pe`. -/
lemma hasDerivAt_cubic (c3 c2 c1 c0 a : ℝ) :
    HasDerivAt (fun a : ℝ => c3 * a ^ 3 + c2 * a ^ 2 + c1 * a + c0)
      (3 * c3 * a ^ 2 + 2 * c2 * a + c1) a := by
  have := ((((hasDerivAt_pow 3 a).const_mul c3).add ((hasDerivAt_pow 2 a).const_mul c2)).add
    ((hasDerivAt_id a).const_mul c1)).add_const c0
  convert this using 1
  simp; ring

lemma d0 (a : ℝ) : HasDerivAt h0 (h1 a) a := by
  have := hasDerivAt_pe (a := a) (hasDerivAt_cubic 1 (-4) (π ^ 2) (-π ^ 2) a)
    (hasDerivAt_cubic 1 4 (π ^ 2) (π ^ 2) a)
  convert this using 1
  · funext x; simp only [h0]; ring
  · simp only [h1]; ring

lemma d1 (a : ℝ) : HasDerivAt h1 (h2 a) a := by
  have := hasDerivAt_pe (a := a) (hasDerivAt_cubic 2 (-5) (2 * π ^ 2 - 8) (-π ^ 2) a)
    (hasDerivAt_cubic 0 3 8 (π ^ 2) a)
  convert this using 1
  · funext x; simp only [h1]; ring
  · simp only [h2]; ring

lemma d2 (a : ℝ) : HasDerivAt h2 (h3 a) a := by
  have := hasDerivAt_pe (a := a) (hasDerivAt_cubic 4 (-4) (4 * π ^ 2 - 26) (-8) a)
    (hasDerivAt_cubic 0 0 6 8 a)
  convert this using 1
  · funext x; simp only [h2]; ring
  · simp only [h3]; ring

lemma d3 (a : ℝ) : HasDerivAt h3 (h4 a) a := by
  have := hasDerivAt_pe (a := a) (hasDerivAt_cubic 8 4 (8 * π ^ 2 - 60) (4 * π ^ 2 - 42) a)
    (hasDerivAt_cubic 0 0 0 6 a)
  convert this using 1
  · funext x; simp only [h3]; ring
  · simp only [h4]; ring

lemma h4_pos {a : ℝ} (ha : 0 < a) : 0 < h4 a := by
  have hp := pi_sq_gt_nine
  have : 0 < 16 * a ^ 3 + 32 * a ^ 2 + (16 * π ^ 2 - 112) * a + 16 * π ^ 2 - 144 := by
    nlinarith [pow_pos ha 3, pow_pos ha 2]
  simp only [h4, add_zero]
  exact mul_pos this (exp_pos _)

/-- `h(a) > 0` for every `a > 0`. -/
theorem h_pos {a : ℝ} (ha : 0 < a) : 0 < h0 a := by
  have hp := pi_sq_gt_nine
  have p3 : ∀ x, 0 < x → 0 < h3 x := fun x hx =>
    pos_of_deriv_pos d3 (by simp [h3]; nlinarith) (fun y hy => h4_pos hy) hx
  have p2 : ∀ x, 0 < x → 0 < h2 x := fun x hx =>
    pos_of_deriv_pos d2 (by simp [h2]) p3 hx
  have p1 : ∀ x, 0 < x → 0 < h1 x := fun x hx =>
    pos_of_deriv_pos d1 (by simp [h1]) p2 hx
  exact pos_of_deriv_pos d0 (by simp [h0]) p1 ha

/-- `h0` is the function `h(a) = a(π² + a²)(e^{2a} + 1) - (π² + 4a²)(e^{2a} - 1)`. -/
lemma h0_eq (a : ℝ) :
    h0 a = a * (π ^ 2 + a ^ 2) * (exp (2 * a) + 1) - (π ^ 2 + 4 * a ^ 2) * (exp (2 * a) - 1) := by
  simp only [h0]; ring

/-! ### The integrals -/

/-- Antiderivative of `e^{cψ} cos ψ`. -/
lemma hasDerivAt_exp_cos (c ψ : ℝ) :
    HasDerivAt (fun ψ => exp (c * ψ) * (c * cos ψ + sin ψ) / (c ^ 2 + 1))
      (exp (c * ψ) * cos ψ) ψ := by
  have he : HasDerivAt (fun ψ => exp (c * ψ)) (exp (c * ψ) * c) ψ := by
    simpa using ((hasDerivAt_id ψ).const_mul c).exp
  have hcs : HasDerivAt (fun ψ => c * cos ψ + sin ψ) (c * -sin ψ + cos ψ) ψ :=
    ((hasDerivAt_cos ψ).const_mul c).add (hasDerivAt_sin ψ)
  have h := (he.mul hcs).div_const (c ^ 2 + 1)
  convert h using 1
  have : 0 < c ^ 2 + 1 := by positivity
  field_simp
  ring

/-- Antiderivative of `Q² cos ψ` with `Q = A + B e^{kψ}`. -/
lemma hasDerivAt_F (A B k ψ : ℝ) :
    HasDerivAt (fun ψ => A ^ 2 * sin ψ
        + 2 * A * B * (exp (k * ψ) * (k * cos ψ + sin ψ) / (k ^ 2 + 1))
        + B ^ 2 * (exp ((2 * k) * ψ) * ((2 * k) * cos ψ + sin ψ) / ((2 * k) ^ 2 + 1)))
      ((A + B * exp (k * ψ)) ^ 2 * cos ψ) ψ := by
  have h := (((hasDerivAt_sin ψ).const_mul (A ^ 2)).add
    ((hasDerivAt_exp_cos k ψ).const_mul (2 * A * B))).add
    ((hasDerivAt_exp_cos (2 * k) ψ).const_mul (B ^ 2))
  convert h using 1
  rw [show (2 * k) * ψ = k * ψ + k * ψ by ring, exp_add]
  ring

/-- Antiderivative of `Q = A + B e^{kψ}`. -/
lemma hasDerivAt_G {k : ℝ} (hk : k ≠ 0) (A B ψ : ℝ) :
    HasDerivAt (fun ψ => A * ψ + B * exp (k * ψ) / k) (A + B * exp (k * ψ)) ψ := by
  have he : HasDerivAt (fun ψ => exp (k * ψ)) (exp (k * ψ) * k) ψ := by
    simpa using ((hasDerivAt_id ψ).const_mul k).exp
  convert ((hasDerivAt_id ψ).const_mul A).add ((he.const_mul B).div_const k) using 1
  · rfl
  · field_simp

lemma integral_Q {k : ℝ} (hk : k ≠ 0) (A B : ℝ) :
    ∫ ψ in (0 : ℝ)..π, (A + B * exp (k * ψ)) = A * π + B * (exp (k * π) - 1) / k := by
  rw [integral_eq_sub_of_hasDerivAt (fun ψ _ => hasDerivAt_G hk A B ψ)
    (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [mul_zero, exp_zero]
  ring

lemma integral_Q_sq_cos (A B k : ℝ) :
    ∫ ψ in (0 : ℝ)..π, (A + B * exp (k * ψ)) ^ 2 * cos ψ =
      2 * A * B * (-k * (exp (k * π) + 1) / (k ^ 2 + 1)) +
        B ^ 2 * (-(2 * k) * (exp (k * π) ^ 2 + 1) / ((2 * k) ^ 2 + 1)) := by
  rw [integral_eq_sub_of_hasDerivAt (fun ψ _ => hasDerivAt_F A B k ψ)
    (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [sin_pi, cos_pi, sin_zero, cos_zero, mul_zero, exp_zero]
  rw [show 2 * k * π = k * π + k * π by ring, exp_add]
  ring

/-- **Main theorem.** In the small-angle model `Q(ψ) = A + B e^{kψ}` with `k = 2μ > 0` and
`B ≠ 0`, the closure condition `∫₀^π Q = 0` forces `∫₀^π Q² cos ψ dψ < 0`. -/
theorem integral_Q_sq_cos_neg {k A B : ℝ} (hk : 0 < k) (hB : B ≠ 0)
    (hclos : ∫ ψ in (0 : ℝ)..π, (A + B * exp (k * ψ)) = 0) :
    ∫ ψ in (0 : ℝ)..π, (A + B * exp (k * ψ)) ^ 2 * cos ψ < 0 := by
  rw [integral_Q hk.ne'] at hclos
  rw [integral_Q_sq_cos]
  set E := exp (k * π) with hE
  have hπ := pi_pos
  have h3 : A * π * k + B * (E - 1) = 0 := by
    calc A * π * k + B * (E - 1) = (A * π + B * (E - 1) / k) * k := by
          field_simp
      _ = 0 := by rw [hclos, zero_mul]
  have hA : A = -B * (E - 1) / (k * π) := by
    field_simp; linear_combination h3
  have hh := h_pos (mul_pos hk hπ)
  rw [h0_eq, show 2 * (k * π) = k * π + k * π by ring, exp_add, ← hE] at hh
  have hB2 : 0 < B ^ 2 := by positivity
  -- the integral equals `-2 B² h(kπ) / (π³ (k²+1)(4k²+1))`
  have key : 2 * A * B * (-k * (E + 1) / (k ^ 2 + 1)) +
      B ^ 2 * (-(2 * k) * (E ^ 2 + 1) / ((2 * k) ^ 2 + 1)) =
      -2 * B ^ 2 * (k * π * (π ^ 2 + (k * π) ^ 2) * (E * E + 1) -
        (π ^ 2 + 4 * (k * π) ^ 2) * (E * E - 1)) /
        (π ^ 3 * (k ^ 2 + 1) * (4 * k ^ 2 + 1)) := by
    rw [hA]; field_simp; ring
  rw [key]
  apply div_neg_of_neg_of_pos
  · nlinarith
  · positivity

/-- In the small-angle model the meeting integral
`I = ∫₀^π (1 - Q²/2) cos ψ dψ` is strictly positive, so the balls do not meet, for every
drag `k = 2μ > 0` and `B = -λ/(2μ) ≠ 0`. -/
theorem meeting_integral_pos {k A B : ℝ} (hk : 0 < k) (hB : B ≠ 0)
    (hclos : ∫ ψ in (0 : ℝ)..π, (A + B * exp (k * ψ)) = 0) :
    0 < ∫ ψ in (0 : ℝ)..π, (1 - (A + B * exp (k * ψ)) ^ 2 / 2) * cos ψ := by
  have hneg := integral_Q_sq_cos_neg hk hB hclos
  have hsplit : ∫ ψ in (0 : ℝ)..π, (1 - (A + B * exp (k * ψ)) ^ 2 / 2) * cos ψ =
      (∫ ψ in (0 : ℝ)..π, cos ψ) -
        (1 / 2) * ∫ ψ in (0 : ℝ)..π, (A + B * exp (k * ψ)) ^ 2 * cos ψ := by
    rw [← intervalIntegral.integral_const_mul, ← intervalIntegral.integral_sub
      (by apply Continuous.intervalIntegrable; fun_prop)
      (by apply Continuous.intervalIntegrable; fun_prop)]
    congr 1; funext ψ; ring
  rw [hsplit, integral_cos, sin_pi, sin_zero]
  linarith

end SmallAngle

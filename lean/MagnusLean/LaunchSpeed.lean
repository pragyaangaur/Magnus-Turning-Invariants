/-
Copyright (c) 2026 Pragyaan Gaur. All rights reserved.
Released under the MIT license as described in the file LICENSE.
-/
import MagnusLean.Rendezvous

/-!
# The drag-free launch speed

With `Q(0) = tan θ`, the drag-free meeting condition `λπ = 2 G(tan θ)` (Eq. (lamfree))
together with `λ = g L_L / (v₀ cos θ)²` (Eq. (lambda)) is equivalent to Eq. (v0):
`v₀² = π g L_L / f(θ)`, `f(θ) = sin θ + cos² θ · ln(tan θ + sec θ)`.
We also prove `f(θ) > 0` for `0 < θ < π/2`, so the required speed exists and is unique.
-/

namespace Rendezvous

open Real

/-- `f(θ) = sin θ + cos² θ · ln(tan θ + sec θ)`. -/
noncomputable def f (θ : ℝ) : ℝ := sin θ + cos θ ^ 2 * log (tan θ + 1 / cos θ)

lemma sqrt_one_add_tan_sq {θ : ℝ} (hc : 0 < cos θ) : √(1 + tan θ ^ 2) = 1 / cos θ := by
  rw [show 1 + tan θ ^ 2 = (1 / cos θ) ^ 2 by
    rw [tan_eq_sin_div_cos]; field_simp; nlinarith [sin_sq_add_cos_sq θ]]
  exact Real.sqrt_sq (by positivity)

/-- `2 G(tan θ) = tan θ sec θ + arsinh(tan θ) = f(θ) / cos² θ`. -/
lemma two_G_tan {θ : ℝ} (hc : 0 < cos θ) : 2 * G (tan θ) = f θ / cos θ ^ 2 := by
  simp only [G, f, arsinh]
  rw [sqrt_one_add_tan_sq hc, tan_eq_sin_div_cos]
  field_simp

lemma f_pos {θ : ℝ} (h1 : 0 < θ) (h2 : θ < π / 2) : 0 < f θ := by
  have hc : 0 < cos θ := cos_pos_of_mem_Ioo ⟨by linarith, h2⟩
  have hs : 0 < sin θ := sin_pos_of_pos_of_lt_pi h1 (by linarith)
  have ht : 0 < tan θ := by rw [tan_eq_sin_div_cos]; positivity
  have hsec : 1 ≤ 1 / cos θ := by rw [le_div_iff₀ hc]; linarith [cos_le_one θ]
  have hlog : 0 < log (tan θ + 1 / cos θ) := log_pos (by linarith)
  unfold f; positivity

/-- Eq. (v0): with `λ = g L / (v₀ cos θ)²`, the drag-free meeting condition
`λπ = 2 G(tan θ)` holds exactly when `v₀² = π g L / f(θ)`. -/
theorem launch_speed {θ g L v₀ : ℝ} (h1 : 0 < θ) (h2 : θ < π / 2) (hv : 0 < v₀) :
    g * L / (v₀ * cos θ) ^ 2 * π = 2 * G (tan θ) ↔ v₀ ^ 2 = π * g * L / f θ := by
  have hc : 0 < cos θ := cos_pos_of_mem_Ioo ⟨by linarith, h2⟩
  have hf := f_pos h1 h2
  rw [two_G_tan hc]
  constructor
  · intro h; field_simp at h ⊢; linarith
  · intro h
    have hgL : g * L ≠ 0 := by
      intro h0
      have h00 : v₀ ^ 2 = 0 := by rw [h, mul_assoc, h0]; simp
      have := pow_pos hv 2; linarith
    rw [mul_pow, h]; field_simp; exact div_self hgL

end Rendezvous

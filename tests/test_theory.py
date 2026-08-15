"""Regression tests for the exact analytic structure found in src/theory.py."""
import numpy as np
import pytest

from src.model import make_ball, Params, initial_state
from src.integrate import fly
from src.solve import solve_v0
from src.sweep import ground_track_curvature
from src.theory import horizontal_speed_law, chord_bearing_deg

SPIN = 200.0
BALLS = ["baseball", "frisbee", "soccer", "pingpong"]


@pytest.mark.parametrize("nm", BALLS)
@pytest.mark.parametrize("td", [25.0, 50.0, 75.0])
def test_horizontal_speed_is_exact_log_spiral(nm, td):
    """|w(s)| = |w_0| exp(-(C_D/C_L) * dpsi), EXACTLY, with gravity and drag both on.

    This is the user's exp(-pi C_D/C_L) formula, correctly attached to the HORIZONTAL
    speed rather than the total speed.
    """
    p = make_ball(nm)
    r = fly(initial_state(60.0, np.radians(td), SPIN), p, t_max_s=400.0, stop_on="height")
    Y = r["sol"].sol(np.linspace(0.0, r["t_height_s"], 2000))
    w = np.hypot(Y[3], Y[4])
    pred = w[0] * horizontal_speed_law(p, Y[9] - Y[9][0])
    assert np.abs(w / pred - 1.0).max() < 1e-9


def test_horizontal_speed_law_is_independent_of_gravity():
    """The law comes from dw/ds = (i k_L - k_D) w, which has no g in it."""
    for g in (9.80665, 1.62, 24.8):
        p = make_ball("frisbee", g_m_per_s2=g)
        r = fly(initial_state(40.0, np.radians(40.0), SPIN), p, t_max_s=900.0, stop_on="height")
        Y = r["sol"].sol(np.linspace(0.0, r["t_height_s"], 1000))
        w = np.hypot(Y[3], Y[4])
        pred = w[0] * horizontal_speed_law(p, Y[9] - Y[9][0])
        assert np.abs(w / pred - 1.0).max() < 1e-9


@pytest.mark.parametrize("nm,drag", [("baseball", False), ("frisbee", True), ("soccer", True)])
@pytest.mark.parametrize("td", [25.0, 55.0])
def test_curvature_equals_LL_cos_gamma(nm, drag, td):
    """R_h = L_L cos(gamma) exactly, so max R_h = L_L at the apex -- drag or no drag."""
    p = make_ball(nm, drag_on=drag)
    th = np.radians(td)
    v0 = solve_v0(th, p, v0_cap_m_per_s=1000.0)
    r = fly(initial_state(v0, th, SPIN), p, t_max_s=400.0, stop_on="height")
    _, Y, Rh = ground_track_curvature(r["sol"], r["t_height_s"], p)
    cos_gamma = np.hypot(Y[3], Y[4]) / np.linalg.norm(Y[3:6], axis=0)
    assert np.abs(Rh / (p.lift_length_m * cos_gamma) - 1.0).max() < 1e-12
    assert abs(Rh.max() / p.lift_length_m - 1.0) < 1e-6


@pytest.mark.parametrize("td", [20.0, 45.0, 70.0])
def test_chord_is_perpendicular_without_drag(td):
    """Drag-free: cos(gamma) is symmetric about the apex, which sits at psi = pi/2,
    so the real part of the chord integral vanishes and the bearing is exactly 90 deg."""
    p = make_ball("baseball", drag_on=False)
    th = np.radians(td)
    v0 = solve_v0(th, p, v0_cap_m_per_s=1000.0)
    assert abs(chord_bearing_deg(v0, th, p) - 90.0) < 1e-7


@pytest.mark.parametrize("nm", ["frisbee", "soccer", "pingpong"])
@pytest.mark.parametrize("td", [10.0, 30.0, 60.0])
def test_drag_always_tips_the_chord_below_perpendicular(nm, td):
    """With drag the descent is steeper than the ascent at matched turn angle, so the
    chord bearing is strictly < 90 deg.  Hence a mirror pair thrown in exactly opposite
    directions can never meet -- the symmetric problem is unsolvable with any drag."""
    p = make_ball(nm)
    th = np.radians(td)
    v0 = solve_v0(th, p, v0_cap_m_per_s=3000.0)
    b = chord_bearing_deg(v0, th, p)
    assert b < 90.0 - 1e-6, f"{nm} at {td} deg gave bearing {b}"


def test_bearing_deficit_depends_only_on_CD_over_CL_and_theta():
    """At closure the deficit (90 - bearing) is a function of mu = C_D/C_L and theta only,
    not of the individual coefficients or the ball's size and mass."""
    th = np.radians(45.0)
    vals = []
    for cl, cd in [(0.6, 0.02), (1.5, 0.05), (3.0, 0.10)]:      # all mu = 1/30
        p = Params(mass_kg=0.175, radius_m=0.137, cl_const=cl, cd_const=cd)
        vals.append(chord_bearing_deg(solve_v0(th, p, v0_cap_m_per_s=3000.0), th, p))
    assert max(vals) - min(vals) < 1e-6


def test_bearing_deficit_vanishes_quadratically_as_theta_goes_to_zero():
    """(90 - bearing) ~ theta^2, so the symmetric solution is recovered only in the
    degenerate flat-throw limit theta -> 0 (where v0 -> infinity)."""
    p = make_ball("frisbee")
    d = {}
    for td in (1.0, 2.0, 4.0):
        th = np.radians(td)
        d[td] = 90.0 - chord_bearing_deg(solve_v0(th, p, v0_cap_m_per_s=3000.0), th, p)
    assert 3.5 < d[2.0] / d[1.0] < 4.5
    assert 3.5 < d[4.0] / d[2.0] < 4.5


@pytest.mark.parametrize("nm", ["baseball", "frisbee"])
@pytest.mark.parametrize("drag", [False, True])
def test_zero_gravity_track_is_a_circle_of_radius_LL(nm, drag):
    """Drag is antiparallel to v, so it cannot bend the path -- only slow travel along it.
    With g=0 the ground track is therefore an exact circle of radius L_L either way."""
    from src.sweep import fit_circle
    p = make_ball(nm, gravity_on=False, drag_on=drag)
    r = fly(initial_state(50.0, 0.0, SPIN), p, t_max_s=5000.0,
            stop_on="psi", psi_target_rad=0.5 * np.pi)
    Y = r["sol"].sol(np.linspace(0.0, r["t_psi_s"], 3000))
    *_, R, rms = fit_circle(Y[0], Y[1])
    assert abs(R / p.lift_length_m - 1.0) < 1e-9
    assert rms / R < 1e-9

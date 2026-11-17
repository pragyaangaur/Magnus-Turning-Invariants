"""The horizontal solution with lift and drag coefficients that change along the path.

With a vertical spin axis, d(zeta)/ds = (i k_L(s) - k_D(s)) zeta holds at every point,
so the heading is the integral of k_L over path length and the horizontal speed decays
as exp(-integral of k_D). Drag never enters the heading. These tests check that with a
spin-dependent C_L and with the drag crisis switched on.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

from src.model import make_ball, initial_state, rhs, c_lift, c_drag
from src.integrate import fly

SPIN = 200.0


def _path_integrals(p, v0, theta_deg, n=40001):
    r = fly(initial_state(v0, np.radians(theta_deg), SPIN), p, t_max_s=30.0,
            stop_on="height", rtol=1e-11, atol=1e-13)
    ts = np.linspace(0.0, r["t_height_s"], n)
    Y = r["sol"].sol(ts)
    speed = np.linalg.norm(Y[3:6], axis=0)
    spin = np.linalg.norm(Y[6:9], axis=0)
    k = p.rho_air_kg_per_m3 * p.area_m2 / (2.0 * p.mass_kg)
    kL = k * c_lift(p, p.radius_m * spin / speed)
    kD = k * c_drag(p, speed)
    s = Y[10]
    return Y, cumulative_trapezoid(kL, s, initial=0.0), cumulative_trapezoid(kD, s, initial=0.0)


def test_heading_is_integral_of_kL_with_spin_dependent_lift():
    p = make_ball("soccer", cl_model="saturating", spin_decay_on=True)
    Y, int_kL, _ = _path_integrals(p, 28.0, 25.0)
    assert np.abs(Y[9] - int_kL).max() < 1e-7


def test_horizontal_speed_follows_integral_of_kD_with_drag_crisis():
    p = make_ball("soccer", cd_model="crisis")
    Y, _, int_kD = _path_integrals(p, 30.0, 30.0)
    vh = np.hypot(Y[3], Y[4])
    assert np.abs(vh / vh[0] - np.exp(-int_kD)).max() < 1e-7


def test_drag_does_not_enter_the_heading():
    """Two balls that differ only in C_D have the same heading at the same path length."""
    a = make_ball("soccer", cd_const=0.10)
    b = make_ball("soccer", cd_const=0.50)
    s_grid = np.linspace(0.0, 15.0, 200)
    psis = []
    for p in (a, b):
        r = fly(initial_state(25.0, np.radians(20.0), SPIN), p, t_max_s=30.0, stop_on="height")
        ts = np.linspace(0.0, r["t_height_s"], 20001)
        Y = r["sol"].sol(ts)
        psis.append(np.interp(s_grid, Y[10], Y[9]))
    assert np.abs(psis[0] - psis[1]).max() < 1e-8

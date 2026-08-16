"""Tests for the exact reduction to a single scalar ODE in the turn angle."""
import numpy as np
import pytest

from src.model import make_ball, v0_closed_form
from src.solve import solve_v0
from src.theory import chord_bearing_deg
from src.reduced import (solve_Q, solve_lambda, lambda_dragfree, bearing_deg,
                         deficit_deg, deficit_grid_deg)


@pytest.mark.parametrize("nm", ["frisbee", "soccer", "pingpong", "baseball"])
@pytest.mark.parametrize("td", [20.0, 45.0, 70.0])
def test_reduced_ode_reproduces_full_simulation(nm, td):
    """gamma(psi) from the scalar ODE matches the full 9-D integration."""
    from src.integrate import fly
    from src.model import initial_state
    p = make_ball(nm)
    th = np.radians(td)
    v0 = solve_v0(th, p, v0_cap_m_per_s=5000.0)
    r = fly(initial_state(v0, th, 200.0), p,
            t_max_s=1.5 * 2 * v0 * np.sin(th) / p.g_m_per_s2 + 5, stop_on="height")
    Y = r["sol"].sol(np.linspace(0.0, r["t_height_s"], 4000))
    psi = Y[9]
    gam_full = np.arctan2(Y[5], np.hypot(Y[3], Y[4]))

    mu = p.cd_const / p.cl_const
    lam = p.g_m_per_s2 / (p.k_lift_per_m * (v0 * np.cos(th)) ** 2)
    u, _, _ = solve_Q(th, lam, mu, u_end=psi[-1], n=2)
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda uu, Q: -lam * np.exp(2 * mu * uu) / np.sqrt(1 + Q[0] ** 2),
                    (0.0, psi[-1]), [np.tan(th)], rtol=1e-12, atol=1e-14, dense_output=True)
    gam_red = np.arctan(sol.sol(psi)[0])
    assert np.abs(gam_full - gam_red).max() < 1e-8


@pytest.mark.parametrize("td", [10.0, 30.0, 45.0, 60.0, 80.0])
def test_dragfree_lambda_closed_form(td):
    """mu=0 is separable: lambda = [tan(th)sec(th) + asinh(tan th)]/pi."""
    th = np.radians(td)
    assert abs(lambda_dragfree(th) / solve_lambda(th, 0.0) - 1.0) < 1e-10


@pytest.mark.parametrize("td", [15.0, 45.0, 75.0])
def test_reduced_closure_reproduces_v0_formula(td):
    """lambda = g/(k_L v0^2 cos^2 th) with the drag-free lambda gives back v0(theta)."""
    p = make_ball("baseball", drag_on=False)
    th = np.radians(td)
    v0 = np.sqrt(p.g_m_per_s2 / (p.k_lift_per_m * lambda_dragfree(th) * np.cos(th) ** 2))
    assert abs(v0 / float(v0_closed_form(th, p)) - 1.0) < 1e-12


@pytest.mark.parametrize("td", [10.0, 45.0, 80.0])
def test_dragfree_bearing_is_exactly_90(td):
    """Antisymmetry Q(u) = -Q(pi-u) makes the real part of the chord integral vanish."""
    assert abs(bearing_deg(np.radians(td), 0.0)[0] - 90.0) < 1e-9


@pytest.mark.parametrize("nm", ["frisbee", "soccer", "pingpong", "baseball"])
@pytest.mark.parametrize("td", [20.0, 45.0, 70.0])
def test_reduced_bearing_matches_full_simulation(nm, td):
    p = make_ball(nm)
    th = np.radians(td)
    v0 = solve_v0(th, p, v0_cap_m_per_s=5000.0)
    b_full = chord_bearing_deg(v0, th, p)
    b_red, _ = bearing_deg(th, p.cd_const / p.cl_const)
    assert abs(b_red - b_full) < 1e-4


def test_deficit_is_strictly_positive_over_the_domain():
    """The rendezvous fails everywhere: no (mu, theta) makes the chord perpendicular."""
    D, _ = deficit_grid_deg(np.radians(np.linspace(2.0, 88.0, 40)),
                            np.logspace(-2, np.log10(3.0), 40), n_bisect=70)
    assert D.min() > 0.0


def test_leading_asymptotics_constant():
    """(90deg - beta) -> (8/3 - 24/pi^2) * mu * theta^2, positive because pi^2 > 9."""
    C = 8.0 / 3.0 - 24.0 / np.pi ** 2
    assert C > 0.0
    th = np.radians(0.5)
    mus = np.array([0.005, 0.0025, 0.00125])
    vals = np.array([np.radians(deficit_deg(th, m)) / (m * th ** 2) for m in mus])
    # Richardson extrapolation in mu, assuming C(mu) = C0 + c1*mu
    C0 = vals[-1] + (vals[-1] - vals[-2]) * mus[-1] / (mus[-2] - mus[-1])
    assert abs(C0 / C - 1.0) < 1e-4


def test_pointwise_inequality_actually_fails():
    """Guards the corrected claim: cos(gamma(u)) - cos(gamma(pi-u)) changes sign.

    The impossibility result is an integral statement, NOT a pointwise one; the apex
    sits at turn angle > pi/2, so near the midpoint the ordering reverses.
    """
    th, mu = np.radians(45.0), 1.0
    u, _, gam = solve_Q(th, solve_lambda(th, mu), mu, n=4001)
    c = np.cos(gam)
    half = (len(u) - 1) // 2 + 1
    d = c[:half] - c[::-1][:half]
    assert d[1] > 0.0 and d.min() < 0.0

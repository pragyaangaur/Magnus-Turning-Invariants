"""Task 1 regression tests: the drag-free analytic claims, at tight tolerances."""
import numpy as np
import pytest

from src.model import (make_ball, initial_state, rhs, ballistic_arc_length_m,
                       shape_factor, v0_closed_form)
from src.integrate import fly
from src.solve import solve_v0

SPIN = 200.0
V0S = [10.0, 35.0, 70.0, 120.0]
THETAS_DEG = [5.0, 20.0, 45.0, 70.0, 85.0]


@pytest.fixture(scope="module")
def ideal():
    return make_ball("baseball", drag_on=False, cl_model="const", spin_decay_on=False)


def test_a_turning_rate_constant(ideal):
    """dpsi/ds == k_L to machine precision along the whole path."""
    kL = ideal.k_lift_per_m
    worst = 0.0
    for v0 in (20.0, 60.0, 110.0):
        for td in (10.0, 45.0, 80.0):
            r = fly(initial_state(v0, np.radians(td), SPIN), ideal,
                    t_max_s=8000.0, stop_on="psi")
            ts = np.linspace(0.0, r["t_psi_s"], 1500)
            d = rhs(0.0, r["sol"].sol(ts), ideal)
            worst = max(worst, np.abs(d[9] / d[10] / kL - 1.0).max())
    assert worst < 1e-13, f"max relative deviation {worst:.3e}"


def test_a_holds_with_drag_on():
    """Drag is antiparallel to v, so it cannot torque the azimuth: dpsi/ds = k_L still."""
    p = make_ball("baseball", drag_on=True)
    r = fly(initial_state(60.0, np.radians(40.0), SPIN), p, t_max_s=8000.0, stop_on="psi")
    ts = np.linspace(0.0, r["t_psi_s"], 1500)
    d = rhs(0.0, r["sol"].sol(ts), p)
    assert np.abs(d[9] / d[10] / p.k_lift_per_m - 1.0).max() < 1e-13


@pytest.mark.parametrize("v0", V0S)
@pytest.mark.parametrize("td", THETAS_DEG)
def test_b_total_path_is_pi_over_kL(ideal, v0, td):
    r = fly(initial_state(v0, np.radians(td), SPIN), ideal, t_max_s=20000.0, stop_on="psi")
    assert abs(r["y_psi"][10] / (np.pi * ideal.lift_length_m) - 1.0) < 1e-12


@pytest.mark.parametrize("v0", V0S)
@pytest.mark.parametrize("td", THETAS_DEG)
def test_c_ballistic_arc_length_closed_form(ideal, v0, td):
    """Magnus does no work and has no z-component, so the speed profile stays ballistic."""
    th = np.radians(td)
    r = fly(initial_state(v0, th, SPIN), ideal, t_max_s=2000.0, stop_on="height")
    exact = ballistic_arc_length_m(v0, th, ideal.g_m_per_s2)
    assert abs(r["y_height"][10] / exact - 1.0) < 1e-8


def test_shape_factor_matches_log_form():
    th = np.radians(np.linspace(1.0, 89.0, 50))
    logform = np.sin(th) + np.cos(th) ** 2 * np.log(np.tan(th) + 1.0 / np.cos(th))
    assert np.allclose(shape_factor(th), logform, rtol=0, atol=1e-14)


def test_flight_time_unaffected_by_magnus(ideal):
    """Vertical spin -> Magnus has no z-component -> vz is purely ballistic."""
    v0, th = 55.0, np.radians(38.0)
    r = fly(initial_state(v0, th, SPIN), ideal, t_max_s=2000.0, stop_on="height")
    assert abs(r["t_height_s"] - 2 * v0 * np.sin(th) / ideal.g_m_per_s2) < 1e-9


def test_horizontal_speed_conserved(ideal):
    v0, th = 45.0, np.radians(50.0)
    r = fly(initial_state(v0, th, SPIN), ideal, t_max_s=2000.0, stop_on="height")
    ts = np.linspace(0.0, r["t_height_s"], 500)
    Y = r["sol"].sol(ts)
    assert np.abs(np.hypot(Y[3], Y[4]) / (v0 * np.cos(th)) - 1.0).max() < 1e-9


@pytest.mark.parametrize("td", [15.0, 35.0, 60.0, 80.0])
def test_closed_form_v0_solves_the_ideal_closure(ideal, td):
    th = np.radians(td)
    assert abs(solve_v0(th, ideal) / float(v0_closed_form(th, ideal)) - 1.0) < 1e-9


def test_no_silent_clamping_on_divergence():
    """A hopeless case must raise, not return a fudged number."""
    from src.integrate import DivergedError
    p = make_ball("baseball", drag_on=True)
    with pytest.raises(DivergedError):
        solve_v0(np.radians(45.0), p)

"""Regression tests for the Task 3 / Task 5 findings, including two refuted claims."""
import numpy as np
import pytest

from src.model import make_ball, initial_state
from src.integrate import fly
from src.solve import solve_v0
from src import sweep

SPIN = 200.0


@pytest.fixture(scope="module")
def ideal():
    return make_ball("baseball", drag_on=False)


@pytest.fixture(scope="module")
def pair45(ideal):
    th = np.radians(45.0)
    return sweep.mirror_pair(solve_v0(th, ideal, v0_cap_m_per_s=1000.0), th, ideal)


def test_mirror_pair_arrives_together(pair45):
    assert pair45["dt_s"] < 1e-9
    assert pair45["miss_m"] < 1e-6          # event-location limited; see state test below


def test_mirror_symmetry_is_exact_at_common_times(pair45):
    """The sharp test: B(t) == reflect(A(t)).  Free of event-location error."""
    pos_err, vel_err, scale = sweep.mirror_state_error(pair45)
    assert pos_err / scale < 1e-10
    assert vel_err < 1e-8


def test_meeting_point_is_perpendicular_to_launch(pair45):
    """Launch bearing 0 deg; the chord to the meeting point comes out at exactly 90 deg."""
    P = pair45["yA"]
    assert abs(np.degrees(np.arctan2(P[1], P[0])) - 90.0) < 1e-8
    assert abs(P[2]) < 1e-9                 # back at launch height


@pytest.mark.parametrize("td", [10.0, 30.0, 45.0, 60.0, 80.0])
def test_closing_speed_is_2cos_theta_not_2(ideal, td):
    """REFUTES 'closing speed = 2|v_f|'.  Vertical velocity is common mode and cancels."""
    th = np.radians(td)
    b = sweep.mirror_pair(solve_v0(th, ideal, v0_cap_m_per_s=1000.0), th, ideal)
    ratio = b["closing_speed_m_per_s"] / b["vfA_m_per_s"]
    assert abs(ratio / (2.0 * np.cos(th)) - 1.0) < 1e-9
    if td > 1.0:
        assert abs(ratio - 2.0) > 1e-3      # the claimed value is genuinely wrong


def test_spin_magnitude_is_inert_under_constant_cl(ideal):
    """With C_L const, Magnus depends on omega_hat only, so |omega| cannot matter."""
    th, v0 = np.radians(45.0), 86.888703
    ends = [fly(initial_state(v0, th, w), ideal, t_max_s=100.0, stop_on="height")["y_height"][:3]
            for w in (50.0, 200.0, 800.0)]
    assert np.abs(ends[0] - ends[1]).max() < 1e-9
    assert np.abs(ends[0] - ends[2]).max() < 1e-9


def test_spin_magnitude_matters_under_saturating_cl():
    p = make_ball("baseball", drag_on=False, cl_model="saturating")
    th, v0 = np.radians(45.0), 86.888703
    a = fly(initial_state(v0, th, 200.0), p, t_max_s=100.0, stop_on="height")["y_height"]
    b = fly(initial_state(v0, th, 220.0), p, t_max_s=100.0, stop_on="height")["y_height"]
    assert np.linalg.norm(a[:3] - b[:3]) > 1.0


@pytest.mark.parametrize("td", [20.0, 45.0, 70.0])
def test_curvature_ratio_is_sec_theta(ideal, td):
    """Ground track is an oval: R_h = |v_h|/(k_L|v|), so R_max/R_min = sec(theta) exactly."""
    th = np.radians(td)
    v0 = solve_v0(th, ideal, v0_cap_m_per_s=1000.0)
    r = fly(initial_state(v0, th, SPIN), ideal,
            t_max_s=1.5 * 2 * v0 * np.sin(th) / ideal.g_m_per_s2 + 5, stop_on="height")
    _, _, Rh = sweep.ground_track_curvature(r["sol"], r["t_height_s"], ideal)
    assert abs(Rh.max() / Rh.min() / (1.0 / np.cos(th)) - 1.0) < 1e-6


def test_ground_track_is_not_a_circle(ideal):
    """REFUTES nothing -- confirms the user's claim that the track is an oval."""
    th = np.radians(45.0)
    v0 = solve_v0(th, ideal, v0_cap_m_per_s=1000.0)
    r = fly(initial_state(v0, th, SPIN), ideal,
            t_max_s=1.5 * 2 * v0 * np.sin(th) / ideal.g_m_per_s2 + 5, stop_on="height")
    _, Y, _ = sweep.ground_track_curvature(r["sol"], r["t_height_s"], ideal)
    *_, R, rms = sweep.fit_circle(Y[0], Y[1])
    assert rms / R > 1e-3                   # a genuine, resolvable departure from circular

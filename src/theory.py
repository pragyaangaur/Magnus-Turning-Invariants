"""Exact analytic structure, and the asymmetric collision that drag forces.

Central identity (constant C_L, C_D, vertical spin axis).  Writing the horizontal
velocity as a complex number w = v_x + i v_y, the horizontal equation of motion is

    dw/dt = (i k_L - k_D) |v| w        =>       dw/ds = (i k_L - k_D) w

because drag is antiparallel to v and Magnus is i*(horizontal v).  In arc length s
this is LINEAR with constant coefficients, so

    w(s) = w_0 exp(-(k_D - i k_L) s)

exactly, whatever gravity is doing.  Consequences:
  * psi(s) = psi_0 + k_L s                      (the turning law, drag-independent)
  * |w(s)| = |w_0| exp(-k_D s)                  (horizontal speed, gravity-independent)
  * |w_f|/|w_0| = exp(-(C_D/C_L) * dpsi)        (hodograph is a logarithmic spiral)
  * R_h = L_L cos(gamma)                        (ground-track curvature; gamma = flight-path angle)
"""
import numpy as np
from scipy.optimize import least_squares

from .model import initial_state, Params
from .integrate import fly, DivergedError


def horizontal_speed_law(p: Params, dpsi):
    """|w_f|/|w_0| = exp(-(C_D/C_L)*dpsi).  Exact; at dpsi=pi this is exp(-pi C_D/C_L)."""
    return np.exp(-(p.cd_const / p.cl_const) * np.asarray(dpsi, dtype=float))


def _end(v0, theta, p, azimuth, spin_sign, spin=200.0):
    y0 = initial_state(v0, theta, spin, azimuth_rad=azimuth, spin_sign=spin_sign)
    t_max = 1.5 * (2.0 * v0 * np.sin(theta) / p.g_m_per_s2) + 5.0
    r = fly(y0, p, t_max_s=t_max, stop_on="height")
    if not np.isfinite(r["t_height_s"]):
        raise DivergedError(f"no return to launch height at v0={v0:.3f}, "
                            f"theta={np.degrees(theta):.2f}deg")
    return r["t_height_s"], r["y_height"]


def chord_bearing_deg(v0, theta, p, spin=200.0):
    """Bearing of the launch->landing chord, for a ball launched along bearing 0."""
    _, y = _end(v0, theta, p, 0.0, +1.0, spin)
    return float(np.degrees(np.arctan2(y[1], y[0])))


def asymmetric_residuals(u, p, spin=200.0, scale=None):
    """Head-on collision of two OPPOSITELY thrown balls, without assuming a mirror pair.

    Unknowns u = (v0_A, theta_A, v0_B, theta_B).  Ball A is launched along bearing 0
    with +z spin, ball B along bearing pi with -z spin.  Conditions:
        0: they return to launch height at the same instant
        1,2: they are at the same horizontal point then
        3: their headings are anti-parallel (head-on)
    With drag the turns satisfy dpsi_A + dpsi_B = 2*pi, NOT dpsi_A = dpsi_B = pi.
    """
    v0A, thA, v0B, thB = u
    tA, yA = _end(v0A, thA, p, 0.0, +1.0, spin)
    tB, yB = _end(v0B, thB, p, np.pi, -1.0, spin)
    T, L = (1.0, 1.0) if scale is None else scale
    return np.array([(tA - tB) / T, (yA[0] - yB[0]) / L, (yA[1] - yB[1]) / L,
                     (yA[9] - yB[9]) - np.pi])


def solve_asymmetric(p, seed, spin=200.0, tol=1e-9):
    """Root-find the drag collision with unequal turns.  seed = (v0A, thetaA, v0B, thetaB).

    Raises rather than returning a near-miss if it does not converge.  In practice every
    seed tried drains to the degenerate theta -> 0 corner and stalls near 1.5 mm; see the
    README section "The one thing that did not work out".
    """
    seed = np.asarray(seed, dtype=float)
    t0, y0 = _end(seed[0], seed[1], p, 0.0, +1.0, spin)
    scale = (t0, float(np.hypot(y0[0], y0[1])))

    def F(u):
        return asymmetric_residuals(u, p, spin, scale)

    vmax = 20.0 * max(seed[0], seed[2])
    lo = np.array([0.5, np.radians(0.5), 0.5, np.radians(0.5)])
    hi = np.array([vmax, np.radians(89.5), vmax, np.radians(89.5)])
    sol = least_squares(F, seed, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14,
                        diff_step=1e-5)
    if np.max(np.abs(sol.fun)) > tol:
        raise DivergedError(f"asymmetric solve failed: |F|={np.abs(sol.fun).max():.3e}")
    v0A, thA, v0B, thB = sol.x
    tA, yA = _end(v0A, thA, p, 0.0, +1.0, spin)
    tB, yB = _end(v0B, thB, p, np.pi, -1.0, spin)
    return dict(
        v0A_m_per_s=v0A, theta_A_deg=np.degrees(thA),
        v0B_m_per_s=v0B, theta_B_deg=np.degrees(thB),
        dpsi_A_deg=np.degrees(yA[9]), dpsi_B_deg=np.degrees(np.pi - yB[9]),
        dpsi_sum_deg=np.degrees(yA[9] + (np.pi - yB[9])),
        t_collision_s=tA, miss_m=float(np.linalg.norm(yA[0:3] - yB[0:3])),
        dt_s=abs(tA - tB),
        meet_xy=(float(yA[0]), float(yA[1])),
        closing_speed_m_per_s=float(np.linalg.norm(yA[3:6] - yB[3:6])),
        vfA=float(np.linalg.norm(yA[3:6])), vfB=float(np.linalg.norm(yB[3:6])),
        residual=float(np.abs(sol.fun).max()),
    )

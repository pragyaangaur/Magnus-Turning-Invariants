"""Closure problem: find (v0, theta) so that return-to-launch-height and psi=pi coincide."""
import numpy as np
from scipy.optimize import brentq, root

from .model import initial_state, v0_closed_form
from .integrate import fly, DivergedError

SPIN_RAD_PER_S = 200.0


def _flight(v0, theta, p, spin=SPIN_RAD_PER_S, t_max_s=None, rtol=1e-10):
    """Integrate to the downward launch-height crossing (terminal).  psi=pi is logged en route."""
    if t_max_s is None:
        # drag only shortens the hang time, so the vacuum value is a safe ceiling
        t_max_s = 1.5 * (2.0 * v0 * np.sin(theta) / p.g_m_per_s2) + 5.0
    y0 = initial_state(v0, theta, spin)
    return fly(y0, p, t_max_s=t_max_s, stop_on="height", rtol=rtol)


def residual_1d(v0, theta, p, spin=SPIN_RAD_PER_S):
    """psi at the moment the ball returns to launch height, minus pi.

    Zero <=> both events fire together.  This is the whole closure condition.
    """
    r = _flight(v0, theta, p, spin)
    if not np.isfinite(r["t_height_s"]):
        raise DivergedError(f"no return to launch height for v0={v0:.3f}, theta={np.degrees(theta):.2f}deg")
    return r["y_height"][9] - np.pi


def residuals_2d(u, p, spin=SPIN_RAD_PER_S):
    """The user's 2-component system, evaluated at the two event times.

    F1 = z(t_psi) - z_launch,  F2 = psi(t_height) - pi.
    Both vanish iff t_height == t_psi, so the two rows are not independent -- see
    the Jacobian singular values reported by solve_2d.
    """
    v0, theta = u
    first = _flight(v0, theta, p, spin)
    t_h = first["t_height_s"]
    if not np.isfinite(t_h):
        raise DivergedError(f"no return to launch height at v0={v0:.3f}, theta={np.degrees(theta):.2f}deg")
    # re-run without a terminal event, a little past t_h, so the psi=pi event is captured
    # even when it lands marginally after the height crossing
    r = fly(initial_state(v0, theta, spin), p, t_max_s=1.25 * t_h, stop_on=None, rtol=1e-10)
    if not np.isfinite(r["t_psi_s"]):
        raise DivergedError(
            f"psi never reaches pi before 1.25*t_height at v0={v0:.3f}, "
            f"theta={np.degrees(theta):.2f}deg (psi_max={r['sol'].y[9, -1]:.4f})")
    return np.array([r["y_psi"][2] - 0.0, r["y_height"][9] - np.pi])


def solve_v0(theta, p, spin=SPIN_RAD_PER_S, bracket=None, v0_cap_m_per_s=150.0):
    """1D solve for v0 at fixed theta.

    Expands the upper bracket geometrically up to `v0_cap_m_per_s`.  Raises rather
    than returning anything approximate if no sign change exists below the cap.
    """
    seed = float(v0_closed_form(theta, p))
    if bracket is not None:
        lo, hi = bracket
    else:
        lo, hi = 0.1 * seed, min(4.0 * seed, v0_cap_m_per_s)
    f_lo = residual_1d(lo, theta, p, spin)
    f_hi = residual_1d(hi, theta, p, spin)
    while f_lo * f_hi > 0.0 and hi < v0_cap_m_per_s:
        hi = min(2.0 * hi, v0_cap_m_per_s)
        f_hi = residual_1d(hi, theta, p, spin)
    if f_lo * f_hi > 0.0:
        raise DivergedError(
            f"no closure for theta={np.degrees(theta):.2f}deg below v0={v0_cap_m_per_s} m/s: "
            f"R({lo:.2f})={f_lo:+.4f}, R({hi:.2f})={f_hi:+.4f} "
            f"(psi_max = {f_hi + np.pi:.4f} rad = {(f_hi + np.pi)/np.pi:.4f} pi)")
    return brentq(residual_1d, lo, hi, args=(theta, p, spin), xtol=1e-12, rtol=1e-13)


def solve_2d(seed_v0, seed_theta, p, spin=SPIN_RAD_PER_S):
    """Run scipy.optimize.root on the 2x2 system and report the Jacobian conditioning.

    The finite-difference Jacobian is evaluated at the solution scipy lands on.
    """
    def F(u):
        return residuals_2d(u, p, spin)
    sol = root(F, np.array([seed_v0, seed_theta]), method="hybr",
               options=dict(eps=1e-5, xtol=1e-12))
    J = _fd_jacobian(F, sol.x, eps=1e-5)
    sv = np.linalg.svd(J, compute_uv=False)
    return sol, J, sv


def _fd_jacobian(F, u, eps=1e-6):
    f0 = F(u)
    J = np.empty((len(f0), len(u)))
    for j in range(len(u)):
        du = np.zeros_like(u)
        du[j] = eps * max(1.0, abs(u[j]))
        J[:, j] = (F(u + du) - f0) / du[j]
    return J


def max_turn_scan(p, v0_grid, theta_grid, spin=SPIN_RAD_PER_S):
    """psi(t_height) over a (v0, theta) grid.  max over the grid vs pi tells feasibility."""
    P = np.full((len(v0_grid), len(theta_grid)), np.nan)
    for i, v0 in enumerate(v0_grid):
        for j, th in enumerate(theta_grid):
            try:
                P[i, j] = residual_1d(v0, th, p, spin) + np.pi
            except DivergedError:
                pass
    return P


def closure_report(p, theta_deg_grid=np.arange(10.0, 86.0, 5.0), spin=SPIN_RAD_PER_S,
                   v0_cap_m_per_s=150.0):
    """Solve the family theta -> v0.  Returns list of dicts; empty if infeasible."""
    out = []
    for td in theta_deg_grid:
        th = np.radians(td)
        try:
            v0 = solve_v0(th, p, spin, v0_cap_m_per_s=v0_cap_m_per_s)
        except (DivergedError, ValueError):
            continue
        r = _flight(v0, th, p, spin)
        yh = r["y_height"]
        vf = float(np.linalg.norm(yh[3:6]))
        out.append(dict(theta_deg=td, v0_m_per_s=v0, t_collision_s=r["t_height_s"],
                        radius_m=float(np.hypot(yh[0], yh[1])) / 2.0,
                        path_m=float(yh[10]), vf_m_per_s=vf, vf_over_v0=vf / v0))
    return out

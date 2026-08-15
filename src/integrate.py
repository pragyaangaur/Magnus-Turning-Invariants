"""Integrators.  RK45 with event detection for single trajectories; batched RK4 for sweeps."""
import numpy as np
from scipy.integrate import solve_ivp

from .model import rhs, Params


class DivergedError(RuntimeError):
    pass


def _events(z_launch_m, psi_target_rad):
    def ev_height(t, y, p):
        return y[2] - z_launch_m
    ev_height.direction = -1.0          # downward crossing only  => vz < 0
    ev_height.terminal = False

    def ev_psi(t, y, p):
        return y[9] - psi_target_rad
    ev_psi.direction = 0.0
    ev_psi.terminal = False
    return ev_height, ev_psi


def fly(y0, p: Params, t_max_s=60.0, psi_target_rad=np.pi, rtol=1e-10, atol=1e-12,
        dense=True, stop_on="height"):
    """Integrate one trajectory with RK45 + event detection.

    Returns dict with the solution and the two event times (nan if never reached).
    `stop_on` in {"height", "psi", None} selects which event terminates the run.
    """
    z_launch = y0[2]
    ev_h, ev_psi = _events(z_launch, psi_target_rad)
    if stop_on == "height":
        ev_h.terminal = True
    elif stop_on == "psi":
        ev_psi.terminal = True
    elif stop_on is not None:
        raise ValueError(stop_on)

    sol = solve_ivp(rhs, (0.0, t_max_s), y0, args=(p,), method="RK45",
                    rtol=rtol, atol=atol, events=(ev_h, ev_psi), dense_output=dense,
                    max_step=t_max_s / 50.0)
    if not sol.success:
        raise DivergedError(f"solve_ivp failed: {sol.message}")

    t_h = sol.t_events[0][0] if len(sol.t_events[0]) else np.nan
    t_psi = sol.t_events[1][0] if len(sol.t_events[1]) else np.nan
    return dict(sol=sol,
                t_height_s=t_h, t_psi_s=t_psi,
                y_height=sol.y_events[0][0] if len(sol.y_events[0]) else None,
                y_psi=sol.y_events[1][0] if len(sol.y_events[1]) else None)


def rk4(y0, p: Params, t_end_s, n_steps):
    """Fixed-step RK4.  y0 may be (11,) or (11, N) for a batch of trajectories."""
    y = np.array(y0, dtype=float)
    h = t_end_s / n_steps
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1,) + y.shape)
    ts[0], ys[0] = 0.0, y
    t = 0.0
    for i in range(n_steps):
        k1 = rhs(t, y, p)
        k2 = rhs(t + h / 2, y + h / 2 * k1, p)
        k3 = rhs(t + h / 2, y + h / 2 * k2, p)
        k4 = rhs(t + h, y + h * k3, p)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            raise DivergedError(f"RK4 produced non-finite state at step {i+1}")
        ts[i + 1], ys[i + 1] = t, y
    return ts, ys


def batched_crossing(ts, ys, comp, level, downward=True):
    """Linear-in-time refinement of the first `comp`-crosses-`level` event per column.

    ys has shape (nt, 11, N).  Returns (t_cross (N,), y_cross (11, N)); nan where absent.
    """
    f = ys[:, comp, :] - level
    if downward:
        hit = (f[:-1] > 0.0) & (f[1:] <= 0.0)
    else:
        hit = (f[:-1] < 0.0) & (f[1:] >= 0.0)
    n = f.shape[1]
    idx = np.where(hit.any(axis=0), hit.argmax(axis=0), -1)
    cols = np.arange(n)
    ok = idx >= 0
    i0 = np.where(ok, idx, 0)
    f0, f1 = f[i0, cols], f[i0 + 1, cols]
    frac = np.where(f1 != f0, f0 / (f0 - f1), 0.0)
    t_c = ts[i0] + frac * (ts[i0 + 1] - ts[i0])
    y_c = ys[i0, :, cols].T + frac * (ys[i0 + 1, :, cols].T - ys[i0, :, cols].T)
    t_c = np.where(ok, t_c, np.nan)
    y_c = np.where(ok, y_c, np.nan)
    return t_c, y_c

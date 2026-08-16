"""Exact reduction of the full system to a single scalar ODE in the turn angle.

With u = psi (azimuth turned) as the independent variable and Q = tan(gamma) the
tangent of the flight-path angle, the 9-dimensional system collapses exactly to

    dQ/du = -lambda * exp(2*mu*u) / sqrt(1 + Q^2),      Q(0) = tan(theta)

    lambda = g / (k_L * v0^2 * cos^2(theta)),           mu = C_D / C_L

Derivation: dv_z/dt = -g - k_D|v|v_z (Magnus has no z-component) and dpsi/dt =
k_L|v|, so dv_z/du + mu*v_z = -g/(k_L|v|).  The integrating factor e^{mu u} turns
the left side into d(v_z e^{mu u})/du, and |w| = |w_0|e^{-mu u} exactly (see
theory.py) gives e^{mu u}/|v| = e^{2 mu u} cos(gamma)/|w_0|.

Two functionals of the solution carry the geometry, both exact:

    closure (back to launch height at u = pi):   INT_0^pi sin(gamma) du = 0
    chord perpendicular to launch direction:     INT_0^pi cos(gamma) cos(u) du = 0

Because closure fixes lambda once (mu, theta) are given, the chord bearing at
closure is a function of (mu, theta) ALONE: it cannot depend on C_L and C_D
separately, nor on the ball's mass or size.
"""
import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import brentq

from .integrate import DivergedError


def lambda_dragfree(theta_rad):
    """Exact drag-free closure value.  The mu=0 ODE is separable, giving
    (1/2)[Q sqrt(1+Q^2) + asinh Q] = const - lambda*u, and Q(pi) = -tan(theta)
    then forces lambda = [tan(th)sec(th) + asinh(tan th)]/pi.
    """
    t = np.tan(theta_rad)
    return (t / np.cos(theta_rad) + np.arcsinh(t)) / np.pi


def solve_Q(theta_rad, lam, mu, u_end=np.pi, n=1201, rtol=1e-12, atol=1e-14):
    """Integrate the reduced ODE.  Returns (u grid, Q, gamma)."""
    def f(u, Q):
        return -lam * np.exp(2.0 * mu * u) / np.sqrt(1.0 + Q[0] ** 2)

    sol = solve_ivp(f, (0.0, u_end), [np.tan(theta_rad)],
                    rtol=rtol, atol=atol, dense_output=True)
    if not sol.success:
        raise DivergedError(f"reduced ODE failed: {sol.message}")
    u = np.linspace(0.0, u_end, n)
    Q = sol.sol(u)[0]
    return u, Q, np.arctan(Q)


def height_functional(lam, theta_rad, mu):
    """INT_0^pi sin(gamma) du.  Zero exactly when the ball is back at launch height.

    `lam` comes first so this can be passed straight to a 1-D root finder.
    """
    u, _, gam = solve_Q(theta_rad, lam, mu)
    return simpson(np.sin(gam), x=u)


def solve_lambda(theta_rad, mu, seed=None):
    """Find lambda such that the ball returns to launch height exactly at u = pi.

    Seeded from the drag-free closed form and expanded geometrically, which keeps
    the bracket a few multiples wide instead of six decades.
    """
    lam0 = lambda_dragfree(theta_rad) if seed is None else seed
    lo, hi = 0.5 * lam0, 2.0 * lam0
    f_lo = height_functional(lo, theta_rad, mu)
    f_hi = height_functional(hi, theta_rad, mu)
    for _ in range(60):
        if f_lo * f_hi <= 0.0:
            break
        if abs(f_lo) < abs(f_hi):
            lo *= 0.5
            f_lo = height_functional(lo, theta_rad, mu)
        else:
            hi *= 2.0
            f_hi = height_functional(hi, theta_rad, mu)
    else:
        raise DivergedError(
            f"no closure bracket at theta={np.degrees(theta_rad):.3f}deg, mu={mu:.4f}")
    return brentq(height_functional, lo, hi, args=(theta_rad, mu), xtol=1e-14, rtol=8.9e-16)


def chord_functional(lam, theta_rad, mu):
    """(Re, Im) of INT_0^pi cos(gamma) e^{iu} du.  Re = 0 means chord perpendicular."""
    u, _, gam = solve_Q(theta_rad, lam, mu)
    c = np.cos(gam)
    return simpson(c * np.cos(u), x=u), simpson(c * np.sin(u), x=u)


def bearing_deg(theta_rad, mu, seed=None):
    """Chord bearing at closure, in degrees, as a function of (mu, theta) only."""
    lam = solve_lambda(theta_rad, mu, seed=seed)
    re, im = chord_functional(lam, theta_rad, mu)
    return float(np.degrees(np.arctan2(im, re))), lam


def deficit_deg(theta_rad, mu, seed=None):
    """90 deg minus the chord bearing.  Positive means the rendezvous fails."""
    b, _ = bearing_deg(theta_rad, mu, seed=seed)
    return 90.0 - b


# ---------------------------------------------------------------- batched version
def _batch_functionals(theta, lam, mu, n_steps=600):
    """Fixed-step RK4 on the reduced ODE for a whole array of (theta, lam, mu).

    Returns (H, Re, Im) where H = INT sin(gamma) du is the closure functional and
    (Re, Im) = INT cos(gamma) (cos u, sin u) du give the chord.  All inputs
    broadcast to a common shape; integration runs on u in [0, pi].
    """
    theta, lam, mu = np.broadcast_arrays(theta, lam, mu)
    h = np.pi / n_steps
    Q = np.tan(theta)

    def f(u, Q):
        return -lam * np.exp(2.0 * mu * u) / np.sqrt(1.0 + Q ** 2)

    sin_g = np.empty((n_steps + 1,) + Q.shape)
    cos_g = np.empty_like(sin_g)
    us = np.linspace(0.0, np.pi, n_steps + 1)
    for i, u in enumerate(us):
        r = np.sqrt(1.0 + Q ** 2)
        sin_g[i], cos_g[i] = Q / r, 1.0 / r
        if i == n_steps:
            break
        k1 = f(u, Q)
        k2 = f(u + h / 2, Q + h / 2 * k1)
        k3 = f(u + h / 2, Q + h / 2 * k2)
        k4 = f(u + h, Q + h * k3)
        Q = Q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    cu = np.cos(us).reshape((-1,) + (1,) * Q.ndim)
    su = np.sin(us).reshape((-1,) + (1,) * Q.ndim)
    quad = lambda y: simpson(y, dx=h, axis=0)
    return quad(sin_g), quad(cos_g * cu), quad(cos_g * su)


def deficit_grid_deg(theta_grid_rad, mu_grid, n_bisect=60, n_steps=600):
    """Chord-bearing deficit over a (mu, theta) grid, vectorised.

    Bisects the closure condition H(lambda) = 0 for every grid point at once.
    H is strictly decreasing in lambda, so plain bisection is safe here.
    """
    MU, TH = np.meshgrid(mu_grid, theta_grid_rad, indexing="ij")
    # H(lambda) is strictly decreasing, with H -> pi*sin(theta) > 0 as lambda -> 0
    # and H -> -pi as lambda -> infinity, so a root always exists.  Heavy drag pushes
    # the root many decades below the drag-free value, so bisect in log(lambda).
    llo = np.full(TH.shape, np.log(1e-12))
    lhi = np.full(TH.shape, np.log(1e3))
    for _ in range(n_bisect):
        lmid = 0.5 * (llo + lhi)
        H, _, _ = _batch_functionals(TH, np.exp(lmid), MU, n_steps)
        too_high = H < 0.0          # overshot: ball already below launch height
        lhi = np.where(too_high, lmid, lhi)
        llo = np.where(too_high, llo, lmid)
    lam = np.exp(0.5 * (llo + lhi))
    _, re, im = _batch_functionals(TH, lam, MU, n_steps)
    return 90.0 - np.degrees(np.arctan2(im, re)), lam

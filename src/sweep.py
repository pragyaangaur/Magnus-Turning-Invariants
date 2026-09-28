"""Sweeps, symmetry checks, feasibility map, shape analysis, and all figures."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import Params, make_ball, initial_state, rhs, BALLS
from .integrate import fly, rk4, batched_crossing, DivergedError
from .solve import SPIN_RAD_PER_S

FIG = "figures"

# Figure geometry, in inches.  These are the sizes used for the repository
# figures; override FIG and FIGSIZE before calling the fig_* helpers to retarget
# a different layout, for example to draw each figure at its final printed width
# so that the smallest label stays at or above an 8 pt floor.
FIGSIZE = {
    "invariants": (10, 4),
    "feasibility": (12, 4.6),
    "residual_speed": (6, 4.2),
    "ground_track": (11, 4.6),
    "trajectory_pair": (6.5, 5),
    "deficit": (13.5, 4.0),
}


def use_ajp_style():
    """Journal-figure defaults: Times-like serif, 8 pt floor, framed axes.

    AJP asks for a Times face for consistency with the running text, a smallest
    label of 8 pt at the printed width, and axes drawn as a closed frame with
    tick marks on all four sides.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ---------------------------------------------------------------- mirror symmetry
def mirror_pair(v0, theta, p, spin=SPIN_RAD_PER_S, dv0=0.0, dazi_rad=0.0, dspin=0.0):
    """Ball A at azimuth 0 with +z spin; ball B at azimuth pi with -z spin.

    That pair is the reflection of one another through the vertical plane containing
    the launch-to-meeting chord, which is the symmetry that makes them collide.
    Perturbations (dv0, dazi_rad, dspin) are applied to ball B only.
    """
    yA = initial_state(v0, theta, spin, azimuth_rad=0.0, spin_sign=+1.0)
    yB = initial_state(v0 * (1 + dv0), theta, spin * (1 + dspin),
                       azimuth_rad=np.pi + dazi_rad, spin_sign=-1.0)
    rA = fly(yA, p, t_max_s=1.5 * 2 * v0 * np.sin(theta) / p.g_m_per_s2 + 5, stop_on="height")
    rB = fly(yB, p, t_max_s=1.5 * 2 * v0 * np.sin(theta) / p.g_m_per_s2 + 5, stop_on="height")
    A, B = rA["y_height"], rB["y_height"]
    return dict(
        rA=rA, rB=rB, yA=A, yB=B,
        miss_m=float(np.linalg.norm(A[0:3] - B[0:3])),
        dt_s=abs(rA["t_height_s"] - rB["t_height_s"]),
        closing_speed_m_per_s=float(np.linalg.norm(A[3:6] - B[3:6])),
        vfA_m_per_s=float(np.linalg.norm(A[3:6])),
    )


def mirror_state_error(pair, n=400):
    """Ball B should be ball A reflected in x at every instant: the exact ODE symmetry.

    Checking states at common times avoids the event-location error that dominates
    the endpoint comparison, so this is the sharp test of the symmetry.
    """
    tA, tB = pair["rA"]["t_height_s"], pair["rB"]["t_height_s"]
    ts = np.linspace(0.0, min(tA, tB), n)
    A = pair["rA"]["sol"].sol(ts)
    B = pair["rB"]["sol"].sol(ts)
    M = A.copy()
    M[0] *= -1.0; M[3] *= -1.0; M[8] *= -1.0           # x, vx, wz flip
    M[9] = np.pi - A[9]                                # heading reflects as psi -> pi - psi
    pos_err = np.abs(M[0:3] - B[0:3]).max()
    vel_err = np.abs(M[3:6] - B[3:6]).max()
    return float(pos_err), float(vel_err), float(np.abs(M[0:3]).max())


def symmetry_table(v0, theta, p, p_spin=None):
    """Exact-mirror check plus the three deliberate symmetry breaks.

    The spin mismatch is run under `p_spin` (a saturating-C_L model) because with a
    constant C_L the Magnus term depends on the spin *direction* only, so |omega|
    cannot matter and the row would be a trivial zero.
    """
    base = mirror_pair(v0, theta, p)
    rows = [("+5% speed on B", mirror_pair(v0, theta, p, dv0=0.05), p),
            ("+2 deg azimuth on B", mirror_pair(v0, theta, p, dazi_rad=np.radians(2.0)), p),
            ("+10% spin on B (const C_L)", mirror_pair(v0, theta, p, dspin=0.10), p)]
    if p_spin is not None:
        rows.append(("+10% spin on B (saturating C_L)",
                     mirror_pair(v0, theta, p_spin, dspin=0.10), p_spin))
    return base, rows


# ---------------------------------------------------------------- feasibility map
def dimensionless_flight_grid(vhat_grid, theta_grid, n_steps=4000):
    """One batched RK4 over the whole (v0/vt, theta) grid.

    Units: k_D = g = 1, so terminal speed vt = 1 and drag length L_D = 1.
    Magnus does no work and has no z-component, so |v| and vz -- and hence the
    path length G = S/L_D and v_f/v_0 -- are exactly independent of C_L.
    Returns G and vf/v0, each shaped (len(vhat), len(theta)).
    """
    p = Params(mass_kg=0.5, radius_m=np.sqrt(1.0 / (np.pi * 1.225)),  # gives rho*C_D*A/2m = 1
               rho_air_kg_per_m3=1.225, cd_const=1.0, g_m_per_s2=1.0,
               magnus_on=False, drag_on=True, name="dimensionless")
    assert abs(p.k_drag_per_m - 1.0) < 1e-12, p.k_drag_per_m

    V, TH = np.meshgrid(vhat_grid, theta_grid, indexing="ij")
    V, TH = V.ravel(), TH.ravel()
    Y0 = np.zeros((11, V.size))
    Y0[3] = V * np.cos(TH)
    Y0[5] = V * np.sin(TH)

    # ascent <= arctan(vhat) < pi/2; apex height <= 0.5*ln(1+vhat^2); descent at vt=1
    t_end = 4.0 + np.log(1.0 + vhat_grid.max() ** 2)
    ts, ys = rk4(Y0, p, t_end, n_steps)
    t_c, y_c = batched_crossing(ts, ys, comp=2, level=0.0, downward=True)
    if not np.all(np.isfinite(t_c)):
        raise DivergedError("some dimensionless trajectories never returned to launch height")
    G = y_c[10]
    vf = np.linalg.norm(y_c[3:6], axis=0)
    shape = (len(vhat_grid), len(theta_grid))
    return G.reshape(shape), (vf / V).reshape(shape)


def feasibility_map(cl_cd_grid, loading_grid, v0_cap_m_per_s=150.0,
                    rho=1.225, g=9.80665, n_vhat=90, n_theta=45):
    """psi_max = (C_L/C_D) * max_theta G(v0/vt, theta), capped at a physical v0.

    loading = m/(C_L*A) [kg/m^2];  vt = sqrt(2*g/rho * loading * C_L/C_D).
    Returns (psi_max, vf_over_v0_at_closure) on the grid.
    """
    vhat = np.linspace(0.05, 60.0, n_vhat)
    th = np.radians(np.linspace(2.0, 88.0, n_theta))
    G, VF = dimensionless_flight_grid(vhat, th)
    G_of_vhat = G.max(axis=1)                       # best theta at each vhat
    best_th = th[G.argmax(axis=1)]
    VF_of_vhat = VF[np.arange(len(vhat)), G.argmax(axis=1)]

    R, L = np.meshgrid(cl_cd_grid, loading_grid, indexing="ij")
    vt = np.sqrt(2.0 * g / rho * L * R)
    vhat_cap = np.clip(v0_cap_m_per_s / vt, vhat[0], vhat[-1])
    G_cap = np.interp(vhat_cap, vhat, G_of_vhat)
    psi_max = R * G_cap

    # v_f/v_0 at the vhat that just closes the loop (psi = pi), where reachable
    vhat_close = np.interp(np.pi / np.maximum(R, 1e-12), G_of_vhat, vhat,
                           left=vhat[0], right=np.nan)
    vf_close = np.interp(vhat_close, vhat, VF_of_vhat, left=np.nan, right=np.nan)
    return psi_max, vf_close, (vhat, G_of_vhat, best_th, VF_of_vhat)


def ball_points(rho=1.225):
    pts = {}
    for name in BALLS:
        p = make_ball(name)
        pts[name] = (p.cl_const / p.cd_const, p.mass_kg / (p.cl_const * p.area_m2))
    p = make_ball("frisbee")   # frisbee uses an effective C_L/C_D ~ 3
    pts["frisbee"] = (3.0, p.mass_kg / (p.cl_const * p.area_m2))
    return pts


# ---------------------------------------------------------------- ground-track shape
def fit_circle(x, y):
    """Algebraic (Kasa) circle fit.  Returns (xc, yc, R, rms_residual)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0] / 2.0, c[1] / 2.0
    R = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    res = np.hypot(x - xc, y - yc) - R
    return xc, yc, R, float(np.sqrt(np.mean(res ** 2)))


def ground_track_curvature(sol, t_end, p, n=2000):
    """Radius of curvature of the horizontal projection, sampled along the flight."""
    ts = np.linspace(0.0, t_end, n)
    Y = sol.sol(ts)
    d = rhs(0.0, Y, p)
    vh = np.hypot(Y[3], Y[4])
    dpsi_dt = d[9]
    return ts, Y, vh / np.abs(dpsi_dt)          # R_h = (ds_h/dt)/(dpsi/dt)


# ---------------------------------------------------------------- figures
def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)


def fig_invariants(p_ideal):
    kL, LL = p_ideal.k_lift_per_m, p_ideal.lift_length_m
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE["invariants"])
    for (v0, td), ls in zip([(30, 25), (60, 45), (100, 70)], ["-", "--", ":"]):
        r = fly(initial_state(v0, np.radians(td), SPIN_RAD_PER_S), p_ideal,
                t_max_s=4000.0, stop_on="psi")
        Y = r["sol"].sol(np.linspace(0, r["t_psi_s"], 2000))
        d = rhs(0.0, Y, p_ideal)
        ax[0].plot(Y[10], d[9] / d[10] / kL - 1.0, ls=ls,
                   label=f"$v_0$={v0}, $\\theta$={td}$^\\circ$")
    ax[0].set(xlabel="path length $s$ [m]", ylabel=r"$(\mathrm{d}\psi/\mathrm{d}s)/k_L - 1$",
              title="(a) turning rate")

    v0s = np.linspace(10, 120, 8)
    for td, mk in zip([10, 30, 50, 70, 85], ["o", "s", "^", "v", "d"]):
        S = [fly(initial_state(v, np.radians(td), SPIN_RAD_PER_S), p_ideal,
                 t_max_s=8000.0, stop_on="psi")["y_psi"][10] for v in v0s]
        ax[1].plot(v0s, np.array(S) / (np.pi * LL) - 1.0, marker=mk, ls="-", ms=3,
                   label=f"$\\theta$={td}$^\\circ$")
    ax[1].set(xlabel="$v_0$ [m/s]", ylabel=r"$S_{\rm tot}/(\pi L_L) - 1$",
              title=r"(b) path length of a $180^\circ$ turn")
    # Both panels are noise floors that fill their boxes, so open headroom above
    # the data rather than letting the key sit on top of it.
    for a in ax:
        lo, hi = a.get_ylim()
        a.set_ylim(lo, lo + 2.0 * (hi - lo))
        a.legend(loc="upper right", ncol=1, handlelength=1.8, borderpad=0.2,
                 labelspacing=0.25)
    _save(fig, "invariants")


def fig_feasibility(psi_max, vf_close, cl_cd, loading, pts):
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE["feasibility"])
    XL = r"ballistic loading $m/(C_L A)$ [kg/m$^2$]"
    # pcolormesh, not imshow: both axes are logarithmic, so the cells must carry their
    # true (log-spaced) coordinates rather than be stretched linearly across the box.
    for a, Z, cm, lab, ttl, vmax in (
            (ax[0], psi_max / np.pi, "viridis", r"$\psi_{max}/\pi$ (closure iff $\geq 1$)",
             r"closure feasibility ($v_0 \leq 150$ m/s)", 2.0),
            (ax[1], vf_close, "magma", r"$v_f/v_0$ at closure", "residual speed fraction", None)):
        im = a.pcolormesh(loading, cl_cd, Z, cmap=cm, vmin=0, vmax=vmax, shading="nearest")
        fig.colorbar(im, ax=a, label=lab)
        a.set(xscale="log", yscale="log", xlabel=XL, ylabel=r"$C_L/C_D$", title=ttl)
        for nm, (rr, ll) in pts.items():
            a.plot(ll, rr, "o", ms=6, mfc="none", mec="w", mew=1.5)
            a.annotate(nm, (ll, rr), color="w", fontsize=7,
                       xytext=(4, 4), textcoords="offset points")
    cs = ax[0].contour(loading, cl_cd, psi_max / np.pi, levels=[1.0], colors="w")
    ax[0].clabel(cs, fmt=r"$\psi_{max}=\pi$")
    ax[0].axhline(np.pi, color="r", ls="--", lw=1, label=r"predicted boundary $C_L/C_D=\pi$")
    ax[0].legend(fontsize=8, loc="upper right")
    _save(fig, "feasibility")


def fig_vf_law(vhat, G_of_vhat, VF_of_vhat):
    """Check v_f/v_0 = exp(-pi C_D/C_L) against the simulated value at closure."""
    R = np.linspace(0.3, 20.0, 400)                      # C_L/C_D
    vhat_close = np.interp(np.pi / R, G_of_vhat, vhat, left=np.nan, right=np.nan)
    vf_sim = np.interp(vhat_close, vhat, VF_of_vhat, left=np.nan, right=np.nan)
    fig, ax = plt.subplots(figsize=FIGSIZE["residual_speed"])
    ax.plot(R, np.exp(-np.pi / R), "r--", label=r"prediction $e^{-\pi C_D/C_L}$")
    ax.plot(R, vf_sim, "k-", label="simulated, at closure")
    ax.set_xlabel(r"$C_L/C_D$"); ax.set_ylabel(r"$v_f/v_0$")
    ax.set_xscale("log"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("residual speed law")
    _save(fig, "residual_speed")
    return R, vf_sim


def fig_shape_and_3d(v0, theta, p):
    y0 = initial_state(v0, theta, SPIN_RAD_PER_S)
    r = fly(y0, p, t_max_s=1.5 * 2 * v0 * np.sin(theta) / p.g_m_per_s2 + 5, stop_on="height")
    ts, Y, Rh = ground_track_curvature(r["sol"], r["t_height_s"], p)
    xc, yc, Rfit, rms = fit_circle(Y[0], Y[1])

    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE["ground_track"])
    tt = np.linspace(0, 2 * np.pi, 400)
    ax[0].plot(Y[0], Y[1], "k-", lw=1.4, label="ground track")
    ax[0].plot(xc + Rfit * np.cos(tt), yc + Rfit * np.sin(tt), "r--", lw=1.0,
               label=f"circle fit, RMS {rms:.2f} m")
    ax[0].plot([0, Y[0, -1]], [0, Y[1, -1]], "ko", ms=4, mfc="w")
    ax[0].set(xlabel="$x$ [m]", ylabel="$y$ [m]", title="(a) ground track and circle fit",
              aspect="equal")
    ax[0].legend(loc="upper left", handlelength=1.8, borderpad=0.2, labelspacing=0.25)
    ax[1].plot(Y[10], Rh, "k-")
    ax[1].set(xlabel="path length $s$ [m]", ylabel=r"$R_h$ [m]",
              title=r"(b) $R_h=L_L\cos\gamma$")
    _save(fig, "ground_track")

    pair = mirror_pair(v0, theta, p)
    fig = plt.figure(figsize=FIGSIZE["trajectory_pair"])
    ax3 = fig.add_subplot(111, projection="3d")
    for res, c, ls, lab in ((pair["rA"], "k", "-", "ball $A$ (spin $+\\hat{z}$)"),
                            (pair["rB"], "C3", "--", "ball $B$ (spin $-\\hat{z}$)")):
        tsx = np.linspace(0, res["t_height_s"], 800)
        W = res["sol"].sol(tsx)
        ax3.plot(W[0], W[1], W[2], color=c, ls=ls, lw=1.2, label=lab)
    P = pair["yA"]
    ax3.scatter([0], [0], [0], color="k", s=18, label="release")
    ax3.scatter([P[0]], [P[1]], [P[2]], color="C3", s=26, marker="*", label="meeting point")
    # 3-D panes carry three sets of ticks into a small box, so thin them out and
    # push the labels clear of the tick text.
    ax3.set(xlabel="$x$ [m]", ylabel="$y$ [m]", zlabel="$z$ [m]")
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))
        axis.set_minor_locator(plt.NullLocator())
    ax3.tick_params(labelsize=7, pad=-1.5)
    ax3.xaxis.labelpad = -3
    ax3.yaxis.labelpad = -3
    ax3.zaxis.labelpad = 2
    ax3.view_init(elev=22, azim=-58)
    ax3.legend(loc="upper left", bbox_to_anchor=(-0.06, 1.02), handlelength=1.6,
               borderpad=0.2, labelspacing=0.25)
    # tight_layout mismeasures a 3-D axes and pushes the z label off the canvas,
    # so reserve the margins by hand for this one figure.
    os.makedirs(FIG, exist_ok=True)
    fig.subplots_adjust(left=0.0, right=0.83, bottom=0.02, top=0.99)
    fig.savefig(f"{FIG}/trajectory_pair.pdf", bbox_inches=None)
    plt.close(fig)
    return dict(rms_circle_m=rms, R_fit_m=Rfit, R_ratio=float(Rh.max() / Rh.min()),
                sec_theta=float(1 / np.cos(theta)), y_end=Y[:, -1], pair=pair,
                t_s=r["t_height_s"])


# ---------------------------------------------------------------- reduced-model figure
def fig_deficit(mu_grid=None, theta_deg_grid=None):
    """Three-panel figure for the impossibility result, all from the reduced ODE."""
    from .reduced import deficit_deg, deficit_grid_deg, solve_lambda, solve_Q

    mu_grid = np.logspace(-2, np.log10(3.0), 60) if mu_grid is None else mu_grid
    theta_deg_grid = np.linspace(2.0, 88.0, 60) if theta_deg_grid is None else theta_deg_grid
    D, _ = deficit_grid_deg(np.radians(theta_deg_grid), mu_grid, n_bisect=70)

    fig, ax = plt.subplots(1, 3, figsize=FIGSIZE["deficit"])

    im = ax[0].pcolormesh(theta_deg_grid, mu_grid, D, cmap="magma", shading="nearest",
                          norm=matplotlib.colors.LogNorm(vmin=max(D.min(), 1e-4), vmax=D.max()))
    fig.colorbar(im, ax=ax[0], label=r"$90^\circ-\beta$  [deg]")
    for nm, p in ((n, make_ball(n)) for n in BALLS):
        ax[0].plot(45.0, p.cd_const / p.cl_const, "w+", ms=7, mew=1.4)
    ax[0].set(yscale="log", xlabel=r"$\theta$ [deg]", ylabel=r"$\mu=C_D/C_L$",
              title=r"(a) deficit $90^\circ-\beta>0$")

    # panel 2: the folded integrand, showing the pointwise inequality fails
    for mu, td, c, ls in ((1.0, 45.0, "k", "-"), (0.5, 45.0, "C0", "--"),
                          (1.5, 30.0, "C3", ":")):
        th = np.radians(td)
        u, _, gam = solve_Q(th, solve_lambda(th, mu), mu, n=4001)
        cg = np.cos(gam)
        half = (len(u) - 1) // 2 + 1
        d = cg[:half] - cg[::-1][:half]
        ax[1].plot(u[:half], d, color=c, ls=ls,
                   label=rf"$\mu$={mu}, $\theta$={td:.0f}$^\circ$")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set(xlabel=r"$u$ [rad]", ylabel=r"$\cos\gamma(u)-\cos\gamma(\pi-u)$",
              title="(b) folded integrand")
    ax[1].legend(handlelength=1.8, borderpad=0.2, labelspacing=0.25)

    # panel 3: asymptotic law
    C = 8.0 / 3.0 - 24.0 / np.pi ** 2
    mus = np.logspace(-3, -0.5, 12)
    th = np.radians(0.5)
    num = [np.radians(deficit_deg(th, m)) / (m * th ** 2) for m in mus]
    ax[2].semilogx(mus, num, "ko-", ms=3, label="reduced ODE")
    ax[2].axhline(C, color="r", ls="--", label=r"$8/3-24/\pi^2$")
    ax[2].set(xlabel=r"$\mu=C_D/C_L$", ylabel=r"$(90^\circ-\beta)/(\mu\theta^2)$ [rad]",
              title=r"(c) limit $\mu\to0$")
    ax[2].legend(handlelength=1.8, borderpad=0.2, labelspacing=0.25)
    _save(fig, "deficit")
    return D

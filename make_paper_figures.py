"""Figures for the paper, and every number its text quotes.

Run from the repository root:

    python make_paper_figures.py              # writes figures/manuscript/*.pdf
    python make_paper_figures.py OUT_DIR      # writes OUT_DIR/*.pdf instead

Main text
  geometry.pdf         forces, spin axis, heading psi and climb angle gamma
  same_turn.pdf        four soccer kicks: tracks, psi(t), psi(s), speeds against psi
  horizontal.pdf       table-tennis lob: hodograph, ground track, R_h along the path
  trajectory_pair.pdf  drag-free meeting of two counter-spinning balls
  deficit.pdf          how far the meeting fails with drag, and its small-drag limit
Supplementary material
  invariants.pdf       round-off level checks of the turning invariants
  folded.pdf           the folded integrand of the meeting condition
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc

from src import sweep
from src.model import make_ball, initial_state, rhs, BALLS
from src.integrate import fly
from src.solve import solve_v0, SPIN_RAD_PER_S
from src.reduced import deficit_deg, deficit_grid_deg, solve_lambda, solve_Q

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("figures", "manuscript")
os.makedirs(OUT, exist_ok=True)
sweep.FIG = OUT
sweep.use_ajp_style()
W = 6.8  # printed width in inches


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)


def flight(ball, v0, theta_deg, n=1500, **over):
    p = make_ball(ball, **over)
    th = np.radians(theta_deg)
    r = fly(initial_state(v0, th, SPIN_RAD_PER_S), p, t_max_s=40.0, stop_on="height")
    t = np.linspace(0.0, r["t_height_s"], n)
    Y = r["sol"].sol(t)
    return p, t, Y


# ------------------------------------------------------------------ Fig. 1
def arrow(ax, x0, y0, x1, y1, color="k", lw=1.3, ms=10):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def fig_geometry():
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.0))
    for a in ax:
        a.set_aspect("equal")
        a.axis("off")

    # (a) vertical plane containing the velocity
    a = ax[0]
    a.set_xlim(-1.6, 2.6)
    a.set_ylim(-1.7, 1.9)
    g = np.radians(30)
    a.plot([-1.4, 2.4], [0, 0], color="0.6", lw=0.8, ls="--")
    a.text(2.42, -0.03, "horizontal", fontsize=8, color="0.4", ha="right", va="top")
    a.add_patch(Circle((0, 0), 0.22, fc="0.92", ec="k", lw=1.0))
    arrow(a, 0, 0, 1.9 * np.cos(g), 1.9 * np.sin(g))
    a.text(1.9 * np.cos(g) + 0.05, 1.9 * np.sin(g) + 0.05, r"$\mathbf{v}$", fontsize=10)
    arrow(a, 0, 0, 1.9 * np.cos(g), 0, color="0.35", lw=1.0)
    a.text(1.0, -0.2, r"$v_h$", fontsize=9, color="0.35")
    arrow(a, 1.9 * np.cos(g), 0, 1.9 * np.cos(g), 1.9 * np.sin(g), color="0.35", lw=1.0)
    a.text(1.9 * np.cos(g) + 0.07, 0.4, r"$v_z$", fontsize=9, color="0.35")
    a.add_patch(Arc((0, 0), 1.1, 1.1, theta1=0, theta2=30, lw=0.9))
    a.text(0.62, 0.12, r"$\gamma$", fontsize=10)
    arrow(a, 0, 0, -0.95 * np.cos(g), -0.95 * np.sin(g), color="C3")
    a.text(-1.55, -0.72, r"drag $\mathbf{F}_D$", fontsize=8.5, color="C3")
    arrow(a, 0, 0, 0, -1.35, color="C0")
    a.text(0.08, -1.45, r"$m\mathbf{g}$", fontsize=9, color="C0")
    arrow(a, -0.9, 0.55, -0.9, 1.45, color="C2")
    a.text(-1.32, 1.55, r"spin axis $\hat{\boldsymbol{\omega}}=\hat{\mathbf{z}}$",
           fontsize=8.5, color="C2")
    a.text(-1.5, 1.85, "(a) side view", fontsize=9, va="top", fontweight="bold")

    # (b) horizontal plane, viewed from above
    a = ax[1]
    a.set_xlim(-1.7, 2.5)
    a.set_ylim(-1.7, 1.9)
    a.plot([0, 2.2], [0, 0], color="0.6", lw=0.8, ls="--")
    a.text(2.25, 0.0, r"$x$", fontsize=9, color="0.4", va="center")
    psi = np.radians(35)
    # curved ground track bending to the left
    s = np.linspace(-1.2, 1.4, 100)
    Rtr = 2.4
    cx, cy = -Rtr * np.sin(psi), Rtr * np.cos(psi)
    ang = np.arctan2(-cy, -cx) + s / Rtr
    a.plot(cx + Rtr * np.cos(ang), cy + Rtr * np.sin(ang), color="0.5", lw=1.0)
    a.text(-1.62, -0.98, "ground track", fontsize=8, color="0.4")
    a.add_patch(Circle((0, 0), 0.22, fc="0.92", ec="k", lw=1.0))
    a.plot(0, 0, "o", color="C2", ms=3)
    a.add_patch(Circle((0, 0), 0.07, fc="none", ec="C2", lw=1.0))
    a.text(0.2, -0.38, r"$\hat{\boldsymbol{\omega}}$ (out of page)", fontsize=8, color="C2")
    arrow(a, 0, 0, 1.7 * np.cos(psi), 1.7 * np.sin(psi))
    a.text(1.7 * np.cos(psi) + 0.04, 1.7 * np.sin(psi) + 0.04, r"$\mathbf{v}_h$", fontsize=10)
    a.add_patch(Arc((0, 0), 1.2, 1.2, theta1=0, theta2=35, lw=0.9))
    a.text(0.68, 0.15, r"$\psi$", fontsize=10)
    arrow(a, 0, 0, -1.15 * np.sin(psi), 1.15 * np.cos(psi), color="C1")
    a.text(-1.55, 1.12, r"Magnus $\mathbf{F}_M$", fontsize=8.5, color="C1")
    arrow(a, 0, 0, -0.75 * np.cos(psi), -0.75 * np.sin(psi), color="C3")
    a.text(-1.25, -0.72, r"$\mathbf{F}_{D,h}$", fontsize=9, color="C3")
    a.text(-1.6, 1.85, "(b) top view", fontsize=9, va="top", fontweight="bold")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.05)
    save(fig, "geometry")


# ------------------------------------------------------------------ Fig. 2
KICKS = [(20.0, 15.0), (28.0, 20.0), (25.0, 30.0), (30.0, 45.0)]
STYLES = [("k", "-"), ("C0", "--"), ("C3", "-."), ("C2", ":")]


def fig_same_turn():
    fig, ax = plt.subplots(2, 2, figsize=(W, 5.0))
    marks = np.arange(10.0, 70.0, 10.0)
    print("\nFig. 2: soccer ball, L_L = %.2f m, mu = %.3f" %
          (make_ball("soccer").lift_length_m, 0.25 / 0.25))
    for (v0, td), (c, ls) in zip(KICKS, STYLES):
        p, t, Y = flight("soccer", v0, td)
        LL, mu = p.lift_length_m, p.cd_const / p.cl_const
        lab = rf"$v_0={v0:.0f}$ m/s, $\theta={td:.0f}^\circ$"
        ax[0, 0].plot(Y[0], Y[1], color=c, ls=ls, label=lab)
        # heading ticks at equal path lengths
        for sm in marks[marks < Y[10, -1]]:
            k = np.searchsorted(Y[10], sm)
            ax[0, 0].plot(Y[0, k], Y[1, k], "o", color=c, ms=3)
        ax[0, 1].plot(t, np.degrees(Y[9]), color=c, ls=ls)
        ax[1, 0].plot(Y[10], np.degrees(Y[9]), color=c, ls=ls, lw=2.2 if c == "k" else 1.2)
        vh = np.hypot(Y[3], Y[4])
        sp = np.linalg.norm(Y[3:6], axis=0)
        ax[1, 1].plot(np.degrees(Y[9]), vh / vh[0], color=c, ls=ls)
        ax[1, 1].plot(np.degrees(Y[9]), sp / sp[0], color=c, ls=ls, lw=0.8, alpha=0.6)
        law = np.exp(-mu * Y[9])
        relh = np.max(np.abs(vh / vh[0] - law) / law)
        rel_rate = np.max(np.abs(np.gradient(Y[9], Y[10]) * LL - 1.0)[5:-5])
        print(f"  v0={v0:.0f} th={td:.0f}: T={t[-1]:.2f} s, S={Y[10,-1]:.1f} m, "
              f"turn={np.degrees(Y[9,-1]):.1f} deg, |v_h| ratio={vh[-1]/vh[0]:.4f} "
              f"(law {law[-1]:.4f}, max rel dev {relh:.1e}), |v| ratio={sp[-1]/sp[0]:.3f}, "
              f"range={np.hypot(Y[0,-1],Y[1,-1]):.1f} m")
    psi = np.linspace(0, 50, 50)
    ax[1, 1].plot(psi, np.exp(-mu * np.radians(psi)), color="0.5", lw=3.0, alpha=0.35,
                  zorder=0, label=r"$e^{-\mu\psi}$")
    ax[0, 0].set(xlabel=r"$x$ (m)", ylabel=r"$y$ (m)", aspect="equal")
    ax[0, 0].legend(loc="upper left", fontsize=7, handlelength=2.2, labelspacing=0.2)
    ax[0, 1].set(xlabel=r"time $t$ (s)", ylabel=r"heading $\psi$ (deg)")
    ax[1, 0].set(xlabel=r"path length $s$ (m)", ylabel=r"heading $\psi$ (deg)")
    ax[1, 1].set(xlabel=r"heading $\psi$ (deg)", ylabel="speed / launch value", ylim=(0.35, 1.02))
    ax[1, 1].text(14, 0.47, r"total $|\mathbf{v}|$ (thin)", fontsize=8, color="0.3")
    ax[1, 1].text(33, 0.62, r"horizontal $v_h$", fontsize=8)
    ax[1, 1].legend(loc="lower left", fontsize=8)
    for a, lab in zip(ax.ravel(), "abcd"):
        a.text(0.97, 0.05 if lab != "a" else 0.05, f"({lab})", transform=a.transAxes,
               ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    save(fig, "same_turn")


# ------------------------------------------------------------------ Fig. 3
def fig_horizontal():
    v0, td = 35.0, 60.0
    p, t, Y = flight("pingpong", v0, td, n=3000)
    LL, mu = p.lift_length_m, p.cd_const / p.cl_const
    # same launch with gravity switched off, followed for the same path length
    p0 = make_ball("pingpong", gravity_on=False)
    th = np.radians(td)
    y0 = initial_state(v0 * np.cos(th), 0.0, SPIN_RAD_PER_S)  # same horizontal launch speed
    r0 = fly(y0, p0, t_max_s=60.0, stop_on=None, psi_target_rad=np.pi)
    tt = np.linspace(0, r0["sol"].t[-1], 6000)
    Y0 = r0["sol"].sol(tt)
    keep = Y0[9] <= 2 * np.pi
    Y0 = Y0[:, keep]

    fig, ax = plt.subplots(1, 3, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1.45, 1, 1.15]))
    vh0 = np.hypot(Y[3, 0], Y[4, 0])
    ax[0].plot(Y[3], Y[4], "k-", label="simulated")
    ps = np.linspace(0, Y[9, -1], 200)
    ax[0].plot(vh0 * np.exp(-mu * ps) * np.cos(ps), vh0 * np.exp(-mu * ps) * np.sin(ps),
               color="C1", lw=3, alpha=0.35, label="log spiral")
    ax[0].plot(0, 0, "k+", ms=5)
    ax[0].set(xlabel=r"$v_x$ (m/s)", ylabel=r"$v_y$ (m/s)", aspect="equal")
    ax[0].legend(loc="upper right", fontsize=7, handlelength=1.5)

    ax[1].plot(Y[0], Y[1], "k-", label=r"with gravity")
    ax[1].plot(Y0[0], Y0[1], "C0--", lw=1.0, label=r"$g=0$")
    ax[1].plot(0, 0, "ko", ms=3)
    ax[1].set(xlabel=r"$x$ (m)", ylabel=r"$y$ (m)", aspect="equal")
    ax[1].legend(loc="center left", fontsize=7, handlelength=1.8)

    d = rhs(0.0, Y, p)
    vh = np.hypot(Y[3], Y[4])
    Rh = vh / np.abs(d[9])
    gam = np.arctan2(Y[5], vh)
    ax[2].plot(Y[10], Rh / LL, "k-", label=r"$R_h/L_L$")
    ax[2].plot(Y[10], np.cos(gam), color="C1", lw=3, alpha=0.35, label=r"$\cos\gamma$")
    ax[2].set(xlabel=r"path length $s$ (m)", ylabel=r"$R_h/L_L$", ylim=(0, 1.05))
    ax[2].legend(loc="lower left", fontsize=7)
    for a, lab in zip(ax, "abc"):
        a.set_title(f"({lab})", fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, "horizontal")

    xc, yc, Rf, rms = sweep.fit_circle(Y0[0], Y0[1])
    print(f"\nFig. 3: table-tennis ball, L_L = {LL:.3f} m, mu = {mu:.3f}, v0 = {v0}, theta = {td}")
    print(f"  flight T={t[-1]:.2f} s, S={Y[10,-1]:.2f} m, turn={np.degrees(Y[9,-1]):.1f} deg, "
          f"v_h ratio {vh[-1]/vh[0]:.4f} vs law {np.exp(-mu*Y[9,-1]):.4f}")
    print(f"  R_h/L_L: max dev from cos(gamma) {np.max(np.abs(Rh/LL-np.cos(gam))):.1e}; "
          f"min R_h/L_L {np.min(Rh/LL):.3f}")
    sp0 = np.linalg.norm(Y0[3:6], axis=0)
    print(f"  g=0: circle fit R={Rf:.6f} vs L_L={LL:.6f}, rms {rms:.1e} m, "
          f"speed {sp0[0]:.2f} -> {sp0[-1]:.4f} m/s over turn {np.degrees(Y0[9,-1]):.0f} deg")


# ------------------------------------------------------------------ Fig. 5 and S2
def fig_deficit_two():
    mu_grid = np.logspace(-2, np.log10(3.0), 60)
    th_grid = np.linspace(2.0, 88.0, 60)
    D, _ = deficit_grid_deg(np.radians(th_grid), mu_grid, n_bisect=70)
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7))
    im = ax[0].pcolormesh(th_grid, mu_grid, D, cmap="magma", shading="nearest",
                          norm=matplotlib.colors.LogNorm(vmin=max(D.min(), 1e-4), vmax=D.max()),
                          rasterized=True, edgecolors="face", linewidth=0)
    ax[0].minorticks_off()
    fig.colorbar(im, ax=ax[0], label=r"$90^\circ-\beta$ (deg)")
    for nm in BALLS:
        if nm == "frisbee":
            continue
        p = make_ball(nm)
        ax[0].plot(45.0, p.cd_const / p.cl_const, "w+", ms=7, mew=1.4)
    ax[0].set(yscale="log", xlabel=r"launch angle $\theta$ (deg)", ylabel=r"$\mu=C_D/C_L$")
    C = 8.0 / 3.0 - 24.0 / np.pi ** 2
    mus = np.logspace(-3, -0.5, 12)
    th = np.radians(0.5)
    num = [np.radians(deficit_deg(th, m)) / (m * th ** 2) for m in mus]
    ax[1].semilogx(mus, num, "ko", ms=3.5, label="reduced equation")
    ax[1].axhline(C, color="C3", ls="--", label=r"$8/3-24/\pi^2$")
    ax[1].set(xlabel=r"$\mu=C_D/C_L$", ylabel=r"$(90^\circ-\beta)/(\mu\theta^2)$")
    ax[1].legend(fontsize=8)
    for a, lab in zip(ax, "ab"):
        a.text(0.04, 0.95, f"({lab})", transform=a.transAxes, va="top", fontsize=9,
               color="w" if lab == "a" else "k")
    fig.tight_layout()
    save(fig, "deficit")
    print(f"\nFig. 5: min deficit {D.min():.3e} deg, max {D.max():.2f} deg; "
          f"coefficient {C:.7f}, smallest-mu ratio {num[0]:.7f} (rel {abs(num[0]/C-1):.1e})")

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for mu, tdd, c, ls in ((1.0, 45.0, "k", "-"), (0.5, 45.0, "C0", "--"),
                           (1.5, 30.0, "C3", ":")):
        thh = np.radians(tdd)
        u, _, gam = solve_Q(thh, solve_lambda(thh, mu), mu, n=4001)
        cg = np.cos(gam)
        half = (len(u) - 1) // 2 + 1
        ax.plot(u[:half], cg[:half] - cg[::-1][:half], color=c, ls=ls,
                label=rf"$\mu={mu}$, $\theta={tdd:.0f}^\circ$")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set(xlabel=r"$\psi$ (rad)", ylabel=r"$\cos\gamma(\psi)-\cos\gamma(\pi-\psi)$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "folded")


# ------------------------------------------------------------------ worked numbers
def worked_examples():
    print("\nWorked examples (Sec. IV)")
    for nm in ("baseball", "soccer", "tennis", "golf", "pingpong"):
        p = make_ball(nm)
        print(f"  {nm:9s} m={p.mass_kg} r={p.radius_m} C_L={p.cl_const} C_D={p.cd_const} "
              f"L_L={p.lift_length_m:.2f} m  mu={p.cd_const/p.cl_const:.3f}  "
              f"90deg costs {np.pi/2*p.lift_length_m:.1f} m, 180deg {np.pi*p.lift_length_m:.1f} m")
    # pitched baseball, nearly flat, 16.8 m of travel
    for nm, v0, td, dist in (("baseball", 40.0, 1.0, 16.8), ("soccer", 25.0, 15.0, 25.0)):
        p = make_ball(nm)
        r = fly(initial_state(v0, np.radians(td), SPIN_RAD_PER_S), p, t_max_s=10, stop_on=None)
        tt = np.linspace(0, 3.0, 20000)
        Y = r["sol"].sol(tt)
        k = np.argmax(Y[0] >= dist)
        print(f"  {nm} v0={v0} th={td}: at x={dist} m: y={Y[1,k]:.3f} m, s={Y[10,k]:.2f} m, "
              f"psi={np.degrees(Y[9,k]):.2f} deg; estimate x^2/(2L_L)={dist**2/(2*p.lift_length_m):.3f} m, "
              f"t={tt[k]:.3f} s, z={Y[2,k]:.2f} m")


if __name__ == "__main__":
    fig_geometry()
    fig_same_turn()
    fig_horizontal()
    worked_examples()
    ideal = make_ball("baseball", drag_on=False)
    theta = np.radians(45.0)
    v0 = solve_v0(theta, ideal)
    sweep.FIGSIZE.update({"invariants": (6.8, 2.9), "ground_track": (6.8, 3.0),
                          "trajectory_pair": (3.8, 3.2)})
    sh = sweep.fig_shape_and_3d(v0, theta, ideal)  # writes trajectory_pair.pdf
    os.remove(f"{OUT}/ground_track.pdf")
    print(f"\nFig. 4: theta=45, v0={v0:.4f} m/s, R_max/R_min={sh['R_ratio']:.6f} "
          f"sec={sh['sec_theta']:.6f}, miss {sh['pair']['miss_m']:.2e} m")
    sweep.fig_invariants(ideal)  # Fig. S1
    fig_deficit_two()

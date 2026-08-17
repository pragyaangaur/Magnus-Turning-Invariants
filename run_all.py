"""Driver: reproduces every number in the README and writes figures/*.pdf."""
import json
import numpy as np

from src.model import (make_ball, initial_state, v0_closed_form, BALLS, rhs,
                       ballistic_arc_length_m)
from src.integrate import fly, DivergedError
from src.solve import solve_v0, residual_1d, closure_report, solve_2d, SPIN_RAD_PER_S
from src import sweep

np.set_printoptions(precision=6, suppress=True)
OUT = {}


def sec(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ---------------- Invariants
sec("INVARIANTS  (drag OFF, const C_L, no spin decay)")
ideal = make_ball("baseball", drag_on=False)
kL = ideal.k_lift_per_m
print(f"k_L = {kL:.12e} 1/m   L_L = {ideal.lift_length_m:.6f} m   pi*L_L = {np.pi*ideal.lift_length_m:.6f} m")

worst_a = 0.0
for v0 in (20.0, 60.0, 110.0):
    for td in (10.0, 45.0, 80.0):
        r = fly(initial_state(v0, np.radians(td), SPIN_RAD_PER_S), ideal, t_max_s=9000.0, stop_on="psi")
        d = rhs(0.0, r["sol"].sol(np.linspace(0, r["t_psi_s"], 2000)), ideal)
        worst_a = max(worst_a, np.abs(d[9] / d[10] / kL - 1.0).max())
print(f"(a) max relative deviation of dpsi/ds from k_L : {worst_a:.3e}")

worst_b = 0.0
for v0 in np.linspace(10, 120, 12):
    for td in np.linspace(5, 85, 9):
        r = fly(initial_state(v0, np.radians(td), SPIN_RAD_PER_S), ideal, t_max_s=20000.0, stop_on="psi")
        worst_b = max(worst_b, abs(r["y_psi"][10] / (np.pi * ideal.lift_length_m) - 1.0))
print(f"(b) max relative deviation of S_tot from pi*L_L over v0x theta sweep : {worst_b:.3e}")

worst_c = 0.0
for v0 in np.linspace(10, 120, 12):
    for td in np.linspace(5, 85, 9):
        th = np.radians(td)
        r = fly(initial_state(v0, th, SPIN_RAD_PER_S), ideal, t_max_s=2000.0, stop_on="height")
        worst_c = max(worst_c, abs(r["y_height"][10] / ballistic_arc_length_m(v0, th, ideal.g_m_per_s2) - 1.0))
print(f"(c) max relative deviation of ballistic arc length from closed form : {worst_c:.3e}")
OUT["invariants"] = dict(a=worst_a, b=worst_b, c=worst_c, kL=kL, L_L=ideal.lift_length_m)
sweep.fig_invariants(ideal)

# drag-on bonus check
p_bb = make_ball("baseball")
r = fly(initial_state(60.0, np.radians(40.0), SPIN_RAD_PER_S), p_bb, t_max_s=9000.0, stop_on="psi")
d = rhs(0.0, r["sol"].sol(np.linspace(0, r["t_psi_s"], 2000)), p_bb)
extra = np.abs(d[9] / d[10] / p_bb.k_lift_per_m - 1.0).max()
print(f"    bonus: with DRAG ON, dpsi/ds still equals k_L to {extra:.3e}")
OUT["invariants"]["a_with_drag"] = extra


# ---------------- Closure Of A 180 Degree Turn
sec("CLOSURE OF A 180 DEGREE TURN")
print("ideal (drag off): theta is free, 1D solve for v0")
tbl = []
for td in [15, 30, 45, 60, 75, 85]:
    th = np.radians(td)
    v0 = solve_v0(th, ideal)
    cf = float(v0_closed_form(th, ideal))
    tbl.append((td, v0, cf, abs(v0 / cf - 1)))
    print(f"   theta={td:2d}deg  v0_num={v0:9.5f}  v0_closedform={cf:9.5f}  rel diff={abs(v0/cf-1):.2e}")
OUT["closure_ideal"] = tbl

print("\n2D system on (v0, theta) -- Jacobian singular values at the converged point:")
sol2, J, sv = solve_2d(tbl[2][1], np.radians(45.0), ideal)
print(f"   residual at solution = {sol2.fun},  x = {sol2.x}")
print(f"   J =\n{J}")
print(f"   singular values = {sv},  ratio = {sv.min()/sv.max():.3e}")
print("   -> rank 1: F1 and F2 encode the SAME scalar condition, so the zero set is a curve.")
OUT["closure_svd"] = dict(sv=sv.tolist(), ratio=float(sv.min() / sv.max()))

print("\ndrag ON, baseball:")
try:
    solve_v0(np.radians(45.0), p_bb)
except DivergedError as e:
    print("   ", e)

vg = np.linspace(20, 150, 27)
tg = np.radians(np.linspace(10, 88, 20))
P = np.full((len(vg), len(tg)), np.nan)
for i, v0 in enumerate(vg):
    for j, th in enumerate(tg):
        try:
            P[i, j] = residual_1d(v0, th, p_bb)
        except DivergedError:
            pass
k = np.nanargmax(P)
i, j = np.unravel_index(k, P.shape)
print(f"   min |R| over 20<=v0<=150, 10<=theta<=88 : {abs(P[i,j]):.5f} rad "
      f"({abs(P[i,j])/np.pi:.4f} pi short) at v0={vg[i]:.2f} m/s, theta={np.degrees(tg[j]):.2f} deg")
print(f"   psi_max attained = {P[i,j]+np.pi:.5f} rad = {(P[i,j]+np.pi)/np.pi:.4f} pi")
print("   F1 = z(t_psi) - z_launch is UNDEFINED here: the psi=pi event never fires at all.")
OUT["closure_drag_baseball"] = dict(min_absR=float(abs(P[i, j])), v0=float(vg[i]),
                                  theta_deg=float(np.degrees(tg[j])),
                                  psi_max_over_pi=float((P[i, j] + np.pi) / np.pi))

# where does the baseball root actually live?
th85 = np.radians(85.0)
v0_root = solve_v0(th85, p_bb, v0_cap_m_per_s=5000.0)
print(f"   the root does exist, but at v0 = {v0_root:.1f} m/s (theta=85deg) -- Mach {v0_root/343:.1f}.")
OUT["closure_drag_baseball"]["root_v0"] = float(v0_root)


# ---------------- Mirror Symmetry And Throw Tolerance
sec("MIRROR SYMMETRY AND THROW TOLERANCE")
v0i, thi = solve_v0(np.radians(45.0), ideal), np.radians(45.0)
p_sat = make_ball("baseball", drag_on=False, cl_model="saturating")
base, rows = sweep.symmetry_table(v0i, thi, ideal, p_spin=p_sat)
P_end = base["yA"]
chord = float(np.hypot(P_end[0], P_end[1]))
print(f"exact mirror pair (ideal, v0={v0i:.4f} m/s, theta=45deg):")
print(f"   endpoint miss distance = {base['miss_m']:.3e} m   (event-location limited)")
print(f"   arrival time diff      = {base['dt_s']:.3e} s")
print(f"   meeting point          = ({P_end[0]:.6e}, {P_end[1]:.6f}, {P_end[2]:.3e})")
print(f"   chord |OP| = {chord:.6f} m, launch->meeting bearing = "
      f"{np.degrees(np.arctan2(P_end[1], P_end[0])):.6f} deg (launch bearing 0 deg)")
print(f"   |v_f| = {base['vfA_m_per_s']:.6f}, closing speed = {base['closing_speed_m_per_s']:.6f}, "
      f"ratio = {base['closing_speed_m_per_s']/base['vfA_m_per_s']:.12f}")
print("   the ratio is NOT 2: the vertical velocities are common mode (both balls")
print("   descending) and cancel in the difference, so the law is 2*cos(theta):")
for td in [10.0, 30.0, 45.0, 60.0, 80.0]:
    th = np.radians(td)
    b = sweep.mirror_pair(solve_v0(th, ideal, v0_cap_m_per_s=1000.0), th, ideal)
    rr = b["closing_speed_m_per_s"] / b["vfA_m_per_s"]
    print(f"      theta={td:4.1f}deg  ratio={rr:.12f}  2cos(theta)={2*np.cos(th):.12f}  "
          f"rel diff={abs(rr/(2*np.cos(th))-1):.2e}")
pe, ve, scale = sweep.mirror_state_error(base)
print(f"   state-level mirror test at common times: max |dr| = {pe:.3e} m, "
      f"max |dv| = {ve:.3e} m/s, over a path scale of {scale:.1f} m")
print("\nsymmetry breaking (perturbation applied to ball B only):")
per = []
for nm, d, _pp in rows:
    print(f"   {nm:32s} miss = {d['miss_m']:9.4f} m   dt = {d['dt_s']:.4e} s")
    per.append((nm, d["miss_m"], d["dt_s"]))
OUT["symmetry"] = dict(miss=base["miss_m"], dt=base["dt_s"], state_pos_err=pe, state_vel_err=ve,
                    closing_ratio=base["closing_speed_m_per_s"] / base["vfA_m_per_s"],
                    meeting=[float(x) for x in P_end[0:3]], perturbed=per,
                    v0=v0i, chord=chord, R=chord / 2)


# ---------------- Feasibility Map
sec("FEASIBILITY MAP")
cl_cd = np.logspace(np.log10(0.2), np.log10(20.0), 90)
loading = np.logspace(np.log10(1.0), np.log10(400.0), 90)
psi_max, vf_close, (vhat, G_of_vhat, best_th, VF_of_vhat) = sweep.feasibility_map(cl_cd, loading)
pts = sweep.ball_points()
sweep.fig_feasibility(psi_max, vf_close, cl_cd, loading, pts)
Rgrid, vf_sim = sweep.fig_vf_law(vhat, G_of_vhat, VF_of_vhat)

print(f"G = S/L_D is monotone in v0/vt; G(max vhat={vhat[-1]:.0f}) = {G_of_vhat[-1]:.4f}")
print("boundary C_L/C_D = pi/G, so it sits at pi only where the path is exactly one drag length:")
for gv in [0.5, 1.0, 2.0, 4.0, 8.0]:
    vv = np.interp(gv, G_of_vhat, vhat)
    print(f"   G={gv:4.1f} -> C_L/C_D boundary = {np.pi/gv:6.3f}, needs v0/vt = {vv:6.2f}")
print("\nresidual TOTAL speed vs exp(-pi C_D/C_L), which is exact only for horizontal speed:")
for rr in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    pred = np.exp(-np.pi / rr)
    simv = float(np.interp(rr, Rgrid, vf_sim))
    print(f"   C_L/C_D={rr:5.1f}   exp(-pi C_D/C_L) {pred:.4f}   simulated {simv:.4f}   ratio {simv/pred:.3f}")
OUT["feasibility"] = dict(G_max=float(G_of_vhat[-1]),
                    vf_law=[(float(rr), float(np.exp(-np.pi / rr)),
                             float(np.interp(rr, Rgrid, vf_sim))) for rr in [0.5, 1, 2, 3, 5, 10]])

print("\nper-ball closure (drag on, v0 capped at 150 m/s):")
ball_rows = []
for nm in BALLS:
    p = make_ball(nm)
    try:
        fam = closure_report(p, np.arange(20.0, 89.0, 4.0))
    except DivergedError:
        fam = []
    if not fam:
        # report how far short it falls
        best = -np.inf
        for v0 in np.linspace(20, 150, 20):
            for td in np.linspace(20, 88, 15):
                try:
                    best = max(best, residual_1d(v0, np.radians(td), p) + np.pi)
                except DivergedError:
                    pass
        print(f"   {nm:9s} INFEASIBLE  C_L/C_D={p.cl_const/p.cd_const:.3f}  psi_max={best/np.pi:.4f} pi")
        ball_rows.append(dict(ball=nm, feasible=False, cl_cd=p.cl_const / p.cd_const,
                              psi_max_over_pi=float(best / np.pi)))
    else:
        b = min(fam, key=lambda d: d["v0_m_per_s"])
        print(f"   {nm:9s} OK  theta={b['theta_deg']:.0f}deg v0={b['v0_m_per_s']:.3f} R={b['radius_m']:.3f} m "
              f"t={b['t_collision_s']:.4f} s vf={b['vf_m_per_s']:.3f} vf/v0={b['vf_over_v0']:.4f}")
        ball_rows.append(dict(ball=nm, feasible=True, cl_cd=p.cl_const / p.cd_const, **b))
OUT["balls"] = ball_rows


# ---------------- Ground-Track Shape
sec("GROUND-TRACK SHAPE")
sh = sweep.fig_shape_and_3d(v0i, thi, ideal)
print(f"ideal, theta=45deg: circle fit R={sh['R_fit_m']:.4f} m, RMS residual={sh['rms_circle_m']:.4f} m "
      f"({100*sh['rms_circle_m']/sh['R_fit_m']:.3f}% of R)")
print(f"   radius of curvature ratio max/min = {sh['R_ratio']:.6f}   sec(theta) = {sh['sec_theta']:.6f}")
OUT["shape"] = dict(R_fit=sh["R_fit_m"], rms=sh["rms_circle_m"],
                    ratio=sh["R_ratio"], sec_theta=sh["sec_theta"])
for td in [20.0, 45.0, 70.0]:
    th = np.radians(td)
    v0 = solve_v0(th, ideal)
    s = sweep.fig_shape_and_3d(v0, th, ideal)
    print(f"   theta={td:4.1f}deg  R_max/R_min={s['R_ratio']:.6f}  sec(theta)={s['sec_theta']:.6f}  "
          f"circle-fit RMS={100*s['rms_circle_m']/s['R_fit_m']:.3f}% of R")

with open("results.json", "w") as f:
    json.dump(OUT, f, indent=1, default=float)
print("\nwrote results.json and figures/*.pdf")

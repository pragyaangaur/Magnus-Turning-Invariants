"""Physical model: gravity + quadratic drag + Magnus, vertical spin axis.

State vector (11): [x, y, z, vx, vy, vz, wx, wy, wz, psi, s]
  psi = unwrapped azimuth heading of the horizontal velocity (integrated, never atan2'd)
  s   = path length
"""
from dataclasses import dataclass
import numpy as np

G_M_PER_S2 = 9.80665


@dataclass(frozen=True)
class Params:
    """Every physical constant, units in the field names."""
    mass_kg: float
    radius_m: float
    rho_air_kg_per_m3: float = 1.225
    nu_air_m2_per_s: float = 1.5e-5
    g_m_per_s2: float = G_M_PER_S2
    # lift
    cl_model: str = "const"          # "const" | "saturating"
    cl_const: float = 0.20
    # drag
    cd_model: str = "const"          # "const" | "crisis"
    cd_const: float = 0.47
    cd_sub: float = 0.47             # pre-crisis plateau
    cd_super: float = 0.15           # post-crisis plateau
    re_crit: float = 3.0e5           # centre of the tanh blend
    re_width: float = 5.0e4          # blend half-width
    # switches
    drag_on: bool = True
    magnus_on: bool = True
    gravity_on: bool = True
    spin_decay_on: bool = False
    cm_spin_decay: float = 1.0e-2
    name: str = "generic"

    @property
    def area_m2(self) -> float:
        return np.pi * self.radius_m ** 2

    @property
    def k_lift_per_m(self) -> float:
        """k_L = rho*C_L*A/(2m).  Only meaningful for cl_model='const'."""
        return self.rho_air_kg_per_m3 * self.cl_const * self.area_m2 / (2.0 * self.mass_kg)

    @property
    def k_drag_per_m(self) -> float:
        return self.rho_air_kg_per_m3 * self.cd_const * self.area_m2 / (2.0 * self.mass_kg)

    @property
    def lift_length_m(self) -> float:
        """L_L = 1/k_L, the natural turning length."""
        return 1.0 / self.k_lift_per_m

    @property
    def inertia_kg_m2(self) -> float:
        return 0.4 * self.mass_kg * self.radius_m ** 2


def c_lift(p: Params, spin_param_S):
    if p.cl_model == "const":
        return np.full_like(np.asarray(spin_param_S, dtype=float), p.cl_const)
    if p.cl_model == "saturating":
        # C_L = 1/(2 + 1/S); S=0 -> 0
        S = np.asarray(spin_param_S, dtype=float)
        out = np.zeros_like(S)
        nz = S > 0.0
        out[nz] = 1.0 / (2.0 + 1.0 / S[nz])
        return out
    raise ValueError(f"unknown cl_model {p.cl_model!r}")


def c_drag(p: Params, speed_m_per_s):
    if p.cd_model == "const":
        return np.full_like(np.asarray(speed_m_per_s, dtype=float), p.cd_const)
    if p.cd_model == "crisis":
        re = 2.0 * p.radius_m * np.asarray(speed_m_per_s, dtype=float) / p.nu_air_m2_per_s
        blend = 0.5 * (1.0 + np.tanh((re - p.re_crit) / p.re_width))
        return p.cd_sub + (p.cd_super - p.cd_sub) * blend
    raise ValueError(f"unknown cd_model {p.cd_model!r}")


def rhs(t, y, p: Params):
    """Batched right-hand side.  y has shape (11,) or (11, N)."""
    y = np.asarray(y, dtype=float)
    flat = y.ndim == 1
    Y = y.reshape(11, -1)
    v = Y[3:6]
    w = Y[6:9]

    speed = np.sqrt(np.einsum("ij,ij->j", v, v))
    spin = np.sqrt(np.einsum("ij,ij->j", w, w))

    a = np.zeros_like(v)
    if p.gravity_on:
        a[2] -= p.g_m_per_s2

    if p.drag_on:
        cd = c_drag(p, speed)
        kd = p.rho_air_kg_per_m3 * cd * p.area_m2 / (2.0 * p.mass_kg)
        a -= kd * speed * v

    if p.magnus_on:
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.where(speed > 0.0, p.radius_m * spin / np.where(speed > 0.0, speed, 1.0), 0.0)
        cl = c_lift(p, S)
        kl = p.rho_air_kg_per_m3 * cl * p.area_m2 / (2.0 * p.mass_kg)
        what = np.where(spin > 0.0, w / np.where(spin > 0.0, spin, 1.0), 0.0)
        a += kl * speed * np.cross(what, v, axis=0)

    dw = np.zeros_like(w)
    if p.spin_decay_on:
        coef = (p.cm_spin_decay * p.rho_air_kg_per_m3 * p.radius_m ** 5
                / p.inertia_kg_m2)
        dw = -coef * spin * w

    vh2 = v[0] ** 2 + v[1] ** 2
    dpsi = np.where(vh2 > 0.0, (v[0] * a[1] - v[1] * a[0]) / np.where(vh2 > 0.0, vh2, 1.0), 0.0)

    out = np.empty_like(Y)
    out[0:3] = v
    out[3:6] = a
    out[6:9] = dw
    out[9] = dpsi
    out[10] = speed
    return out.reshape(-1) if flat else out


def initial_state(v0_m_per_s, theta_rad, spin_rad_per_s, azimuth_rad=0.0, spin_sign=+1.0,
                  origin_m=(0.0, 0.0, 0.0)):
    """Launch state.  Heading `azimuth_rad` in the horizontal plane, spin about +/- z."""
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    y = np.zeros(11)
    y[0:3] = origin_m
    y[3:6] = [v0_m_per_s * ct * ca, v0_m_per_s * ct * sa, v0_m_per_s * st]
    y[8] = spin_sign * spin_rad_per_s
    y[9] = azimuth_rad
    y[10] = 0.0
    return y


def ballistic_arc_length_m(v0_m_per_s, theta_rad, g_m_per_s2=G_M_PER_S2):
    """Closed form arc length of a drag-free parabola, launch height back to launch height."""
    v0 = np.asarray(v0_m_per_s, dtype=float)
    th = np.asarray(theta_rad, dtype=float)
    st, ct = np.sin(th), np.cos(th)
    return (v0 ** 2 / g_m_per_s2) * (st + ct ** 2 * np.arctanh(st))


def shape_factor(theta_rad):
    """f(theta) = sin(theta) + cos^2(theta)*ln(tan+sec) = sin + cos^2 * artanh(sin)."""
    th = np.asarray(theta_rad, dtype=float)
    return np.sin(th) + np.cos(th) ** 2 * np.arctanh(np.sin(th))


def v0_closed_form(theta_rad, p: Params):
    """v0 = sqrt(pi*g*L_L/f(theta)) -- the drag-free closure prediction."""
    return np.sqrt(np.pi * p.g_m_per_s2 * p.lift_length_m / shape_factor(theta_rad))


BALLS = {
    "baseball":   dict(mass_kg=0.145,  radius_m=0.0366, cl_const=0.20, cd_const=0.35),
    "tennis":     dict(mass_kg=0.0577, radius_m=0.0335, cl_const=0.25, cd_const=0.55),
    "pingpong":   dict(mass_kg=0.0027, radius_m=0.020,  cl_const=0.30, cd_const=0.45),
    "golf":       dict(mass_kg=0.0459, radius_m=0.0213, cl_const=0.25, cd_const=0.25),
    "soccer":     dict(mass_kg=0.430,  radius_m=0.110,  cl_const=0.25, cd_const=0.25),
    "frisbee":    dict(mass_kg=0.175,  radius_m=0.137,  cl_const=0.60, cd_const=0.20),
}


def make_ball(name: str, **overrides) -> Params:
    kw = dict(BALLS[name])
    kw.update(overrides)
    return Params(name=name, **kw)

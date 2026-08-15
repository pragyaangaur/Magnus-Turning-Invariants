# Two Balls, Opposite Throws, Vertical-Axis Spin

A numerical and analytic study of a deceptively simple mechanics problem: a thrower at the origin releases two balls simultaneously in opposite horizontal directions, each spinning about a vertical axis (one about `+z`, one about `-z`). Under gravity, quadratic drag and the Magnus force, can the two balls curve around and collide head-on at the height they were released, each having turned exactly 180 degrees in azimuth?

The short answer is that the drag-free version of the problem has an exact and rather elegant solution, and the version with air resistance is impossible. Getting to that answer turned up an exactly integrable structure hiding inside a system that the literature generally treats as requiring perturbation theory.

Everything below is reproduced by `python run_all.py` and checked by 101 tests in `pytest`.

## Contents

1. [Explain it simply](#explain-it-simply)
2. [The model](#the-model)
3. [Results](#results)
4. [The analytic structure](#the-analytic-structure)
5. [Novelty check](#novelty-check)
6. [Running it](#running-it)
7. [Repository layout](#repository-layout)

## Explain it simply

Spin a ball like a merry-go-round, with its axis pointing straight up, and throw it. It curves sideways. That is the Magnus effect, the same physics that makes a curveball curve or a free kick bend around a wall.

Because the spin axis points straight up, the sideways push is always horizontal, and it is always at right angles to the direction the ball is travelling horizontally. That is precisely what a steering wheel does. The key discovery is that this steering wheel is locked at a fixed angle. The ball turns a fixed number of degrees for every metre it travels. Not per second, per metre.

That single fact drives everything. To turn 180 degrees, the ball must cover a fixed distance, full stop. Throw it hard, throw it gently, high or low, it makes no difference. For a baseball that distance is 884 metres. That is the result `S_tot = pi * L_L`, and it holds to fifteen decimal places.

Where does air resistance fit in? Drag is a brake, not a steering input. It pushes straight backwards along the direction of travel, so it cannot bend the path at all. It only changes how quickly the ball covers ground. The clean demonstration: switch gravity off and the ball flies a perfect circle even while drag strips away 94 percent of its speed. The circle's radius does not budge in the tenth digit.

So why is the real ground track not a circle? Gravity, and gravity alone. The exact rule is `R = L_L * cos(gamma)`, where gamma is how steeply the ball is climbing or diving. Steep climb means a tight turn; flat at the top of the arc means the widest turn. That is why the track is an oval rather than a circle, and why the ratio of widest to tightest turn is exactly `sec(theta)`.

Why can a baseball not do this? It needs 884 metres of travel to turn around, but drag kills it after roughly 200 metres. It manages about half a turn. You would need to throw at Mach 1.8.

And the deepest reason the two-ball collision fails with air resistance: a ball with drag always comes down more steeply than it went up, because its horizontal speed decays without limit while its falling speed is capped at terminal velocity. That asymmetry tilts the finish line. The two balls sail past each other.

## The model

State vector `y = [r(3), v(3), omega(3), psi, s]`, eleven components. The accelerations are

```
a_grav   = -g * z_hat
a_drag   = -(rho * C_D * A / 2m) * |v| * v
a_magnus = +(rho * C_L * A / 2m) * |v| * (omega_hat x v)
```

Two lift models are switchable: constant `C_L`, and the saturating form `C_L = 1/(2 + 1/S)` with spin parameter `S = r*|omega|/|v|`. Two drag models: constant, and a drag-crisis model dropping `C_D` from 0.47 to 0.15 across `Re` of 2e5 to 4e5 with a tanh blend. Optional spin decay `d(omega)/dt = -C_M * rho * r^5 * |omega| * omega / I`. Every one of these is an independent flag on the `Params` dataclass, so effects can be isolated one at a time.

Two constants carry the physics. The **lift length** `L_L = 2m / (rho * C_L * A)` is the natural turning radius, and the **drag length** `1/k_D = 2m / (rho * C_D * A)` is the distance over which drag noticeably bleeds speed. Their ratio is just `C_L/C_D`.

The azimuth `psi` is carried as a state variable and integrated directly from `d(psi) = (vx*dvy - vy*dvx)/(vx^2 + vy^2)`, so it unwraps correctly past plus or minus pi with no `atan2` differencing anywhere. Path length `s` is likewise a state variable. Integration is `scipy.solve_ivp` with RK45 at `rtol=1e-10` and proper event detection, never fixed steps, for the two closure events. A batched fixed-step RK4 handles the vectorised parameter sweeps.

## Results

### The three analytic claims all survive

With drag off, constant `C_L`, no spin decay:

| claim | max relative deviation | verdict |
|---|---|---|
| `d(psi)/ds = k_L`, constant along the path | 4.44e-16 | holds to machine precision |
| `S_tot = pi/k_L`, independent of `v0` and `theta` (12 by 9 sweep) | 1.11e-15 | holds to machine precision |
| ballistic arc length equals the closed form | 3.37e-10 | holds, at the integrator's tolerance floor |

For a baseball at `C_L = 0.20`: `k_L = 3.555331739605e-03` per metre, `L_L = 281.2677 m`, `pi*L_L = 883.6286 m`.

Two of these are stronger than claimed: **they also hold with drag switched on**. Drag is antiparallel to `v`, so its contribution to `vx*ay - vy*ax` is identically zero and it cannot torque the azimuth. Measured with drag on, `d(psi)/ds = k_L` to 6.66e-16. The invariant `S_tot = pi*L_L` is therefore not a drag-free curiosity, it is exact for any constant-`C_L` model with a vertical spin axis. That is what makes the whole closure problem tractable, because the azimuth condition collapses to a pure path-length condition.

Two further exact consequences explain why the third claim survives even with Magnus active. Magnus with a vertical spin axis has no z-component, so `v_z` is purely ballistic and the hang time is exactly `2*v0*sin(theta)/g`, verified to 1e-9 s. And Magnus does no work while rotating the horizontal velocity without changing its magnitude, so `|v_h| = v0*cos(theta)` is conserved, verified to 1e-9. The speed profile is therefore identical to the drag-free parabola's.

### Closure, drag-free

The closed form `v0 = sqrt(pi*g*L_L / f(theta))` with `f(theta) = sin(theta) + cos^2(theta)*ln(tan(theta) + sec(theta))` is exact:

| theta | `v0` numerical | `v0` closed form | rel. diff |
|---|---|---|---|
| 15 deg | 130.87424 | 130.87424 | 8.1e-12 |
| 30 deg | 97.47711 | 97.47711 | 7.6e-12 |
| 45 deg | 86.88870 | 86.88870 | 6.7e-12 |
| 60 deg | 85.14578 | 85.14578 | 5.2e-12 |
| 75 deg | 88.68577 | 88.68577 | 2.6e-12 |
| 85 deg | 92.17206 | 92.17206 | 2.2e-14 |

### The 2D system as originally posed is degenerate

The two residuals `F1 = z(t*) - z_launch` and `F2 = psi(t*) - pi` are two ways of writing one scalar condition, namely that the two events coincide, over three unknowns `(v0, theta, t*)`. Running `scipy.optimize.root` on the 2 by 2 system and taking the finite-difference Jacobian at the converged point:

```
J = [[ 14.382026, 290.054597],
     [  0.072313,   1.458382]]
singular values = [2.904e+02, 1.172e-06],   sigma_min/sigma_max = 4.04e-09
```

Numerically rank 1. One row is a clean multiple of the other, `J[1,:] = J[0,:] / 198.9`. A 2D Newton solve on this system is solving a singular problem. The correct formulation is 1D in `v0` at fixed `theta`. **However**, see the impossibility result below: the physical problem really does become 2D once drag is present, just through a different second equation than the one originally proposed.

### A baseball cannot do it

With `C_D = 0.35` and `C_L = 0.20`, scanning `20 <= v0 <= 150 m/s` and `10 <= theta <= 88 deg`, the best the baseball manages is `psi_max = 1.582 rad = 0.5037*pi`. It falls **0.4963 pi short**, at the corner of the domain. Note that `F1` is undefined over this entire domain, because the `psi = pi` event never fires at all, so there is no `t*` at which to evaluate it. Reporting a combined `min(|F1| + |F2|)` would be misleading; the honest number is the `F2` deficit.

A root does exist mathematically, since achievable path length grows like `ln(v0)` without bound, but at `theta = 85 deg` it sits at `v0 = 616.3 m/s`, which is Mach 1.8 and well outside the validity of an incompressible quadratic-drag model.

### Symmetry, drag-free

Ball A at bearing 0 with `+z` spin, ball B at bearing 180 with `-z` spin, at `theta = 45 deg` and `v0 = 86.8887 m/s`:

| quantity | value |
|---|---|
| arrival time difference | 3.55e-15 s |
| state-level mirror error, max abs position | 5.19e-09 m over a 515.4 m path, 1.0e-11 relative |
| meeting point | (-8.1e-09, 515.408148, 9.8e-15) m |
| chord bearing from launch | 90.000000 deg |

On the requested 1e-9 m endpoint criterion: the raw endpoint miss is 1.56e-08 m and it **does not improve** when `rtol` is tightened from 1e-10 to 1e-13. It is floored by scipy's event root-location on the dense output, not by the integration. In relative terms it is 3e-11 of a 515 m path, at the double-precision limit for a trajectory this long. The state-level test compares `B(t)` against the analytically reflected `A(t)` at common times, sidesteps event location entirely, and is the sharp confirmation that the symmetry is exact.

One note on the geometry: the mirror plane is the vertical plane containing the launch-to-meeting **chord**, not the plane containing the launch direction. The latter leaves ball A's velocity unchanged and is not the relevant symmetry. The chord turns out perpendicular to the launch direction, so for a 180 degree turn the two launches are indeed anti-parallel, exactly as the problem requires.

### Closing speed is `2*cos(theta)*|v_f|`, not `2*|v_f|`

Measured ratio at 45 degrees is 1.414213562352, which is `sqrt(2)`, not 2.

The collision is head-on only in the horizontal plane. At the meeting point both balls are descending, so their vertical velocity components are common mode and cancel in the difference. Only the horizontal parts oppose, giving `|v_A - v_B| = 2*|v_h| = 2*v0*cos(theta) = 2*cos(theta)*|v_f|`.

| theta | measured ratio | `2*cos(theta)` | rel. diff |
|---|---|---|---|
| 10 deg | 1.969615506023 | 1.969615506024 | 4.7e-13 |
| 30 deg | 1.732050807558 | 1.732050807569 | 6.3e-12 |
| 45 deg | 1.414213562352 | 1.414213562373 | 1.5e-11 |
| 60 deg | 0.999999999969 | 1.000000000000 | 3.1e-11 |
| 80 deg | 0.347296355311 | 0.347296355334 | 6.5e-11 |

The value 2 is recovered only as theta goes to zero. At 60 degrees the closing speed equals a single ball's speed, and above 60 degrees the balls converge more slowly than either one is travelling.

### Throw tolerance

| perturbation on ball B | miss distance | arrival difference |
|---|---|---|
| +5 percent launch speed | 82.98 m | 6.27e-01 s |
| +2 degrees azimuth | 17.99 m | 1.78e-15 s |
| +10 percent spin, constant `C_L` | 0.0000 m | 3.55e-15 s |
| +10 percent spin, saturating `C_L` | 1083.64 m | 1.78e-15 s |

Against a 515 m chord and a 0.037 m ball, the throw is hopeless by any human standard. The azimuth number is pure geometry: rotating one trajectory by `d_alpha` moves its endpoint by `chord * d_alpha`, so the required aim precision is `r_ball/chord`, about 4.07e-03 degrees. Speed must be matched to roughly 1 part in 4.5e4.

Two remarks on the spin row. Under constant `C_L`, spin magnitude is **exactly inert**, because the Magnus term depends only on the spin direction `omega_hat`. Endpoints for 50, 200 and 800 rad/s agree to all twelve printed digits, so the 0.0000 m is correct physics rather than a bug, and spin tolerance is only a meaningful question once `C_L` depends on `S`. Under the saturating model the same mismatch gives a 1084 m miss, larger than the chord, making spin matching the tightest of the three tolerances rather than the loosest.

### Feasibility map

The sweep is vectorised through an exact reduction rather than a Python loop over `solve_ivp`. Because Magnus does no work and has no z-component, the `(|v|, v_z)` dynamics decouple completely from the azimuth, so the dimensionless path length `G = S/L_D` and the speed ratio `v_f/v_0` are **exactly independent of `C_L`**. One batched RK4 over a dimensionless `(v_0/v_t, theta)` grid with `k_D = g = 1` then determines the whole map analytically through `psi_max = (C_L/C_D) * G(v_0/v_t, theta)`, with closure whenever `psi_max >= pi`.

The predicted boundary at `C_L/C_D = pi` is not where it sits. The boundary is at `C_L/C_D = pi/G`, and `G` is a monotonically increasing, unbounded function of `v_0/v_t` that grows like `ln(v_0)`:

| `G` | boundary `C_L/C_D = pi/G` | requires `v_0/v_t` |
|---|---|---|
| 0.5 | 6.283 | 0.75 |
| 1.0 | 3.142, which is pi | 1.25 |
| 2.0 | 1.571 | 2.46 |
| 4.0 | 0.785 | 7.23 |
| 8.0 | 0.393 | 54.5 |

So `C_L/C_D = pi` is the special case `G = 1`, meaning a total path of exactly one drag length, which happens only when the launch speed is `1.25*v_t`. That is the implicit assumption in the original derivation. For any faster throw the boundary drops below pi, and throwing harder always helps, which is why the contour is a curve in the `(C_L/C_D, loading)` plane rather than a horizontal line. The loading axis enters only through `v_t = sqrt(2g/rho * m/(C_D*A))`.

Converged solutions, drag on, `v_0` capped at 150 m/s:

| ball | `C_L/C_D` | closes | theta | `v_0` (m/s) | `R` (m) | `t` (s) | `v_f` (m/s) | `v_f/v_0` |
|---|---|---|---|---|---|---|---|---|
| frisbee | 3.00 | yes | 64 deg | 19.847 | 5.268 | 2.8129 | 11.704 | 0.590 |
| pingpong | 0.667 | yes | 80 deg | 90.952 | 2.376 | 3.9662 | 8.698 | 0.096 |
| soccer | 1.000 | yes | 76 deg | 124.204 | 23.321 | 9.5827 | 26.196 | 0.211 |
| golf | 1.000 | no | | | | | | `psi_max` = 0.804 pi |
| tennis | 0.455 | no | | | | | | `psi_max` = 0.565 pi |
| baseball | 0.571 | no | | | | | | `psi_max` = 0.504 pi |

Golf and soccer share `C_L/C_D = 1.0` yet only soccer closes, because feasibility depends on `v_0/v_t` as well as the ratio, and the soccer ball's lower terminal velocity (26.9 versus 45.4 m/s) puts it at a higher `v_0/v_t` for the same capped `v_0`. That dependence is exactly what the `C_L/C_D = pi` prediction leaves out.

**Important caveat on this table.** These `(v_0, theta)` pairs satisfy "return to launch height with `psi = pi`" simultaneously, but with drag on they do **not** produce a collision between two oppositely-thrown balls. See the impossibility result below. Read the table as single-ball closure of a 180 degree turn, not as a verified collision.

For reference, the drag-free closure at 45 degrees:

| ball | `C_L` | `C_D` | `L_L` (m) | `v_0` (m/s) | `R` (m) | `t` (s) | `v_f` (m/s) |
|---|---|---|---|---|---|---|---|
| baseball | 0.20 | 0.35 | 281.27 | 86.889 | 257.704 | 12.530 | 86.889 |
| tennis | 0.25 | 0.55 | 106.88 | 53.561 | 97.925 | 7.724 | 53.561 |
| pingpong | 0.30 | 0.45 | 11.69 | 17.716 | 10.713 | 2.555 | 17.716 |
| golf | 0.25 | 0.25 | 210.31 | 75.133 | 192.690 | 10.835 | 75.133 |
| soccer | 0.25 | 0.25 | 73.87 | 44.529 | 67.684 | 6.422 | 44.529 |
| frisbee | 0.60 | 0.20 | 8.08 | 14.723 | 7.399 | 2.123 | 14.723 |

Here `v_f = v_0` exactly, since neither gravity nor Magnus changes the speed between two points at the same height.

### Shape

The ground track is an oval, tightest at launch and landing where the speed is greatest and flattest at the apex, exactly as predicted. The exact law is `R_h = L_L * cos(gamma)`, giving `R_max/R_min = sec(theta)`:

| theta | measured `R_max/R_min` | `sec(theta)` | circle-fit RMS residual |
|---|---|---|---|
| 20 deg | 1.064178 | 1.064178 | 0.105 percent of `R` |
| 45 deg | 1.414213 | 1.414214 | 0.600 percent of `R` |
| 70 deg | 2.923802 | 2.923804 | 1.670 percent of `R` |

At 45 degrees the best-fit circle has `R = 264.06 m` with an RMS residual of 1.585 m, which is 43 times the ball radius, so the departure from circularity is physically real rather than a fitting artifact. The fitted radius exceeds half the chord (257.70 m), the signature of an oval flattened at its midpoint.

## The analytic structure

This is where the problem turns out to be more interesting than it looks.

### The master identity

Write the horizontal velocity as a single complex number `w = v_x + i*v_y`. Gravity is vertical and contributes nothing. Drag is antiparallel to `v` and contributes `-k_D*|v|*w`. Magnus with a vertical spin axis gives `omega_hat x v = (-v_y, v_x, 0)`, which in complex notation is exactly `i*w`, a 90 degree rotation, and contributes `+i*k_L*|v|*w`. So

```
dw/dt = (i*k_L - k_D) * |v| * w
```

Now change the independent variable from time to arc length via `ds = |v| dt`. The speed cancels identically:

```
dw/ds = (i*k_L - k_D) * w        so        w(s) = w_0 * exp(-(k_D - i*k_L) * s)
```

A nine-dimensional nonlinear system has an exact closed-form solution for its horizontal part, and gravity never enters it. This is the engine behind every result in this section. It also reduces the whole problem to a single scalar ODE in arc length, since with `w(s)` known everything else follows from

```
dv_z/ds + k_D*v_z = -g / sqrt( |w_0|^2 * exp(-2*k_D*s) + v_z^2 )
```

The drag-free limit of that equation is separable and reproduces the classical parabola arc-length formula.

### The exponential formula was right, attached to the wrong quantity

Setting `Delta_psi = pi` in the master identity gives

```
|w_f| / |w_0| = exp(-pi * C_D / C_L)        exactly
```

for the **horizontal** speed, with gravity fully on, for any launch angle, any ball, any speed. Measured maximum relative error across four balls and three launch angles is below 1e-9, which is integrator-limited.

I initially scored `v_f/v_0 = exp(-pi*C_D/C_L)` as refuted, off by a factor of 23 near the feasibility boundary. That verdict is correct for the **total** speed, which is what the brief asked about, and the table stands:

| `C_L/C_D` | predicted | simulated total | ratio |
|---|---|---|---|
| 0.5 | 0.0019 | 0.0435 | 23.3 |
| 1.0 | 0.0432 | 0.2110 | 4.9 |
| 2.0 | 0.2079 | 0.4622 | 2.2 |
| 3.0 | 0.3509 | 0.5938 | 1.7 |
| 5.0 | 0.5335 | 0.7263 | 1.4 |
| 10.0 | 0.7304 | 0.8506 | 1.2 |

But the formula itself is not wrong. The entire discrepancy is `v_z`, which gravity re-accelerates and terminal velocity floors. Applied to horizontal speed the formula is exact.

A useful corollary: rearranged, `C_D/C_L = ln(|w_0|/|w_f|) / Delta_psi`. You can extract a real ball's lift-to-drag ratio from a single tracked trajectory by comparing how much its horizontal speed decayed against how far its heading swung. No force measurement, no wind tunnel, and gravity drops out of the answer.

### Curvature, and why gravity alone makes the oval

The ground-track radius of curvature is `R_h = (ds_h/dt)/(d_psi/dt) = |w|/(k_L*|v|)`, and `|w|/|v| = cos(gamma)` where gamma is the flight-path angle. So `R_h = L_L * cos(gamma)`, verified below 1e-12 relative with drag on or off. Two readings follow. The track is tightest where the ball climbs or dives hardest and flattest at the apex, and at the apex `gamma = 0` so `R_h = L_L` exactly. That makes `L_L` directly measurable: photograph the ground track, measure its radius of curvature at the top of the arc, and you have `2m/(rho*C_L*A)` without knowing `C_L`, `m` or `A` separately.

Since drag is antiparallel to `v` it cannot bend a path, only change the speed along it. With `g = 0` the ground track must therefore be a perfect circle of radius `L_L`, drag or no drag:

| ball | drag | fitted radius | `L_L` | rel. err | circle-fit RMS | speed change |
|---|---|---|---|---|---|---|
| baseball | off | 281.267705304 | 281.267705306 | 5.5e-12 | 3.4e-10 m | 50 to 50.000 m/s |
| baseball | on | 281.267705317 | 281.267705306 | 3.8e-11 | 3.6e-09 m | 50 to 3.200 m/s |
| frisbee | off | 8.075877046 | 8.075877046 | 5.4e-12 | 9.6e-12 m | 50 to 50.000 m/s |
| frisbee | on | 8.075877047 | 8.075877046 | 8.9e-11 | 7.3e-11 m | 50 to 29.619 m/s |

The baseball loses 94 percent of its speed and the circle radius does not move in the tenth digit. Gravity is the sole source of the oval.

### The chord integral, and an impossibility result

Horizontal displacement is `integral of w dt = integral of (w/|v|) ds`. Substituting the master identity and changing variable to `u = psi = k_L*s`:

```
chord = (w_0/k_L) * integral from 0 to Psi of rho(u) * exp(i*u) du,     rho(u) = exp(-mu*u)/|v(u)|,   mu = C_D/C_L
```

with `rho(u) > 0` throughout. Launch is along `psi = 0`, so for a 180 degree turn the chord is perpendicular to the launch direction if and only if the real part vanishes:

```
chord perpendicular to launch    <=>    integral from 0 to pi of rho(u)*cos(u) du = 0
```

that is, if and only if the positive weight `rho` is balanced about `u = pi/2`.

**Drag-free**, `mu = 0` so `rho = 1/|v|`, the apex sits exactly at `psi = pi/2`, and `|v|` is symmetric about it. The weight is symmetric, `cos(u)` is antisymmetric, and the integral vanishes identically. The chord is exactly perpendicular to the launch direction, measured at 90.000000 degrees for every launch angle. This is precisely the condition that makes the mirror image of ball A's launch anti-parallel to it, which is why the two balls can be thrown in opposite directions and still meet.

**With drag** the balance breaks, and always in the same direction. Folding the integral about `u = pi/2`, the sign is set by comparing descent steepness against ascent steepness at matched turn angle. Horizontal speed decays exponentially with no floor, while descent speed is capped at terminal velocity, so the descent is always steeper, the integral is strictly positive, and the chord tips **below** perpendicular:

| ball | `mu` | bearing at 10 deg | 30 deg | 60 deg | 85 deg |
|---|---|---|---|---|---|
| frisbee | 0.333 | 89.741 | 88.078 | 85.293 | 85.139 |
| soccer | 1.000 | 88.359 | 81.944 | 76.302 | 76.531 |
| pingpong | 1.500 | 87.064 | 77.929 | 71.187 | 71.415 |

Two further regularities, both regression-tested. The deficit `90 - bearing` depends **only** on `mu = C_D/C_L` and `theta`, not on `C_L` and `C_D` separately nor on mass or size: the pairs (0.6, 0.02), (1.5, 0.05) and (3.0, 0.10) all give identical bearings to six decimals. And the deficit vanishes as `theta^2`: for the frisbee it is 0.00266, 0.01063 and 0.04235 degrees at 1, 2 and 4 degrees, ratios of 4.00 and 3.99.

Ball A and ball B are mirror images through the vertical plane at 90 degrees, so B's endpoint is A's endpoint reflected in `x`, and they coincide if and only if the bearing is exactly 90 degrees. Therefore:

> **With any nonzero drag, two balls thrown in exactly opposite directions with opposite vertical spin never collide, at any speed or launch angle.**

They miss by `2 * |chord| * cos(bearing)`, which for the closure solutions above is 1.0 to 1.9 m for the frisbee, 7.6 to 9.4 m for the pingpong ball, and 30 to 46 m for the soccer ball.

### Drag does make the problem 2D, through a different equation

The original expectation was that drag collapses the one-parameter family into an isolated 2D root-find. The `(F1, F2)` system as written is rank 1, as shown earlier, so that specific claim does not hold. But the physical problem carries a second condition that is invisible without drag because it holds automatically:

```
G1(v0, theta) = psi(t_height) - pi        = 0     (the two events coincide)
G2(v0, theta) = chord bearing - 90 deg    = 0     (the two throws are anti-parallel)
```

Two independent equations, two unknowns. The instinct that drag makes this genuinely two-dimensional was right; the second equation is just not the one originally written down. And the answer is that `G2 < 0` everywhere, so the system has no solution except in the degenerate limit of `theta` going to zero with `v0` going to infinity.

### The one thing that did not work out

Drop the requirement that each ball turn exactly 180 degrees and keep only "thrown oppositely, collide head-on at launch height". Ball A turns `Delta_psi_A` from bearing 0 and ball B turns `Delta_psi_B` from bearing 180. Head-on means the final headings differ by pi, which gives

```
Delta_psi_A + Delta_psi_B = 2*pi
```

The **sum** of the turns is fixed at 360 degrees. The equal split at 180 each is a drag-free accident. Since `Delta_psi = k_L*s`, this says the two path lengths must sum to `2*pi*L_L`. Counting gives four unknowns `(v0A, thetaA, v0B, thetaB)` against four conditions (same time, same `x`, same `y`, head-on), so isolated solutions should exist generically, with one ball turning past 180 degrees so that its chord bearing exceeds 90.

**This did not pan out numerically and is not being claimed.** A 70 by 70 grid search over ball A, with pairing restricted to `Delta_psi_A < 180 < Delta_psi_B`, found candidates with genuinely unequal turns, for example 145 and 215 degrees at launch angles of 6.3 and 3.8 degrees with a scaled residual of 9.9e-3. But every one of four independent seeds, refined by bounded least-squares, drained back to the near-symmetric degenerate corner at `theta` about 1.1 degrees and `v0` about 120 m/s with turns of 179 and 181 degrees, stalling at a 1.5 to 2.0 mm miss. That residual floor is consistent with the `theta^2` law, meaning it is the same obstruction rather than a converged root. So no isolated finite-angle asymmetric solution was found, and the numerical evidence points against one existing in the region searched, but non-existence has not been proved. This is the one genuinely open question here.

## Novelty check

I searched the projectile-dynamics and sports-ballistics literature to see what here is actually new. The honest summary is that the individual ingredients are all old, the exactly integrable structure appears not to be documented, and one result has a direct and previously unremarked ancestor in aeronautics.

**Definitely old, and should be treated as textbook material.** The equations of motion themselves. The projectile arc-length closed form `S = (v0^2/g)[sin + cos^2 * ln(tan + sec)]`, a classical calculus result. The fact that Magnus does no work, being perpendicular to velocity. That a vertical spin axis deflects a ball sideways, thoroughly familiar as the "banana curl" in football and the curveball in baseball. The saturating lift model and the drag-crisis model, both established empirical fits. And, importantly, the technique of writing horizontal velocity as a complex number `u + i*v`: that is completely standard in geophysical fluid dynamics, where it is the usual way to treat Coriolis-driven inertial oscillations.

**The closest known relative, which I had not connected until the novelty check.** In aircraft performance, a wing turning at constant lift coefficient has turn radius `2m/(rho*C_L*S)`, independent of airspeed, because both the required centripetal force and the available lift scale as `v^2`. That is structurally identical to `L_L = 2m/(rho*C_L*A)`. So the speed-independent turning radius is not new in itself; what is new is recognising that a ball with vertical-axis spin is exactly this system, that the resulting turn-per-unit-distance is therefore a constant, and that drag leaves it untouched.

This also sharpens the Coriolis analogy. Magnus with a vertical spin axis has the same `z_hat x v` structure as the Coriolis force, but Coriolis is linear in speed and gives inertial circles of radius `|v|/f`, which depend on speed, whereas Magnus is quadratic in speed and gives circles of radius `L_L`, which do not. The speed-independence is the structural difference, and it is what makes "180 degrees costs a fixed distance" true.

**Not found in the literature, and the strongest candidates for genuinely new.** The exact closed form `w(s) = w_0 * exp(-(k_D - i*k_L)*s)` for combined quadratic drag and Magnus with a vertical spin axis is the main one. The published position is explicitly the opposite: closed forms are reported when either quadratic drag or the Magnus term is negligible, and for the two acting simultaneously the equations are described as highly nonlinear and admitting only perturbative solutions. That literature treats the general or vertical-plane spin orientation; the vertical-axis case is an integrable special case that appears to have been passed over. Following from it: the constant turn per unit arc length and hence `S_tot = pi*L_L` independent of launch conditions and of drag; the logarithmic-spiral hodograph `|w| = |w_0|*exp(-(C_D/C_L)*Delta_psi)` and the measurement corollary for `C_D/C_L`; the curvature law `R_h = L_L*cos(gamma)` with apex curvature giving `L_L` directly; the chord-perpendicularity criterion and the impossibility theorem that follows from it; and the observation that the bearing deficit depends only on `(C_D/C_L, theta)` and vanishes as `theta^2`.

**Caveat on all of the above.** This was a web-literature search, not a systematic review of the exterior-ballistics monograph literature. Works such as McCoy's *Modern Exterior Ballistics* treat spinning projectiles in considerable depth, and it is entirely possible that some form of the arc-length turning result is recorded there in ballistics notation. The impossibility result for the two-ball collision is a construction specific to this problem and is very unlikely to be in print, but it is also a small enough observation that it may simply never have been worth writing down.

Sources consulted: [Exact and approximate solutions to projectile motion in air incorporating Magnus effect (EPJ Plus)](https://link.springer.com/article/10.1140/epjp/s13360-020-00593-4), [Study of the asymptotic motion of a sporting projectile (arXiv)](https://arxiv.org/pdf/2409.15110), [The effect of spin on the flight of a baseball, Nathan (AJP)](https://baseball.physics.illinois.edu/ajpfeb08.pdf), [An analytic solution for point-mass motion with quadratic resistance (arXiv)](https://arxiv.org/pdf/1305.1283), [a Coriolis tutorial, Price (WHOI)](https://www2.whoi.edu/staff/jprice/wp-content/uploads/sites/199/2019/01/aCt_2003.pdf), [Turning flight performance notes (Virginia Tech)](https://archive.aoe.vt.edu/lutze/AOE3104/turningflight.pdf), [Accelerated performance and turns (Engineering LibreTexts)](https://eng.libretexts.org/Bookshelves/Aerospace_Engineering/Aerodynamics_and_Aircraft_Performance_3e_(Marchman)/08:_Accelerated_Performance-_Turns).

## Scorecard

| claim | verdict |
|---|---|
| `d(psi)/ds = k_L` constant | holds to 4.4e-16, and also holds with drag on |
| `S_tot = pi*L_L`, independent of `v0` and `theta` | holds to 1.1e-15, and also holds with drag on |
| ballistic arc-length closed form | holds to 3.4e-10 |
| `v_0 = sqrt(pi*g*L_L/f(theta))` drag-free | holds to 2e-14 |
| drag collapses the family to a 2D root-find | partly vindicated: `(F1,F2)` is rank 1, but the physical problem is genuinely 2D via the bearing condition |
| baseball closes with drag | no, short by 0.496 pi; root exists only at Mach 1.8 |
| balls meet at same point, same time, antipodally | yes, drag-free: 3.6e-15 s, 5.2e-09 m relative to path, bearing 90.000000 |
| closing speed `= 2*|v_f|` | no, it is `2*cos(theta)*|v_f|`, true only as theta goes to 0 |
| feasibility boundary at `C_L/C_D = pi` | no, boundary is `pi/G`; pi is the special case `G = 1`, meaning `v_0 = 1.25*v_t` |
| `v_f/v_0 = exp(-pi*C_D/C_L)` for total speed | no, lower bound only, off by 23x at `C_L/C_D = 0.5` |
| the same formula for horizontal speed | exact, below 1e-9 |
| track is an oval, tightest at launch and landing | yes, and exactly `R_max/R_min = sec(theta)` |
| oppositely-thrown mirror pair collides with drag | never, for any ball at any speed or angle |

Nothing was tuned, clamped or regularised to make any of these agree. Every solver raises `DivergedError` rather than returning an approximate result.

## Running it

```bash
pip install -r requirements.txt
python run_all.py        # writes figures/*.pdf and results.json, prints every number above
python -m pytest -q      # 101 regression tests
```

`run_all.py` takes a few minutes, dominated by the `rtol=1e-10` event-detected solves in the parameter scans.

## Repository layout

```
src/model.py       Params dataclass, coefficient models, batched right-hand side
src/integrate.py   RK45 with event detection, batched fixed-step RK4, crossing refinement
src/solve.py       closure root-finding, the rank-deficiency diagnostic, feasibility scans
src/theory.py      the exact analytic structure and the asymmetric-collision solver
src/sweep.py       symmetry checks, vectorised feasibility map, shape analysis, figures
run_all.py         driver: runs everything and writes figures/ and results.json
tests/             101 regression tests at machine-precision tolerances
figures/           five PDF figures, matplotlib only, no seaborn styling
```

Source is 771 lines across `src/`, of which roughly 130 are matplotlib figure code and about 400 are the physics and solvers.

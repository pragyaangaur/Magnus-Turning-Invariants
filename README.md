# Magnus Turning Invariants

**Full title: Exact Arc-Length Reduction for Projectile Motion under Quadratic Drag and a Vertical-Axis Magnus Force, with a Non-Existence Result for the Symmetric Two-Ball Rendezvous**

A thrower at the origin releases two balls simultaneously in opposite horizontal directions, each spinning about a vertical axis, one about $`+\hat{z}`$ and one about $`-\hat{z}`$. Under gravity, quadratic drag and the Magnus force, can the two balls curve around and collide head-on at the height they were released, each having turned exactly $`180^\circ`$ in azimuth?

The drag-free version has an exact and rather elegant solution. The version with air resistance is impossible, and the reason is more interesting than "the ball runs out of speed". Getting there turned up an exactly integrable structure hiding inside a system that the sports-ballistics literature uniformly describes as having no closed-form solution.

## Contents

1. [Explain it simply](#explain-it-simply)
2. [The model](#the-model)
3. [Results](#results)
4. [The analytic structure](#the-analytic-structure)
5. [Novelty check](#novelty-check)
6. [Summary of results](#summary-of-results)
7. [Running it](#running-it)
8. [Repository layout](#repository-layout)

## Explain it simply

Spin a ball like a merry-go-round, with its axis pointing straight up, and throw it. It curves sideways. That is the Magnus effect, the same physics that makes a curveball curve or a free kick bend around a wall.

Because the spin axis points straight up, the sideways push is always horizontal, and it is always at right angles to the direction the ball is travelling horizontally. That is exactly what a steering wheel does. The key discovery is that this steering wheel is **locked at a fixed angle**. The ball turns a fixed number of degrees for every metre it travels. Not per second, per metre.

That single fact drives everything. To turn $`180^\circ`$, the ball must cover a fixed distance, full stop. Throw it hard, throw it gently, high or low, it makes no difference. For a baseball that distance is 884 metres. That is the invariant $`S_{\text{tot}} = \pi L_L`$, and it holds to fifteen decimal places.

Where does air resistance fit in? Drag is a brake, not a steering input. It pushes straight backwards along the direction of travel, so it cannot bend the path at all. It only changes how quickly the ball covers ground. The clean demonstration: switch gravity off and the ball flies a perfect circle even while drag strips away 94 percent of its speed, and the circle's radius does not budge in the tenth digit.

So why is the real ground track not a circle? Gravity, and gravity alone. The exact rule is $`R = L_L\cos\gamma`$, where $`\gamma`$ is how steeply the ball is climbing or diving. A steep climb means a tight turn, and flat at the top of the arc means the widest turn. That is why the track is an oval rather than a circle, and why the ratio of widest to tightest turn comes out at exactly $`\sec\theta`$.

Why can a baseball not do this? It needs 884 metres of travel to turn around, but drag kills it after roughly 200 metres. It manages about half a turn. Closing the loop would require a launch speed of Mach 1.8.

And the reason the collision fails once air resistance is admitted: a ball with drag lands more steeply than it was thrown, because its horizontal speed decays without limit while its falling speed is capped at terminal velocity. That asymmetry tilts the finish line off square, and the two balls sail past each other. The full accounting is subtler than "steeper everywhere" (see the analytic section below, where that tempting shortcut turns out to be false), but the tilt is real and always in the same direction.

## The model

State vector $`\mathbf{y} = [\mathbf{r}, \mathbf{v}, \boldsymbol{\omega}, \psi, s]`$, eleven components. The accelerations are

```math
\mathbf{a} = -g\,\hat{\mathbf{z}} \;-\; \frac{\rho\, C_D A}{2m}\,\lvert\mathbf{v}\rvert\,\mathbf{v} \;+\; \frac{\rho\, C_L A}{2m}\,\lvert\mathbf{v}\rvert\,\bigl(\hat{\boldsymbol{\omega}}\times\mathbf{v}\bigr)
```

Two lift models are switchable: constant $`C_L`$, and the saturating form $`C_L = 1/(2 + 1/S)`$ with spin parameter $`S = r\lvert\boldsymbol{\omega}\rvert/\lvert\mathbf{v}\rvert`$. Two drag models: constant, and a drag-crisis model dropping $`C_D`$ from $`0.47`$ to $`0.15`$ across $`\mathrm{Re}`$ of $`2\times10^5`$ to $`4\times10^5`$ with a $`\tanh`$ blend. Optional spin decay $`\dot{\boldsymbol{\omega}} = -C_M \rho r^5 \lvert\boldsymbol{\omega}\rvert\boldsymbol{\omega}/I`$. Every one of these is an independent flag on the `Params` dataclass, so effects can be isolated one at a time.

Two constants carry the physics. The **lift length** and the **drag length**,

```math
L_L = \frac{1}{k_L} = \frac{2m}{\rho\, C_L A}, \qquad L_D = \frac{1}{k_D} = \frac{2m}{\rho\, C_D A}, \qquad \frac{L_D}{L_L} = \frac{C_L}{C_D}
```

are the natural turning radius and the distance over which drag noticeably bleeds speed.

The azimuth $`\psi`$ is carried as a state variable and integrated directly from $`d\psi = (v_x\,dv_y - v_y\,dv_x)/(v_x^2 + v_y^2)`$, so it unwraps correctly past $`\pm\pi`$ with no $`\mathrm{atan2}`$ differencing anywhere. Path length $`s`$ is likewise a state variable. Integration is `scipy.solve_ivp` with RK45 at $`\mathrm{rtol} = 10^{-10}`$ and proper event detection, never fixed steps, for the two closure events. A batched fixed-step RK4 handles the vectorised parameter sweeps.

## Results

### Three exact invariants, verified

With drag off, constant $`C_L`$, no spin decay:

| invariant | max relative deviation | status |
|---|---|---|
| $`d\psi/ds = k_L`$, constant along the path | $`4.44\times10^{-16}`$ | holds to machine precision |
| $`S_{\text{tot}} = \pi/k_L`$, independent of $`v_0`$ and $`\theta`$ (12 by 9 sweep) | $`1.11\times10^{-15}`$ | holds to machine precision |
| ballistic arc length equals the closed form | $`3.37\times10^{-10}`$ | holds, at the integrator's tolerance floor |

For a baseball at $`C_L = 0.20`$: $`k_L = 3.555331739605\times10^{-3}\ \mathrm{m}^{-1}`$, $`L_L = 281.2677\ \mathrm{m}`$, $`\pi L_L = 883.6286\ \mathrm{m}`$.

Two of these are stronger than they first appear: **they also hold with drag switched on**. Drag is antiparallel to $`\mathbf{v}`$, so its contribution to $`v_x a_y - v_y a_x`$ is identically zero and it cannot torque the azimuth. Measured with drag on, $`d\psi/ds = k_L`$ to $`6.66\times10^{-16}`$. The invariant $`S_{\text{tot}} = \pi L_L`$ is therefore not a drag-free curiosity, it is exact for any constant-$`C_L`$ model with a vertical spin axis. That is what makes the closure problem tractable, because the azimuth condition collapses to a pure path-length condition.

Two further exact consequences explain why the third result survives even with Magnus active. Magnus with a vertical spin axis has no $`z`$-component, so $`v_z`$ is purely ballistic and the hang time is exactly $`2v_0\sin\theta/g`$, verified to $`10^{-9}\ \mathrm{s}`$. And Magnus does no work while rotating the horizontal velocity without changing its magnitude, so $`\lvert\mathbf{v}_h\rvert = v_0\cos\theta`$ is conserved, verified to $`10^{-9}`$. The speed profile is therefore identical to the drag-free parabola's, which is why the classical arc-length formula

```math
S = \frac{v_0^2}{g}\Bigl[\sin\theta + \cos^2\theta\,\ln\bigl(\tan\theta + \sec\theta\bigr)\Bigr]
```

survives untouched.

### Closure, drag-free

The closed form $`v_0 = \sqrt{\pi g L_L / f(\theta)}`$ with $`f(\theta) = \sin\theta + \cos^2\theta\,\ln(\tan\theta + \sec\theta)`$ is exact:

| $`\theta`$ | $`v_0`$ numerical | $`v_0`$ closed form | rel. diff |
|---|---|---|---|
| $`15^\circ`$ | 130.87424 | 130.87424 | $`8.1\times10^{-12}`$ |
| $`30^\circ`$ | 97.47711 | 97.47711 | $`7.6\times10^{-12}`$ |
| $`45^\circ`$ | 86.88870 | 86.88870 | $`6.7\times10^{-12}`$ |
| $`60^\circ`$ | 85.14578 | 85.14578 | $`5.2\times10^{-12}`$ |
| $`75^\circ`$ | 88.68577 | 88.68577 | $`2.6\times10^{-12}`$ |
| $`85^\circ`$ | 92.17206 | 92.17206 | $`2.2\times10^{-14}`$ |

### The 2D system as originally posed is degenerate

The two residuals $`F_1 = z(t^*) - z_{\text{launch}}`$ and $`F_2 = \psi(t^*) - \pi`$ are two ways of writing one scalar condition, namely that the two events coincide, over three unknowns $`(v_0, \theta, t^*)`$. Running `scipy.optimize.root` on the $`2\times2`$ system and taking the finite-difference Jacobian at the converged point:

```math
J = \begin{pmatrix} 14.382026 & 290.054597 \\ 0.072313 & 1.458382 \end{pmatrix}, \qquad \sigma = \bigl(2.904\times10^{2},\; 1.172\times10^{-6}\bigr), \qquad \frac{\sigma_{\min}}{\sigma_{\max}} = 4.04\times10^{-9}
```

Numerically rank 1. One row is a clean multiple of the other, $`J_{2,:} = J_{1,:}/198.9`$. A 2D Newton solve on this system is solving a singular problem, and the correct formulation is 1D in $`v_0`$ at fixed $`\theta`$. **However**, see the impossibility result below: the physical problem does become genuinely 2D once drag is present, through a second condition that is invisible in the drag-free case because it holds automatically.

### A baseball cannot do it

With $`C_D = 0.35`$ and $`C_L = 0.20`$, scanning $`20 \le v_0 \le 150\ \mathrm{m/s}`$ and $`10^\circ \le \theta \le 88^\circ`$, the best the baseball manages is $`\psi_{\max} = 1.582\ \mathrm{rad} = 0.5037\pi`$. It falls $`0.4963\pi`$ short, at the corner of the domain. Note that $`F_1`$ is undefined over this entire domain, because the $`\psi = \pi`$ event never fires at all, so there is no $`t^*`$ at which to evaluate it. Reporting a combined $`\min(\lvert F_1\rvert + \lvert F_2\rvert)`$ would be misleading, and the honest number is the $`F_2`$ deficit.

A root does exist mathematically, since achievable path length grows like $`\ln v_0`$ without bound, but at $`\theta = 85^\circ`$ it sits at $`v_0 = 616.3\ \mathrm{m/s}`$, which is Mach 1.8 and far outside the validity of an incompressible quadratic-drag model.

### Symmetry, drag-free

Ball A at bearing $`0^\circ`$ with $`+\hat{z}`$ spin, ball B at bearing $`180^\circ`$ with $`-\hat{z}`$ spin, at $`\theta = 45^\circ`$ and $`v_0 = 86.8887\ \mathrm{m/s}`$:

| quantity | value |
|---|---|
| arrival time difference | $`3.55\times10^{-15}\ \mathrm{s}`$ |
| state-level mirror error, max position | $`5.19\times10^{-9}\ \mathrm{m}`$ over a 515.4 m path, $`10^{-11}`$ relative |
| meeting point | $`(-8.1\times10^{-9},\ 515.408148,\ 9.8\times10^{-15})\ \mathrm{m}`$ |
| chord bearing from launch | $`90.000000^\circ`$ |

On the requested $`10^{-9}\ \mathrm{m}`$ endpoint criterion: the raw endpoint miss is $`1.56\times10^{-8}\ \mathrm{m}`$ and it **does not improve** when $`\mathrm{rtol}`$ is tightened from $`10^{-10}`$ to $`10^{-13}`$. It is floored by scipy's event root-location on the dense output, not by the integration. In relative terms it is $`3\times10^{-11}`$ of a 515 m path, at the double-precision limit for a trajectory this long. The state-level test compares $`B(t)`$ against the analytically reflected $`A(t)`$ at common times, sidesteps event location entirely, and is the sharp confirmation that the symmetry is exact.

One note on the geometry: the mirror plane is the vertical plane containing the launch-to-meeting **chord**, not the plane containing the launch direction. The latter leaves ball A's velocity unchanged and is not the relevant symmetry. The chord turns out perpendicular to the launch direction, so for a $`180^\circ`$ turn the two launches are indeed anti-parallel, exactly as the problem requires.

### Closing speed is $`2\cos\theta\,\lvert v_f\rvert`$, not $`2\lvert v_f\rvert`$

Measured ratio at $`45^\circ`$ is $`1.414213562352`$, which is $`\sqrt{2}`$, not $`2`$.

The collision is head-on only in the horizontal plane. At the meeting point both balls are descending, so their vertical velocity components are common mode and cancel in the difference. Only the horizontal parts oppose, giving

```math
\lvert \mathbf{v}_A - \mathbf{v}_B\rvert = 2\lvert\mathbf{v}_h\rvert = 2v_0\cos\theta = 2\cos\theta\,\lvert\mathbf{v}_f\rvert
```

| $`\theta`$ | measured ratio | $`2\cos\theta`$ | rel. diff |
|---|---|---|---|
| $`10^\circ`$ | 1.969615506023 | 1.969615506024 | $`4.7\times10^{-13}`$ |
| $`30^\circ`$ | 1.732050807558 | 1.732050807569 | $`6.3\times10^{-12}`$ |
| $`45^\circ`$ | 1.414213562352 | 1.414213562373 | $`1.5\times10^{-11}`$ |
| $`60^\circ`$ | 0.999999999969 | 1.000000000000 | $`3.1\times10^{-11}`$ |
| $`80^\circ`$ | 0.347296355311 | 0.347296355334 | $`6.5\times10^{-11}`$ |

The value $`2`$ is recovered only as $`\theta \to 0`$. At $`\theta = 60^\circ`$ the closing speed equals a single ball's speed, and above $`60^\circ`$ the balls converge more slowly than either one is travelling.

### Throw tolerance

| perturbation on ball B | miss distance | arrival difference |
|---|---|---|
| $`+5\%`$ launch speed | $`82.98\ \mathrm{m}`$ | $`6.27\times10^{-1}\ \mathrm{s}`$ |
| $`+2^\circ`$ azimuth | $`17.99\ \mathrm{m}`$ | $`1.78\times10^{-15}\ \mathrm{s}`$ |
| $`+10\%`$ spin, constant $`C_L`$ | $`0.0000\ \mathrm{m}`$ | $`3.55\times10^{-15}\ \mathrm{s}`$ |
| $`+10\%`$ spin, saturating $`C_L`$ | $`1083.64\ \mathrm{m}`$ | $`1.78\times10^{-15}\ \mathrm{s}`$ |

Against a 515 m chord and a 0.037 m ball, the throw is hopeless by any human standard. The azimuth number is pure geometry: rotating one trajectory by $`\Delta\alpha`$ moves its endpoint by $`\text{chord}\times\Delta\alpha`$, so the required aim precision is $`r_{\text{ball}}/\text{chord}`$, about $`4.07\times10^{-3}`$ degrees. Speed must be matched to roughly 1 part in $`4.5\times10^{4}`$.

Two remarks on the spin rows. Under constant $`C_L`$, spin magnitude is **exactly inert**, because the Magnus term $`k_L\lvert\mathbf{v}\rvert(\hat{\boldsymbol{\omega}}\times\mathbf{v})`$ depends only on the spin direction. Endpoints for $`\omega = 50`$, $`200`$ and $`800\ \mathrm{rad/s}`$ agree to all twelve printed digits, so the $`0.0000\ \mathrm{m}`$ is correct physics rather than a bug, and spin tolerance is only a meaningful question once $`C_L`$ depends on $`S`$. Under the saturating model the same mismatch gives a 1084 m miss, larger than the chord itself, making spin matching the tightest of the three tolerances rather than the loosest.

### Feasibility map

The sweep is vectorised through an exact reduction rather than a Python loop over `solve_ivp`. Because Magnus does no work **and** has no $`z`$-component, the $`(\lvert\mathbf{v}\rvert, v_z)`$ dynamics decouple completely from the azimuth, so the dimensionless path length $`G \equiv S/L_D`$ and the speed ratio $`v_f/v_0`$ are exactly independent of $`C_L`$. One batched RK4 over a dimensionless $`(v_0/v_t, \theta)`$ grid with $`k_D = g = 1`$ then determines the whole map analytically through

```math
\psi_{\max} = \frac{C_L}{C_D}\; G\!\left(\frac{v_0}{v_t}, \theta\right), \qquad \text{closure} \iff \psi_{\max} \ge \pi
```

The predicted boundary at $`C_L/C_D = \pi`$ is not where it sits. The boundary is at $`C_L/C_D = \pi/G`$, and $`G`$ is a monotonically increasing, unbounded function of $`v_0/v_t`$ growing like $`\ln v_0`$:

| $`G`$ | boundary $`C_L/C_D = \pi/G`$ | requires $`v_0/v_t`$ |
|---|---|---|
| 0.5 | 6.283 | 0.75 |
| 1.0 | $`3.142 = \pi`$ | 1.25 |
| 2.0 | 1.571 | 2.46 |
| 4.0 | 0.785 | 7.23 |
| 8.0 | 0.393 | 54.5 |

So $`C_L/C_D = \pi`$ is the special case $`G = 1`$, meaning a total path of exactly one drag length, which happens only when the launch speed is $`1.25\,v_t`$. That is the implicit assumption in the original derivation. For any faster throw the boundary drops below $`\pi`$, and throwing harder always helps, which is why the contour is a curve in the $`(C_L/C_D,\ \text{loading})`$ plane rather than a horizontal line. The loading axis enters only through $`v_t = \sqrt{2g\,m/(\rho\, C_D A)}`$.

Converged solutions, drag on, $`v_0`$ capped at $`150\ \mathrm{m/s}`$:

| ball | $`C_L/C_D`$ | closes | $`\theta`$ | $`v_0`$ (m/s) | $`R`$ (m) | $`t`$ (s) | $`v_f`$ (m/s) | $`v_f/v_0`$ |
|---|---|---|---|---|---|---|---|---|
| frisbee | 3.00 | yes | $`64^\circ`$ | 19.847 | 5.268 | 2.8129 | 11.704 | 0.590 |
| pingpong | 0.667 | yes | $`80^\circ`$ | 90.952 | 2.376 | 3.9662 | 8.698 | 0.096 |
| soccer | 1.000 | yes | $`76^\circ`$ | 124.204 | 23.321 | 9.5827 | 26.196 | 0.211 |
| golf | 1.000 | no | | | | | | $`\psi_{\max} = 0.804\pi`$ |
| tennis | 0.455 | no | | | | | | $`\psi_{\max} = 0.565\pi`$ |
| baseball | 0.571 | no | | | | | | $`\psi_{\max} = 0.504\pi`$ |

Golf and soccer share $`C_L/C_D = 1.0`$ yet only soccer closes, because feasibility depends on $`v_0/v_t`$ as well as the ratio, and the soccer ball's lower terminal velocity (26.9 versus 45.4 m/s) puts it at a higher $`v_0/v_t`$ for the same capped $`v_0`$. That dependence is exactly what the $`C_L/C_D = \pi`$ prediction leaves out.

**Important caveat on this table.** These $`(v_0, \theta)`$ pairs satisfy "return to launch height with $`\psi = \pi`$" simultaneously, but with drag on they do **not** produce a collision between two oppositely-thrown balls. See the impossibility result below. Read the table as single-ball closure of a $`180^\circ`$ turn, not as a verified collision.

For reference, the drag-free closure at $`45^\circ`$:

| ball | $`C_L`$ | $`C_D`$ | $`L_L`$ (m) | $`v_0`$ (m/s) | $`R`$ (m) | $`t`$ (s) | $`v_f`$ (m/s) |
|---|---|---|---|---|---|---|---|
| baseball | 0.20 | 0.35 | 281.27 | 86.889 | 257.704 | 12.530 | 86.889 |
| tennis | 0.25 | 0.55 | 106.88 | 53.561 | 97.925 | 7.724 | 53.561 |
| pingpong | 0.30 | 0.45 | 11.69 | 17.716 | 10.713 | 2.555 | 17.716 |
| golf | 0.25 | 0.25 | 210.31 | 75.133 | 192.690 | 10.835 | 75.133 |
| soccer | 0.25 | 0.25 | 73.87 | 44.529 | 67.684 | 6.422 | 44.529 |
| frisbee | 0.60 | 0.20 | 8.08 | 14.723 | 7.399 | 2.123 | 14.723 |

Here $`v_f = v_0`$ exactly, since neither gravity nor Magnus changes the speed between two points at the same height.

### Shape

The ground track is an oval, tightest at launch and landing where the speed is greatest and flattest at the apex, exactly as predicted. The exact law is $`R_h = L_L\cos\gamma`$, giving $`R_{\max}/R_{\min} = \sec\theta`$:

| $`\theta`$ | measured $`R_{\max}/R_{\min}`$ | $`\sec\theta`$ | circle-fit RMS residual |
|---|---|---|---|
| $`20^\circ`$ | 1.064178 | 1.064178 | 0.105% of $`R`$ |
| $`45^\circ`$ | 1.414213 | 1.414214 | 0.600% of $`R`$ |
| $`70^\circ`$ | 2.923802 | 2.923804 | 1.670% of $`R`$ |

At $`45^\circ`$ the best-fit circle has $`R = 264.06\ \mathrm{m}`$ with an RMS residual of $`1.585\ \mathrm{m}`$, which is 43 times the ball radius, so the departure from circularity is physically real rather than a fitting artifact. The fitted radius exceeds half the chord (257.70 m), the signature of an oval flattened at its midpoint.

## The analytic structure

### The master identity

Write the horizontal velocity as a single complex number $`w = v_x + i v_y`$. Gravity is vertical and contributes nothing. Drag is antiparallel to $`\mathbf{v}`$ and contributes $`-k_D\lvert\mathbf{v}\rvert w`$. Magnus with a vertical spin axis gives $`\hat{\boldsymbol{\omega}}\times\mathbf{v} = (-v_y,\, v_x,\, 0)`$, which in complex notation is exactly $`i w`$, a $`90^\circ`$ rotation, and contributes $`+i k_L \lvert\mathbf{v}\rvert w`$. So

```math
\frac{dw}{dt} = \bigl(i k_L - k_D\bigr)\,\lvert\mathbf{v}\rvert\,w
```

Now change the independent variable from time to arc length via $`ds = \lvert\mathbf{v}\rvert\,dt`$. The speed cancels identically:

```math
\frac{dw}{ds} = \bigl(i k_L - k_D\bigr)\,w \qquad\Longrightarrow\qquad {\;w(s) = w_0\,e^{-(k_D - i k_L)\,s}\;}
```

A nine-dimensional nonlinear system has an exact closed-form solution for its horizontal part, and gravity never enters it. This is the engine behind every result in this section.

It also reduces the whole problem to a **single scalar ODE**. Taking the turn angle $`u=\psi`$ as independent variable and $`Q=\tan\gamma`$ with $`\gamma`$ the flight-path angle,

```math
\frac{dQ}{du} = -\lambda\,\frac{e^{2\mu u}}{\sqrt{1+Q^2}},\qquad Q(0)=\tan\theta,\qquad \lambda=\frac{g L_L}{v_0^2\cos^2\theta},\qquad \mu=\frac{C_D}{C_L}
```

Two parameters, one equation. It reproduces $`\gamma(\psi)`$ from the full 3-D integration to better than $`10^{-8}`$ rad. Return to launch height is $`\int_0^\pi\sin\gamma\,du=0`$ and the chord is $`\int_0^\pi\cos\gamma\,e^{iu}du/k_L`$, both exact. Because closure fixes $`\lambda`$ once $`(\mu,\theta)`$ are given, the chord bearing is a function of $`(\mu,\theta)`$ **alone**: that scaling, observed numerically earlier, is now a theorem. At $`\mu=0`$ the equation is separable and gives $`\lambda=[\tan\theta\sec\theta+\operatorname{arcsinh}\tan\theta]/\pi`$, which reproduces $`v_0=\sqrt{\pi g L_L/f(\theta)}`$ exactly.

### The exponential formula was right, attached to the wrong quantity

Setting $`\Delta\psi = \pi`$ in the master identity gives

```math
\frac{\lvert w_f\rvert}{\lvert w_0\rvert} = e^{-\pi C_D/C_L} \qquad \text{exactly}
```

for the **horizontal** speed, with gravity fully on, for any launch angle, any ball, any speed. Measured maximum relative error across four balls and three launch angles is below $`10^{-9}`$, which is integrator-limited. More generally $`\lvert w\rvert = \lvert w_0\rvert e^{-(C_D/C_L)\Delta\psi}`$, so the horizontal hodograph is a **logarithmic spiral** whose pitch is set entirely by the lift-to-drag ratio.

It is worth being explicit about what the law does *not* say. Applied to the **total** speed the same expression is badly wrong, underestimating the residual speed by a factor of 23 near the feasibility boundary:

| $`C_L/C_D`$ | predicted | simulated total | ratio |
|---|---|---|---|
| 0.5 | 0.0019 | 0.0435 | 23.3 |
| 1.0 | 0.0432 | 0.2110 | 4.9 |
| 2.0 | 0.2079 | 0.4622 | 2.2 |
| 3.0 | 0.3509 | 0.5938 | 1.7 |
| 5.0 | 0.5335 | 0.7263 | 1.4 |
| 10.0 | 0.7304 | 0.8506 | 1.2 |

But the formula itself is not wrong. The entire discrepancy is $`v_z`$, which gravity re-accelerates and terminal velocity floors. Applied to horizontal speed it is exact.

A useful corollary, since the relation inverts cleanly:

```math
\frac{C_D}{C_L} = \frac{\ln\bigl(\lvert w_0\rvert/\lvert w_f\rvert\bigr)}{\Delta\psi}
```

A real ball's lift-to-drag ratio can therefore be extracted from a single tracked trajectory, by comparing how much its horizontal speed decayed against how far its heading swung. No force measurement and no wind tunnel are needed, and gravity drops out of the expression identically.

### Curvature, and why gravity alone makes the oval

The ground-track radius of curvature is $`R_h = (ds_h/dt)/(d\psi/dt) = \lvert w\rvert/(k_L\lvert\mathbf{v}\rvert)`$, and $`\lvert w\rvert/\lvert\mathbf{v}\rvert = \cos\gamma`$ where $`\gamma`$ is the flight-path angle. So

```math
R_h = L_L\cos\gamma
```

verified below $`10^{-12}`$ relative with drag on or off. Two readings follow. The track is tightest where the ball climbs or dives hardest and flattest at the apex, and at the apex $`\gamma = 0`$ so $`R_h = L_L`$ exactly. That makes $`L_L`$ directly measurable: photographing the ground track and measuring its radius of curvature at the top of the arc yields $`2m/(\rho C_L A)`$ without separate knowledge of $`C_L`$, $`m`$ or $`A`$.

Since drag is antiparallel to $`\mathbf{v}`$ it cannot bend a path, only change the speed along it. With $`g = 0`$ the ground track must therefore be a perfect circle of radius $`L_L`$, drag or no drag:

| ball | drag | fitted radius | $`L_L`$ | rel. err | circle-fit RMS | speed change |
|---|---|---|---|---|---|---|
| baseball | off | 281.267705304 | 281.267705306 | $`5.5\times10^{-12}`$ | $`3.4\times10^{-10}`$ m | 50 to 50.000 m/s |
| baseball | on | 281.267705317 | 281.267705306 | $`3.8\times10^{-11}`$ | $`3.6\times10^{-9}`$ m | 50 to 3.200 m/s |
| frisbee | off | 8.075877046 | 8.075877046 | $`5.4\times10^{-12}`$ | $`9.6\times10^{-12}`$ m | 50 to 50.000 m/s |
| frisbee | on | 8.075877047 | 8.075877046 | $`8.9\times10^{-11}`$ | $`7.3\times10^{-11}`$ m | 50 to 29.619 m/s |

The baseball loses 94 percent of its speed and the circle radius does not move in the tenth digit. Gravity is the sole source of the oval.

### The chord integral, and an impossibility result

Horizontal displacement is $`\int w\,dt = \int (w/\lvert\mathbf{v}\rvert)\,ds`$. Substituting the master identity and changing variable to $`u = \psi = k_L s`$:

```math
\text{chord} = \frac{w_0}{k_L}\int_0^{\Psi}\rho(u)\,e^{iu}\,du, \qquad \rho(u) = \frac{e^{-\mu u}}{\lvert\mathbf{v}(u)\rvert}, \qquad \mu = \frac{C_D}{C_L}
```

with $`\rho(u) > 0`$ throughout. Launch is along $`\psi = 0`$, so for a $`180^\circ`$ turn the chord is perpendicular to the launch direction if and only if the real part vanishes:

```math
\text{chord} \perp \text{launch} \iff \int_0^{\pi}\rho(u)\cos u\,du = 0
```

that is, if and only if the positive weight $`\rho`$ is balanced about $`u = \pi/2`$.

**Drag-free**, $`\mu = 0`$ so $`\rho = 1/\lvert\mathbf{v}\rvert`$, the apex sits exactly at $`\psi = \pi/2`$, and $`\lvert\mathbf{v}\rvert`$ is symmetric about it. The weight is symmetric, $`\cos u`$ is antisymmetric, and the integral vanishes identically. The chord is exactly perpendicular to the launch direction, measured at $`90.000000^\circ`$ for every launch angle. This is precisely the condition that makes the mirror image of ball A's launch anti-parallel to it, which is why the two balls can be thrown in opposite directions and still meet.

**With drag** the balance breaks, and always in the same direction, though not for the most obvious reason. Folding the integral about $`u=\pi/2`$ turns the sign into a comparison of descent steepness against ascent steepness at matched turn angle, $`\lvert\gamma(\pi-u)\rvert`$ versus $`\lvert\gamma(u)\rvert`$, and it is tempting to claim the descent is always steeper so the bracket is positive pointwise. **That claim is wrong.** Drag makes the ascending arc longer than the descending one, so the apex sits at a turn angle *greater* than $`\pi/2`$ (measured: $`0.543\pi`$, $`0.572\pi`$, $`0.635\pi`$). For $`u`$ slightly below $`\pi/2`$ both $`u`$ and $`\pi-u`$ then lie on the ascent and the ordering reverses, so the folded integrand changes sign exactly once. Positivity of the integral is genuinely an integral statement: a large positive contribution near launch outweighs a smaller negative one near the midpoint. What *is* proved, by perturbation theory in the reduced equation, is the leading behaviour

```math
90^\circ - \beta \;=\; \Bigl(\tfrac{8}{3}-\tfrac{24}{\pi^2}\Bigr)\,\mu\,\theta^2 \;+\; O(\mu^2,\theta^4)
```

in radians, whose coefficient is positive **if and only if $`\pi^2 > 9`$**. The analytic constant $`0.2349583`$ is reproduced by Richardson extrapolation of the numerics to $`7\times10^{-6}`$ relative. Beyond that regime the deficit is established numerically, and is strictly positive over a $`60\times60`$ grid spanning $`10^{-2}\le\mu\le3`$ and $`2^\circ\le\theta\le88^\circ`$ (minimum $`1.67\times10^{-4}`$ deg). A proof for all $`(\mu,\theta)`$ remains open. Measured bearings:

| ball | $`\mu`$ | bearing at $`10^\circ`$ | $`30^\circ`$ | $`60^\circ`$ | $`85^\circ`$ |
|---|---|---|---|---|---|
| frisbee | 0.333 | $`89.741^\circ`$ | $`88.078^\circ`$ | $`85.293^\circ`$ | $`85.139^\circ`$ |
| soccer | 1.000 | $`88.359^\circ`$ | $`81.944^\circ`$ | $`76.302^\circ`$ | $`76.531^\circ`$ |
| pingpong | 1.500 | $`87.064^\circ`$ | $`77.929^\circ`$ | $`71.187^\circ`$ | $`71.415^\circ`$ |

Two further regularities, both regression-tested. The deficit $`90^\circ - \text{bearing}`$ depends **only** on $`\mu = C_D/C_L`$ and $`\theta`$, not on $`C_L`$ and $`C_D`$ separately nor on mass or size: the pairs $`(0.6, 0.02)`$, $`(1.5, 0.05)`$ and $`(3.0, 0.10)`$ all give identical bearings to six decimals. And the deficit vanishes as $`\theta^2`$: for the frisbee it is $`0.00266^\circ`$, $`0.01063^\circ`$ and $`0.04235^\circ`$ at $`\theta = 1^\circ, 2^\circ, 4^\circ`$, ratios of 4.00 and 3.99.

Ball A and ball B are mirror images through the vertical plane at $`90^\circ`$, so B's endpoint is A's endpoint reflected in $`x`$, and they coincide if and only if the bearing is exactly $`90^\circ`$. Therefore:

> **With any nonzero drag, two balls thrown in exactly opposite directions with opposite vertical spin never collide, at any speed or launch angle.**

They miss by $`2\lvert\text{chord}\rvert\cos(\text{bearing})`$, which for the closure solutions above is 1.0 to 1.9 m for the frisbee, 7.6 to 9.4 m for the pingpong ball, and 30 to 46 m for the soccer ball.

### Drag does make the problem 2D, through a different equation

The original expectation was that drag collapses the one-parameter family into an isolated 2D root-find. The $`(F_1, F_2)`$ system as written is rank 1, so that specific claim does not hold. But the physical problem carries a second condition that is invisible without drag because it holds automatically:

```math
G_1(v_0, \theta) = \psi(t_{\text{height}}) - \pi = 0, \qquad G_2(v_0, \theta) = \text{bearing} - 90^\circ = 0
```

Two independent equations, two unknowns. Drag therefore does make the problem genuinely two-dimensional, but through the bearing condition rather than through any re-reading of the closure conditions. The answer is that $`G_2 < 0`$ everywhere, so the system has no solution except in the degenerate limit $`\theta \to 0`$ with $`v_0 \to \infty`$.

### The one thing that did not work out

Drop the requirement that each ball turn exactly $`180^\circ`$ and keep only "thrown oppositely, collide head-on at launch height". Ball A turns $`\Delta\psi_A`$ from bearing $`0`$ and ball B turns $`\Delta\psi_B`$ from bearing $`\pi`$. Head-on means the final headings differ by $`\pi`$, which gives

```math
\Delta\psi_A + \Delta\psi_B = 2\pi
```

The **sum** of the turns is fixed at $`360^\circ`$, and the equal split at $`180^\circ`$ each is a drag-free accident. Since $`\Delta\psi = k_L s`$, this says the two path lengths must sum to $`2\pi L_L`$. Counting gives four unknowns $`(v_{0A}, \theta_A, v_{0B}, \theta_B)`$ against four conditions (same time, same $`x`$, same $`y`$, head-on), so isolated solutions should exist generically, with one ball turning past $`180^\circ`$ so that its chord bearing exceeds $`90^\circ`$.

**This branch did not pan out numerically, and no such solution is claimed here.** A 70 by 70 grid search over ball A, with pairing restricted to $`\Delta\psi_A < 180^\circ < \Delta\psi_B`$, found candidates with genuinely unequal turns, for example $`145^\circ`$ and $`215^\circ`$ at launch angles of $`6.3^\circ`$ and $`3.8^\circ`$ with a scaled residual of $`9.9\times10^{-3}`$. But every one of four independent seeds, refined by bounded least-squares, drained back to the near-symmetric degenerate corner at $`\theta \approx 1.1^\circ`$ and $`v_0 \approx 120\ \mathrm{m/s}`$ with turns of $`179^\circ`$ and $`181^\circ`$, stalling at a 1.5 to 2.0 mm miss. That residual floor is consistent with the $`\theta^2`$ law, meaning it is the same obstruction rather than a converged root. So no isolated finite-angle asymmetric solution was found, and the numerical evidence points against one existing in the region searched, but non-existence has not been proved. This is the one genuinely open question in the study.

## Novelty check

The sports-aerodynamics, exterior-ballistics and projectile-motion literature was surveyed to establish what here is actually new. The picture is unusually clear, because the two canonical references for this exact configuration both state in print that no closed form exists.

### What is definitely old

The equations of motion. The classical projectile arc-length formula, a standard calculus result. The fact that Magnus does no work, being perpendicular to velocity. That a vertical spin axis deflects a ball sideways, thoroughly familiar as the banana curl in football and the curveball in baseball. The saturating lift model and the drag-crisis model, both established empirical fits.

Also old, and the direct methodological ancestor of the master identity: **Johann Bernoulli's 1719 hodograph technique**, in which planar projectile motion under quadratic drag is reparametrised by the tangent angle rather than by time, which decouples the equations and yields an exact implicit solution. Euler, Lambert, Borda and Legendre all built on it, and the modern literature still uses it. My arc-length reparametrisation is in that tradition. The difference is that Bernoulli's method gives an implicit solution requiring quadrature, whereas here the turning angle is an exact affine function of arc length, which makes the horizontal solution fully explicit in elementary functions.

Finally, the speed-independence of a constant-$`C_L`$ turn is not new in itself. In aircraft performance a wing turning at constant lift coefficient has radius $`2m/(\rho C_L S)`$, independent of airspeed, because both the required centripetal force and the available lift scale as $`v^2`$. That is structurally identical to $`L_L`$, and the one-line derivation $`R = mv^2/F_{\text{Magnus}} = 2m/(\rho C_L A)`$ turns up informally in physics-forum discussions of curveballs. What does not appear anywhere is this being recognised as an exact three-dimensional invariant that survives drag, or being developed into anything downstream.

### The Lorentz analogy, and why it breaks in a useful way

Magnus with a vertical spin axis has the same $`\hat{\mathbf{z}}\times\mathbf{v}`$ structure as the Lorentz force, and the Magnus-Lorentz correspondence is well established in condensed matter, where the Magnus force on a superfluid vortex is mapped onto an effective magnetic field. But there the Magnus force is **linear** in velocity, exactly like Lorentz, giving inertial circles of radius $`\lvert\mathbf{v}\rvert/f`$ that depend on speed. The aerodynamic Magnus force is **quadratic** in speed, so the radius is $`L_L`$ and does not depend on speed at all. The same distinction separates this from Coriolis-driven inertial oscillations in geophysical fluid dynamics, which is also where the complex-velocity substitution $`w = u + iv`$ is entirely standard. The technique is borrowed; the speed-independence is what makes it pay off here.

### What the literature explicitly says cannot be done

Two references are decisive because they treat precisely this configuration.

**Bray and Kerwin (2003), "Modelling the flight of a soccer ball in a direct free kick",** is the canonical treatment of a ball spinning about a tilted axis with pure sidespin as a limiting case, which is exactly the vertical-axis problem treated here. They write down the same equations of motion and state plainly: *"These equations have no closed form solutions but can be solved numerically using a Runge-Kutta routine."* A full-text search of the  returns zero occurrences of "arc length", "curvature", "radius" or "analytic".

**Nathan (2008), "The effect of spin on the flight of a baseball" (Am. J. Phys.),** is the standard reference for spinning-ball flight. It integrates the equations with fourth-order Runge-Kutta. Full-text search returns zero occurrences of "arc length", "curvature", "closed form", "analytic" or "exact".

Beyond those: a 2020 *EPJ Plus* study of projectile motion incorporating the Magnus effect reports closed forms only when either the quadratic drag or the Magnus term is negligible, and describes the simultaneous case as highly nonlinear and admitting only perturbative solutions. A 2024 arXiv study of asymptotic motion under drag plus Magnus obtains the velocity hodograph only as an approximate implicit formula. An arXiv abstract search for "Magnus" together with "quadratic drag" returns zero results; "Magnus" with "hodograph" returns exactly one paper, the asymptotic one just mentioned. In exterior ballistics, McCoy's *Modern Exterior Ballistics* notes that the Magnus **force** on a spinning projectile is usually small enough to neglect and that it is the Magnus **moment** that matters, so the relevant lateral deflection there is gyroscopic drift via the yaw of repose, a different mechanism entirely.

### What appears to be new

Against that background, these look genuinely new:

1. The exact closed form $`w(s) = w_0 e^{-(k_D - i k_L)s}`$ for combined quadratic drag and Magnus with a vertical spin axis. The vertical-axis case is an integrable special case that the literature appears to have passed over while stating in general terms that no closed form exists.
2. The constant turn per unit arc length, and hence $`S_{\text{tot}} = \pi L_L`$ independent of launch conditions **and of drag**.
3. The exact logarithmic-spiral hodograph $`\lvert w\rvert = \lvert w_0\rvert e^{-(C_D/C_L)\Delta\psi}`$, and the corollary that $`C_D/C_L`$ can be read off a single tracked trajectory.
4. The curvature law $`R_h = L_L\cos\gamma`$, with apex curvature giving $`L_L`$ directly.
5. The chord-perpendicularity criterion $`\int_0^\pi \rho(u)\cos u\,du = 0`$, and the impossibility theorem that follows from it.
6. The observation that the bearing deficit depends only on $`(C_D/C_L, \theta)`$ and vanishes as $`\theta^2`$.

One caveat remains. The full text of McCoy's monograph and of several paywalled ballistics journals was not available for this survey, so it cannot be ruled out that some form of the arc-length turning law is recorded in ballistics notation somewhere. But the two most authoritative open references for spinning sports balls both proceed numerically and state that no closed form is available, which is a strong signal that this reduction is not in circulation.

### Sources

- [Bray and Kerwin, *Modelling the flight of a soccer ball in a direct free kick*](https://people.stfx.ca/smackenz/courses/hk474/labs/jump%20float%20lab/bray%202002%20modelling%20the%20flight%20of%20a%20soccer%20ball%20in%20a%20direct%20free%20kick.pdf)
- [Nathan, *The effect of spin on the flight of a baseball*, Am. J. Phys. 76, 119 (2008)](https://baseball.physics.illinois.edu/ajpfeb08.pdf)
- [*Exact and approximate solutions to projectile motion in air incorporating Magnus effect*, EPJ Plus (2020)](https://link.springer.com/article/10.1140/epjp/s13360-020-00593-4)
- [*Study of the asymptotic motion of a sporting projectile taking into account the Magnus force*, arXiv:2409.15110](https://arxiv.org/abs/2409.15110)
- [*An analytic solution to the equations governing the motion of a point mass with quadratic resistance*, arXiv:1305.1283](https://arxiv.org/pdf/1305.1283)
- [McCoy, *Modern Exterior Ballistics*, full text](https://archive.org/stream/ModernExteriorBallisticsTheLaunchAndFlightDynamicsOfSymmetricProjectiles2ndEd.R.McCoy/Modern+Exterior+Ballistics+-+The+Launch+and+Flight+Dynamics+of+Symmetric+Projectiles+2nd+ed.+-+R.+McCoy_djvu.txt)
- [*Magnus force in superfluids and superconductors*, Phys. Rev. B 55, 485 (1997)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.55.485)
- [Price, *A Coriolis tutorial* (WHOI)](https://www2.whoi.edu/staff/jprice/wp-content/uploads/sites/199/2019/01/aCt_2003.pdf)
- [*Turning flight performance notes*, Virginia Tech](https://archive.aoe.vt.edu/lutze/AOE3104/turningflight.pdf)
- [*Accelerated performance and turns*, Engineering LibreTexts](https://eng.libretexts.org/Bookshelves/Aerospace_Engineering/Aerodynamics_and_Aircraft_Performance_3e_(Marchman)/08:_Accelerated_Performance-_Turns)
- [Lubarda, *A review of the analysis of wind-influenced projectile motion* (UCSD)](http://maeresearch.ucsd.edu/~vlubarda/research/pdfs/AAM22.pdf)

## Summary of results

| statement | status |
|---|---|
| $`d\psi/ds = k_L`$ constant | holds to $`4.4\times10^{-16}`$, and also holds with drag on |
| $`S_{\text{tot}} = \pi L_L`$, independent of $`v_0`$ and $`\theta`$ | holds to $`1.1\times10^{-15}`$, and also holds with drag on |
| ballistic arc-length closed form | holds to $`3.4\times10^{-10}`$ |
| $`v_0 = \sqrt{\pi g L_L/f(\theta)}`$ drag-free | holds to $`2\times10^{-14}`$ |
| drag makes the closure problem two-dimensional | yes, but via the bearing condition; the $`(F_1,F_2)`$ pair is rank 1 |
| a baseball can close a $`180^\circ`$ turn with drag | no, short by $`0.496\pi`$; a root exists only at Mach 1.8 |
| balls meet at same point, same time, antipodally | yes, drag-free: $`3.6\times10^{-15}`$ s, $`5.2\times10^{-9}`$ m relative to path, bearing $`90.000000^\circ`$ |
| closing speed | $`2\cos\theta\,\lvert v_f\rvert`$, not $`2\lvert v_f\rvert`$; the two agree only as $`\theta\to0`$ |
| feasibility boundary | at $`C_L/C_D = \pi/G`$; the value $`\pi`$ is the special case $`G = 1`$, meaning $`v_0 = 1.25 v_t`$ |
| $`e^{-\pi C_D/C_L}`$ applied to **total** speed | a lower bound only, off by $`23\times`$ at $`C_L/C_D = 0.5`$ |
| $`e^{-\pi C_D/C_L}`$ applied to **horizontal** speed | exact, below $`10^{-9}`$ |
| ground track is an oval, tightest at launch and landing | yes, with exactly $`R_{\max}/R_{\min} = \sec\theta`$ |
| an oppositely-thrown mirror pair collides, with drag | never, for any ball at any speed or launch angle |

Nothing was tuned, clamped or regularised to make any of these agree. Every solver raises `DivergedError` rather than returning an approximate result.

## Running it

```bash
pip install -r requirements.txt
python run_all.py        # writes figures/*.pdf and results.json, prints every number above
python -m pytest -q      # 139 regression tests
```

`run_all.py` takes a few minutes, dominated by the $`\mathrm{rtol} = 10^{-10}`$ event-detected solves in the parameter scans.

## Repository layout

```
src/model.py       Params dataclass, coefficient models, batched right-hand side
src/integrate.py   RK45 with event detection, batched fixed-step RK4, crossing refinement
src/solve.py       closure root-finding, rank-deficiency diagnostic, feasibility scans
src/theory.py      the exact analytic structure and the asymmetric-collision solver
src/reduced.py     reduction to one scalar ODE in the turn angle, and the deficit map
src/sweep.py       symmetry checks, vectorised feasibility map, shape analysis, figures
run_all.py         driver: runs everything and writes figures/ and results.json
tests/             139 regression tests at machine-precision tolerances
figures/           six PDF figures, matplotlib only, no seaborn styling
```

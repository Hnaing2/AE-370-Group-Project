# Spec 2/3 — Numerics: contracts, integrators, simulation, Jacobian

Lean coding extract of the AE 370 README (Revision 4); section numbers match the full README. Covers `src/integrators.py`, `src/simulation.py`, and the Jacobian/stability parts of `src/physics.py` and `validation/`. Conventions (time-first signatures, state order, closures `f(t, u)`) are in spec/model.md.

## 7. Python contracts and data layout

The module map follows the existing repository layout (see Day 1 decisions for branch ownership).

~~~text
src/
  parameters.py       REFERENCE_CASE, STUDY_BASELINE, build_params, validate, initial state
  physics.py          constitutive quantities, rhs, jacobian
  integrators.py      FE, RK4, AB3, BE single steps, step_xx(f, t, u, dt, ...)
  simulation.py       powered/coast segments, apogee, result assembly, metrics
validation/           test_*.py benchmarks and reference check; convergence, stability, dt and cost studies
results/              sweep and figure scripts
results/data/         generated CSV/JSON summaries and the frozen input file
report/figures/       generated final plots
run_all.py            reproduce the workflow
~~~

Two configuration objects are enough:

- **params:** a frozen dataclass (or dict) holding the Section 3.6 inputs, the derived quantities from the builder, the scenario name, and provenance labels.
- **options:** a small dict with method, nominal dt, nonlinear-solver settings, and t_end.

### 7.1 Input/output contracts

| Function | Inputs | Outputs |
|---|---|---|
| build_params(inputs) | Independent inputs (Section 3.6) | Frozen params with derived quantities at full precision |
| validate(params, u0, options) | Params, initial state, and options containing dt | Validation report; raise on nonfinite/invalid geometry, mass, gas, valve bounds, missing data, or t_cut/dt mismatch |
| build_initial_state(params) | h0, V0, gamma0, masses, a0 | NumPy vector u0, shape (8,) |
| derived_quantities(u, params, phase) | One state | Dict: M, Vu, Pu, Ps, Ts, r, rho, g, D, F_eng, T_ax, qp, qr, a_cmd, validity flags |
| rhs(t, u, params, phase) | Time and one state | Derivatives, shape (8,) |
| jacobian(t, u, params, phase) | Same as rhs | J = df/du, shape (8,8), by central finite differences of rhs |
| step_fe / step_rk4 | f(t, u), t, u, dt | Next state and RHS-call count |
| step_be | f(t, u), optional Jacobian function, t, u, dt, scales/tolerances | Next state plus solver status, achieved scaled residual, call counts |
| step_ab3 | u, three equal-spaced derivatives, dt | Next state; history handled by the driver |
| simulate(method, params, options) | Method, params, nominal dt | SimulationResult (below) |
| sample_at_time(result, t_query) | Result and time | State/derived quantities by cubic Hermite interpolation within one phase segment |
| compute_metrics(result) | Result | Pressure extrema and droop, thrust metrics, impulse by the mass-use identity in Section 12, x and h at t_eval, apogee/time/downrange, peak axial specific force, status |
| characterize(result, params) | Pilot states | Eigenvalues, neutral/unstable flags, stability step estimates per method |
| convergence_study(cases, methods, dt_values, reference) | Bounded cases and solvers | Errors, observed orders, failures, cost |
| select_timestep(study, tolerances) | Measured results | Selected method/dt and worst-case evidence row |
| parameter_sweep(cases, method, dt) | Frozen selection and case grid | One row per case plus saved parameter snapshots |

SimulationResult:

- Two segments, powered and coast, each with t, U (n+1, 8), and F (n+1, 8), the RHS at each accepted state. The cutoff state appears in both segments, with powered-side F in the first and coast-side F in the second.
- F must be the derivative at the accepted state. RK4's next-step k1 is that derivative and can be reused; RK4's final stage is not. Count any extra calls.
- events: apogee (time, state).
- status: completed / no_apogee / physical_model_invalid / solver_failed.
- counters: RHS calls, Jacobian calls, nonlinear iterations when exposed, AB3 restarts, wall time.
- the params and options used.

Recompute derived quantities (Pu, thrust, and so on) from U after the run.

### 7.2 RHS outline

~~~python
def rhs(t, u, params, phase):
    q = derived_quantities(u, params, phase)   # thrust, qp, qr are zero in coast
    x, h, V, gamma, mp, ms, mu, a = u

    du = np.zeros(8)
    du[0] = V * np.cos(gamma)
    du[1] = V * np.sin(gamma)
    du[2] = (q["T_ax"] - q["D"]) / q["M"] - q["g"] * np.sin(gamma)
    du[3] = -q["g"] * np.cos(gamma) / V
    du[4] = -q["qp"]
    du[5] = -q["qr"]
    du[6] = q["qr"]
    du[7] = (q["a_cmd"] - a) / params.tau_v
    if params.scenario == "fixed_orifice":
        du[7] = 0.0
    if phase == "coast":
        du[4:] = 0.0  # freeze mp, ms, mu, a
    return du
~~~

The phase is chosen by the driver from the step interval, never inside an RK stage or a BE residual. The RHS must not modify its input, clip masses, or depend on the integrator.

Report x(t_eval), h(t_eval), and apogee by cubic Hermite interpolation within a single phase segment, using stored states and their derivatives. Never interpolate across t_cut. The O(dt^4) interpolation error assumes a smooth segment and accurate endpoint derivatives. Validate it with the ballistic benchmark.

## 8. Numerical methods and cutoff handling

### 8.1 Forward Euler

$$
u_{n+1}=u_n+\Delta t f(t_n,u_n).
$$

One RHS evaluation per step; first-order global accuracy on smooth problems. A baseline, not an assumed final choice.

### 8.2 Classical RK4

Use the four stages and weighted update in the PDF with one consistent argument order. Four RHS calls per step; fourth-order accuracy on smooth problems.

### 8.3 AB3

$$
u_{n+1}=u_n+\frac{\Delta t}{12}
(23f_n-16f_{n-1}+5f_{n-2}).
$$

- Generate u1 and u2 with RK4; cache f0, f1, f2; then one new RHS call per step.
- At t_cut, discard the derivative history and restart with two RK4 steps in the coast segment.
- Count startup and restart evaluations exactly.

### 8.4 Backward Euler

Solve

$$
G(z)=z-u_n-\Delta t f(t_n+\Delta t,z)=0
$$

with the phase of the step interval. Implement the time step and residual yourselves and solve G with scipy.optimize.root or fsolve, as the assignment permits. A custom Newton solver is optional.

1. Scale variables and residuals: u = S w and G_w = w_new − w_old − dt S^-1 f(t_new, S w_new), with positive characteristic scales.
2. Supply the finite-difference Jacobian if desired; the scaled residual Jacobian is I − dt S^-1 J S.
3. Start from a valid predictor or the current state; never clip physical states to rescue an iterate.
4. Accept a step only with solver success and a small achieved scaled residual (start near 1e-10). A success flag or a small `xtol` alone is not sufficient: `xtol` controls convergence between iterates, not the residual or conservation error.
5. Flag nonconverged steps and report the achieved residual. Tighten the algebraic settings once on representative runs to confirm outputs do not change.
6. Count every RHS call, including finite differences, plus wall time; count Jacobian calls and iterations when the library exposes them.

For the unscaled residual, helium conservation gives the exact identity

$$
(m_{s,n+1}+m_{u,n+1})-(m_{s,n}+m_{u,n})=G_{m_s}+G_{m_u},
$$

so accumulated drift reflects the achieved mass-row residuals (plus round-off). Report maximum scaled residual, relative helium drift, solver status, and the tightening check. The Section 11.1 helium limits are measured acceptance targets.

### 8.5 Cutoff, apogee, and validity handling

The scheduled cutoff removes **cutoff** event localization; apogee is still localized within a coast step.

- **Cutoff.** Choose every dt so that N_cut = t_cut/dt is an integer (for example 2.5, 1.25, 0.625 ms, ...). Compute times as t_n = n dt rather than by accumulation. Steps 1 to N_cut use the powered phase; the state at t_cut starts the coast segment. No short step is ever needed. Validate that round(t_cut/dt) × dt equals t_cut to round-off.
- **Fuel depletion.** Expected not to occur in the screened baseline envelope (Section 5.1); assert mp > 0 at trial stages and accepted steps. A failure marks the candidate run physical_model_invalid.
- **Apogee.** Detect when V sin(gamma) changes from positive to negative in the coast segment; form the phase-consistent cubic Hermite interpolant using accepted endpoint states and RHS values, then solve its vertical-velocity zero in the bracket [0,1] (e.g. bisection). Integration stops at the step after apogee.
- **Validity.** Invalid geometry, gas state, or V ≤ 0 marks the run physical_model_invalid. Pu < 0.5 Pset records the assumed engine-validity flag; later results are labeled extrapolation.
- **Saturation.** Command saturation corners are not split. The clip and the cutoff reduce smoothness, so do not promise fourth-order convergence of the full trajectory; measure it.

If a trial stage leaves the physical domain, mark that candidate dt as failed; do not switch to adaptive stepping.

## 9. Jacobian and stability characterization

### 9.1 Analytic Jacobian for the regulated, powered, choked branch

Production code uses a central finite-difference Jacobian of the actual RHS (scaled as in Section 9.2). The analytic choked-branch Jacobian below is kept for the report and as a cross-check; it does not cover the subcritical branch.

Use zero-based indices for [x, h, V, gamma, mp, ms, mu, a]. Define:

$$
P_p=\frac{\partial P_u}{\partial m_p}
=\frac{m_uR_{\rm He}T_u}{\rho_p V_u^2},
\qquad
P_m=\frac{\partial P_u}{\partial m_u}
=\frac{R_{\rm He}T_u}{V_u},
$$
$$
g_h=-\frac{2g}{R_E+h},\quad D_h=-D/H,\quad
D_V=\rho V C_DA_{\rm ref},\quad
K_A=K_F\cos\delta,\quad
B=\frac{K_F}{I_{\rm sp}g_0},\quad A=\frac{T_{\rm ax}-D}{M}.
$$

For choked flow:

$$
q_s=\frac{\partial q_r}{\partial m_s}
=\frac{\gamma_{\rm He}+1}{2m_s}q_r,\qquad
q_a=\frac{\partial q_r}{\partial a}
=C_{\rm reg}P_s/\sqrt{T_s}.
$$

Let S_cmd = 1 when 0 < K_p(Pset − Pu) < 1 and 0 in the saturated branches; document a one-sided convention at the boundaries.

All unlisted entries are zero:

~~~text
J[0,2] = cos(gamma)
J[0,3] = -V*sin(gamma)
J[1,2] = sin(gamma)
J[1,3] = V*cos(gamma)

J[2,1] = D/(H*M) - g_h*sin(gamma)
J[2,2] = -D_V/M
J[2,3] = -g*cos(gamma)
J[2,4] = K_A*P_p/M - A/M
J[2,5] = -A/M
J[2,6] = K_A*P_m/M - A/M

J[3,1] = -g_h*cos(gamma)/V
J[3,2] = g*cos(gamma)/V**2
J[3,3] = g*sin(gamma)/V

J[4,4] = -B*P_p
J[4,6] = -B*P_m

J[5,5] = -q_s
J[5,7] = -q_a
J[6,5] = q_s
J[6,7] = q_a

J[7,4] = -S_cmd*K_p*P_p/tau_v
J[7,6] = -S_cmd*K_p*P_m/tau_v
J[7,7] = -1/tau_v
~~~

This corrects the PDF's angle row and regulator feedback signs. For the fixed-orifice comparator, a_dot = 0 and the **entire last row is zero**; do not reuse the regulated valve row. In the subcritical branch q_r also depends on Pu (and hence on mp and mu); the finite-difference Jacobian captures this automatically.

### 9.2 Jacobian check

Use dimensionless state scaling and central differences at representative smooth states, with more than one perturbation size. Do not perturb across t_cut, command corners, or invalid gas states. Check saturated and unsaturated branches separately.

### 9.3 Trajectory sampling and interpretation

Sample the pilot trajectory at initialization, the first Pset crossing, regulator transitions, late burn, just before cutoff, and coast. Add the fastest-valve/highest-gain corner.

The x column of the Jacobian is zero and ms + mu is conserved, so neutral directions make a max/min eigenvalue ratio meaningless. Report stable eigenvalues and their scales, neutral directions separately, any positive-real-part modes, and the frozen-Jacobian step limits. Frozen eigenvalues are diagnostics, not a nonlinear stability proof.

### 9.4 Stability checks by method

For each stable eigenvalue lambda and trial dt, with z = lambda dt:

$$
R_{\rm FE}=1+z,\quad
R_{\rm RK4}=1+z+z^2/2+z^3/6+z^4/24,\quad
R_{\rm BE}=1/(1-z).
$$

Require |R(z)| ≤ 1 with an interior margin, evaluated at the actual complex z. For AB3, compute all roots of

$$
q^3-(1+23z/12)q^2+(16z/12)q-5z/12=0
$$

(NumPy coefficients [1, -1-23*z/12, 16*z/12, -5*z/12]) and require them inside the unit disk, with any unit-modulus roots simple. Real-axis endpoints (FE −2, RK4 −2.785, AB3 −6/11) and AB3's imaginary-axis extent (~0.72) are screening values only. For FE with Re(lambda) < 0, dt_max = −2 Re(lambda)/|lambda|^2.

BE is A-stable and L-stable for the linear test equation; its usable dt still depends on accuracy and on the nonlinear solve.

A reduced two-state linearization of the regulator loop gives eigenvalue magnitudes up to roughly 400 1/s over the proposed grid, with damping ratios as low as ~0.06. A timestep near 1 ms is a candidate to test, not a result. For FE, dt ≤ tau_v keeps the isolated valve update a convex combination but does not guarantee coupled stability. Never clip a into [0, 1] to hide an unstable run.


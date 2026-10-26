# Spec 3/3 — Verification, timestep selection, studies, figures

Lean coding extract of the AE 370 README (Revision 4); section numbers match the full README. Covers `validation/`, `results/`, and `run_all.py`. Tolerances here feed `select_timestep`.

## 10. Verification: establish both solver order and correct physics

| Test | Configuration | Independent answer and purpose |
|---|---|---|
| Scalar decay | u_dot = -u, u(0)=1 | exp(-t); orders FE=1, BE=1, AB3=3, RK4=4. |
| Time-dependent scalar | u_dot = t-u, u(0)=1 | u = t-1+2exp(-t); catches wrong evaluation times or argument order. |
| Ballistic trajectory | No thrust/drag, constant g | vx = vx0, vz = vz0-gt, x = x0+vx0 t, h = h0+vz0 t-g t^2/2, V = hypot(vx,vz), gamma = atan2(vz,vx). Checks the gamma equation and the Hermite apogee. |
| Vertical constant-force burn | Constant engine force, no drag, gamma = pi/2, constant g and qp | M = M0-qp t; V = V0+Isp g0 cos(delta) ln(M0/M)-g0 t. Checks mass, force projection, and gravity. Needs a test-only fixed-Pu scenario. |
| Constant valve command | Command frozen at a_bar | a = a_bar+(a0-a_bar)exp(-t/tau_v). |
| Passive isothermal ullage | qr = 0 | Pu Vu = mu0 R Tu. |
| Helium balance | Any regulated run | ms + mu constant within the Section 11.1 limits. |
| Ullage pressure derivative | Powered model | dPu/dt = (R Tu/Vu) qr − (Pu/Vu)(qp/rho_p) matches the chain-rule derivative of the RHS. |
| Force-flow balance | Powered model | F_eng/(Isp g0) = qp, T_ax/F_eng = cos(delta), M_dot = −qp. |
| Choked/subcritical flow | Several pressure ratios | Matches the choked branch, continuous at r_*, tends to zero as Pu → Ps. |
| Jacobian | Several smooth states | Analytic and scaled finite-difference matrices agree on the choked branch. |
| Coupling isolation | Same gas inputs, different Cd/gamma0 | Pressure and propellant histories identical. |
| Cutoff bookkeeping | Any run | Thrust and qp zero after t_cut; mp, ms, mu, a constant in coast; t_cut lands exactly on a step. |
| Mission reference | Section 3.6 baseline, RK4 | Section 3.8 reference outputs within ~0.1%. |

**Core set:** scalar decay, time-dependent scalar, ballistic, vertical burn, helium balance, flow law, Jacobian, cutoff bookkeeping, and mission reference. The others are quick optional checks.

Optional altitude benchmark for the vertical burn:

$$
h(t)=h_0+V_0t+c\left[t-\frac{M_0-q_pt}{q_p}
\ln\frac{M_0}{M_0-q_pt}\right]-\frac12g_0t^2,\quad
c=I_{\rm sp}g_0\cos\delta.
$$

Use exact answers coded independently of the production RHS where possible. The ullage is isothermal, so Pu Vu^gamma is not an invariant. Full-model convergence establishes consistency with the implemented equations; the independent tests add correctness; neither validates against hardware.

## 11. Reference solution, errors, and one timestep for the study

### 11.1 Error targets

| Quantity | Tolerance |
|---|---|
| Powered pressure history and extrema | 0.1% of Pset (8 kPa) |
| Powered axial thrust history | 0.1% of nominal axial force (≈ 534 N) |
| Peak altitude | 10 m |
| x and h at t_eval | 10 m |
| Downrange at apogee | 10 m |
| Time of apogee | 0.05 s |
| Total axial impulse | 0.1% of nominal axial impulse |
| Peak axial specific force | 0.01 g0 |
| Helium balance (validity check, not in E_score) | relative drift ≤ 1e-10 for FE, RK4, AB3; ≤ 1e-8 for BE, measured |

Accept a method and dt only with E_score ≤ 0.5 (Section 11.4). These are study choices, not crew-safety or flight-prediction requirements. The 10 m integration target is distinct from atmosphere, vehicle-mass, and force-model uncertainty: changing only nominal density to a piecewise 1976 Standard Atmosphere moved apogee by about 393 m in an independent check. Prototype corner runs differ by up to ~600 kPa in minimum pressure and ~0.8 km in apogee; neighbouring grid points will differ less, so check their separation against these tolerances after the pilot sweep. Avoid a single unscaled vector norm.

### 11.2 Reference construction and history errors

1. Use the self-written RK4 with the same physics at a substantially smaller dt that still divides t_cut.
2. Refine once more; changes in every reported metric should be no more than ~5% of its tolerance.
3. Tighten BE algebraic settings to show they do not dominate.
4. Do not use solve_ivp. The Section 3.8 prototype values check the implementation; they are not the timestep reference.

Because every run cuts off at the same t_cut, pressure and thrust histories are compared at the same physical times over the whole powered interval:

$$
E_P=\max_{0\le t\le t_{\rm cut}}
|P_{u,\Delta t}(t)-P_{u,\rm ref}(t)|,\qquad
E_T=\max_{0\le t\le t_{\rm cut}}
|T_{\rm ax,\Delta t}(t)-T_{\rm ax,\rm ref}(t)|,
$$

using powered-side values at t_cut. Evaluate on a shared time grid with phase-aware interpolation, refining the sampling until the errors change by less than ~5% of their tolerances, with extra points around extrema. Compare pressure extrema separately. Flight outputs at t_eval use the same physical time in both runs; apogee uses each run's own interpolated value.

### 11.3 Timestep trial sequence

After a valid pilot and the stability screening, use a halving sequence of steps that each divide t_cut, for example dt_j = 2.5 ms / 2^j. Start no coarser than about tau_v,min/2 for the explicit methods and include both failing and passing steps.

Compare all four methods on the nominal case and the difficult corners first. Then use only the selected method for the regulator grid, the Cd × gamma0 grid, and the fixed-orifice case, and verify the chosen dt with finer runs on that complete finite case set.

For each method/case/dt save: validity and solver status; pressure and thrust history errors; apogee and t_eval errors; RHS/Jacobian calls, AB3 restarts, nonlinear iterations, and wall time; observed order log2(E(dt)/E(dt/2)) on suitable intervals.

### 11.4 Which cases must pass?

The nominal case; the corners of the regulator grid (tau_v ∈ {5, 100} ms × kappa ∈ {5, 100}); the fixed-orifice case; the four corners of the Cd × gamma0 grid; and the optional gas-starved case only if it is run.

$$
E_{\rm score}=
\max_{\text{tested cases, required outputs}}
\frac{|Q_{\Delta t}-Q_{\rm ref}|}{\mathrm{tol}_Q}.
$$

Select a method/dt with E_score ≤ 0.5, no invalid states, successful nonlinear solves, and acceptable measured cost. Keep the worst-case evidence row. Corner checks are screening, not proof for interior cases; verify the full finite grid with the chosen dt.

## 12. Regulated/passive comparison and engineering outputs

The PDF asks for an equal-initial-wet-mass passive comparison but does not define its plumbing. Use the same hardware with a fixed valve opening, the "unregulated fixed-orifice" system:

$$
a_{\rm fix}=\frac{q_{r,\rm req}\sqrt{T_{s0}}}
{C_{\rm reg}P_{s0}\Phi(P_{\rm set}/P_{s0})}\approx0.231.
$$

It has the same hardware, helium, and initial wet mass, and matches nominal demand at the initial supply conditions. The regulated valve starts closed (qr(0) = 0); the fixed orifice starts at a_fix (qr(0) ≈ 2.8 kg/s), so the comparison includes the startup policy as well as feedback. Set a0 = a_fix and a_dot = 0; do not run the regulator equation in this case. Do not retune a_fix per controller case. A fully open valve would overpressure the tank. Isolated-ullage blowdown and a matched-initial-opening variant are optional.

Metrics:

- Minimum/maximum powered Pu and droop. Measure regulator-induced overshoot after the first Pset crossing; report the 3.5% initial overpressure separately.
- Peak/mean axial thrust and total axial impulse over the burn. Under the specified constant-Isp, constant-cant model, compute impulse from the mass-use identity I_ax = Isp g0 cos(delta) [mp0 − mp(t_cut)], avoiding a low-order post-processing quadrature error; verify against an independently refined thrust integral if used as a physics check.
- Propellant used, propellant and pressurant remaining at cutoff.
- x(t_eval), h(t_eval) − h0, apogee, its time and downrange, peak modeled axial specific force (T_ax − D)/M; distinguish it from dV/dt and from an unqualified reported g-load.
- Validity flags reached.

**Required drag and flight-path-angle sweep (research question 3).** A 3 × 3 grid, Cd ∈ {0.4, 0.6, 0.8} × gamma0 ∈ {0.93, 1.10, 1.27} rad, at the nominal regulator settings with the selected method and dt: 9 runs including the nominal case. The bounds are exploratory (±33% drag, ±10° angle), not uncertainty intervals, and do not compensate for the atmosphere approximation. Prototype runs over a slightly wider grid stayed valid with apogees of roughly 33–46 km. Pressure histories must be identical across the 9 runs (coupling check).

Final figures:

1. tau_v × kappa heatmap of minimum pressure (and of apogee).
2. Downrange displacement or apogee versus tau_v for selected gains.
3. Regulated versus fixed-orifice metrics.
4. Error versus measured cost, showing why the method/dt was chosen.
5. x(t_eval) and apogee versus Cd, one line per gamma0.

Pressure and thrust histories illustrate the mechanism, but the envelope figures answer the research questions. x is displacement from the abort point, not clearance from the booster, since no booster trajectory is modeled.

## 15. Completion criteria

The implementation is ready for final results when:

- One equation set and one canonical input file are frozen.
- The Section 3.8 reference outputs are reproduced.
- All four methods pass the smooth verification cases at their theoretical orders.
- Physical limiting cases, cutoff bookkeeping, and conservation checks pass.
- The Jacobian matches finite differences on the choked branch.
- AB3 restarts correctly at cutoff.
- Reference accuracy and BE algebraic error are below budget.
- One method/dt passes the complete finite case set with E_score ≤ 0.5.
- Measured cost supports the method choice.
- Figures, tables, code, inputs, assumptions, and a clean-run command reproduce the conclusions.
- The Section 14 PDF edits are made.


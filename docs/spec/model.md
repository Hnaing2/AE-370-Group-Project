# Spec 1/3 — Model: parameters and physics

Lean coding extract of the AE 370 README (Revision 4). Section numbers match the full README, which stays the authority for provenance, sources, PDF edits, and report text. Covers `src/parameters.py` and `src/physics.py`.

## Conventions (Day 1 decisions)

- Time first: `rhs(t, u, params, phase)` → shape (8,); `jacobian(t, u, params, phase)` → (8,8) central FD of rhs; `derived_quantities(u, params, phase)` → dict.
- `phase` ∈ {"powered", "coast"}, chosen by the driver per step interval, never inside a stage or residual.
- State u = [x, h, V, gamma, mp, ms, mu, a], float64 NumPy, SI units.
- `params` is a frozen dataclass with attribute access (`params.tau_v`), holding inputs, derived quantities at full precision, scenario, and provenance labels.
- Integrators never see params/phase: the driver passes `f = lambda t, u: rhs(t, u, params, phase)`; steps are `step_xx(f, t, u, dt, ...)`.
- `src/parameters.py` defines `REFERENCE_CASE` (this baseline exactly, never changes; used by `validation/test_reference_case.py`) and `STUDY_BASELINE` (starts equal; later changes are recorded in the frozen input file).
- Python ≥ 3.11; NumPy, SciPy (BE algebraic solve only), Matplotlib, pytest. No external time integrator anywhere (no solve_ivp, including for references).
- Tests: `validation/test_*.py` with pytest. Candidate dt given as integer N_cut = t_cut/dt; times computed as n·dt.
- Gamma is the flight-path angle; gamma_he the helium heat-capacity ratio. Isp and F_ref are engine-axis quantities; T_ax = F_eng cos(delta).

## 3. Parameters (condensed)

### 3.4 Input checklist

Every input must be present; the validator rejects a missing value rather than substituting zero.

| Input | Units | Baseline and basis |
|---|---|---|
| rho_o, rho_f, r | kg/m3, - | 1441.4, 879.8, 1.65 (Section 3.7). |
| H | m | 7500 m in the PDF's exponential form. Report the density difference from the 1976 US Standard Atmosphere over 19–40 km. |
| C_reg | kg sqrt(K)/(s Pa) | Sized per Section 5.5, ≈ 5.90e-6. Not a hardware value. |
| K_p | 1/Pa | kappa = K_p Pset ∈ {5, 10, 20, 50, 100}, nominal 20. Pressure errors in Pa before applying gain. |
| tau_v | s | {0.005, 0.01, 0.02, 0.05, 0.1}, nominal 0.02. |
| a0 | - | 0 for the regulated case (valve closed, Pu0 > Pset); a_fix for the fixed orifice. |
| x0 | m | 0 at the abort point, so x is displacement from the abort point. |
| g0 | m/s2 | 9.80665, used in gravity and in the Isp conversion. |
| P_cal | Pa | Pset (Section 5.1). |
| t_cut | s | 7.5, scheduled cutoff (Section 3.7). |
| Minimum engine pressure | Pa | Pu < 0.5 Pset is an assumed validity flag, not a sourced engine limit. Label later results as extrapolation. |
| t_eval | s | 15, after cutoff and before apogee. |
| t_end | s | Stop at the step after apogee; 200 s safety cap. |
| Tolerances | Pa, N, m | Section 11.1. |
| Passive comparator | scenario | Fixed orifice a_fix (Section 12). |

The regulator gain is parameterized as

$$
\kappa=K_pP_{\rm set},
$$

which is dimensionless and easier to discuss than a gain in inverse pascals.


### 3.6 Baseline (Revision 4, flattened)

The PDF draft dict is merged in; `Vtank` and the draft `ms0` are no longer inputs (Vtank is derived; ms0 is derived in "sized" mode).

~~~python
baseline = {
    # mission conditions (approximate IFA) and assumed mass
    "h0": 19000.0,            # m
    "V0": 536.0,              # m/s
    "gamma0": 1.10,           # rad, calibrated to ~40 km apogee
    "x0": 0.0,                # m
    "t_cut": 7.5,             # s, scheduled cutoff
    "m_dry": 9900.0,          # kg, assumed benchmark incl. trunk
    "mp0": 2562.0,            # kg
    # engines
    "N_engines": 8,
    "force_engine_ref": 71000.0,  # N per engine, engine axis
    "Isp": 235.0,             # s, engine axis
    "delta": 0.349,           # rad, cant
    "Cf": 1.5,                # only to report implied At; not used in the RHS
    "P_cal": 8.0e6,           # Pa, = Pset
    # propellant
    "rho_o": 1441.4,          # kg/m^3, NTO 20 C
    "rho_f": 879.8,           # kg/m^3, MMH 20 C
    "mixture_ratio": 1.65,    # O/F mass ratio
    # pressurization
    "Pset": 8.0e6,            # Pa
    "Ps0": 35.0e6,            # Pa
    "Pu0": 8.28e6,            # Pa
    "Ts0": 293.15,            # K
    "Tu": 293.15,             # K
    "R_he": 2077.1,           # J/(kg K)
    "gamma_he": 1.667,
    "mu0": 1.64,              # kg
    "supply_mode": "sized",   # or "specified_mass" (then "ms0" must be given)
    "Ps_f": 10.0e6,           # Pa, end-of-burn sizing margin
    # regulator
    "a_ref": 0.8,             # reference opening for C_reg sizing (not a0)
    "kappa": 20.0,            # K_p = kappa / Pset
    "tau_v": 0.02,            # s
    "scenario": "regulated",  # or "fixed_orifice"
    # aero / environment
    "Cd": 0.6,
    "A_ref": 12.56,           # m^2
    "rho0": 1.225,            # kg/m^3
    "H": 7500.0,              # m
    "R_E": 6371000.0,         # m
    "g0": 9.80665,            # m/s^2
    # run control
    "t_eval": 15.0,           # s
    "t_end": 200.0,           # s cap; stop at the step after apogee
}
gas_starved_case = {**baseline, "supply_mode": "specified_mass", "ms0": 5.75}  # OPTIONAL
~~~

### Derivation formulas (Sections 3.1, 3.2, 5.5, 12)

Compute in this order, at full precision, never mixing rounded display values with recomputed ones:

$$
\rho_p=\frac{1+r}{1/\rho_f+r/\rho_o},\qquad
V_{u0}=\frac{m_{u0}R_{\rm He}T_u}{P_{u0}},\qquad
V_{\rm tank}=m_{p0}/\rho_p+V_{u0},
$$
$$
K_F=\frac{N F_{\rm ref}}{P_{\rm cal}},\qquad q_{p,\rm ref}=\frac{K_FP_{\rm set}}{I_{\rm sp}g_0},\qquad m_{p,\rm burn}=q_{p,\rm ref}t_{\rm cut},
$$
$$
m_{s0}=\frac{P_{\rm set}\left(V_{u0}+m_{p,\rm burn}/\rho_p\right)/(R_{\rm He}T_u)-m_{u0}}{1-(P_{s,f}/P_{s0})^{1/\gamma_{\rm He}}}\ \text{(sized mode)},\qquad
V_s=\frac{m_{s0}R_{\rm He}T_{s0}}{P_{s0}},
$$
$$
q_{r,\rm req}=\frac{P_{\rm set}q_{p,\rm ref}}{\rho_pR_{\rm He}T_u},\qquad
T_{s,\rm ref}=T_{s0}(P_{s,f}/P_{s0})^{(\gamma_{\rm He}-1)/\gamma_{\rm He}}\approx178\ \mathrm{K},
$$
$$
C_{\rm reg}=\frac{q_{r,\rm req}\sqrt{T_{s,\rm ref}}}{a_{\rm ref}P_{s,f}\,\Phi(0.8)},\qquad
a_{\rm fix}=\frac{q_{r,\rm req}\sqrt{T_{s0}}}{C_{\rm reg}P_{s0}\,\Phi(P_{\rm set}/P_{s0})},\qquad
K_p=\kappa/P_{\rm set}.
$$

Phi is the flow function of Section 5.4. In specified-mass mode keep the given ms0, recompute Vs, and keep C_reg and m_dry unchanged (never resize the starved case automatically). a0 = 0 (regulated, valve closed) or a_fix (fixed orifice). M0 = m_dry + mp0 + ms0 + mu0.

The builder derives, at full precision and in this order:

| Derived quantity | Formula | Display value |
|---|---|---|
| rho_p | (1+r)/(1/rho_f + r/rho_o) | 1161.6 kg/m3 |
| Vu0 | mu0 R_he Tu / Pu0 | 0.12060 m3 |
| Vtank | mp0/rho_p + Vu0 | 2.3262 m3 |
| K_F | N F_ref / P_cal | 0.071 N/Pa |
| q_p,ref | K_F Pset/(Isp g0) | 246.47 kg/s |
| ms0 (sized mode) | Section 3.2 rule | 39.47 kg |
| Vs | ms0 R_he Ts0/Ps0 | 0.6866 m3 |
| q_r,req | Pset q_p,ref/(rho_p R_he Tu) | 2.788 kg/s |
| C_reg | Section 5.5 | 5.900e-6 kg sqrt(K)/(s Pa) |
| a_fix | Section 12 | 0.2311 |
| a0 | 0 (regulated) or a_fix (fixed orifice) | 0 or 0.2311 |
| Initial mass M0 | m_dry + mp0 + ms0 + mu0 | 12,503.1 kg (derived, not independently prescribed) |

In specified-mass mode, preserve the given ms0 and recompute Vs; keep C_reg and m_dry unchanged. Never resize the starved case automatically.


### 3.7 Provenance labels (for the frozen input file)

Each independent input carries one label. Full justification is in README §3.7 and §16.

| Label | Inputs |
|---|---|
| sourced | h0, V0, t_cut (IFA 7–8 s, midpoint), mp0, N_engines, force_engine_ref, A_ref, rho_o, rho_f, R_he, gamma_he, g0 |
| nominal (secondary source) | Isp |
| calibrated | gamma0 (to ~40 km apogee), P_cal = Pset (K_F) |
| assumed | m_dry (benchmark, not IFA mass), delta, mixture_ratio (OMS heritage), Pset, Ps0, Ts0, Tu, Pu0, mu0, Cd, rho0, H, R_E, x0 |
| sized | Ps_f, a_ref (→ ms0, Vs, C_reg, a_fix) |
| exploratory range | kappa {5, 10, 20, 50, 100}, tau_v {5, 10, 20, 50, 100} ms, Cd {0.4, 0.6, 0.8}, gamma0 {0.93, 1.10, 1.27} |
| study choice | t_eval, t_end, scenario, supply_mode |

Freeze rule: changing any baseline input requires rebuilding derived quantities, re-running the Section 3.8 check, and rechecking timestep acceptance.

### 3.8 Reference outputs for the implementation

**Reference outputs for the implementation (nominal regulated case, Section 3.6 baseline).** Modeled peak axial specific force is (T_ax − D)/M; report translational acceleration dV/dt separately if comparing definitions. Mach uses the 1976 US Standard Atmosphere temperature as post-processing only.

| Output | Regulated nominal | Fixed orifice |
|---|---|---|
| Apogee altitude | 40,057 m | 38,968 m |
| Time of apogee | 65.2 s | 63.4 s |
| Downrange at apogee | 21.1 km | 19.9 km |
| x(15 s), h(15 s) | 4,954 m, 27,501 m | 4,842 m, 27,298 m |
| V and h at cutoff | 738.9 m/s, 23,175 m | 713.4 m/s, 23,126 m |
| Propellant remaining at cutoff | 745.7 kg | 881.5 kg |
| Pu at cutoff (minimum powered Pu) | 7.767 MPa (2.9% droop) | 6.556 MPa (18% droop) |
| Valve opening at cutoff | 0.579 | 0.231 (fixed) |
| Peak modeled axial specific force, peak Mach | 3.85 g, 2.49 | 3.65 g, 2.40 |
| Relative helium drift (RK4) | ~1e-14 | ~1e-14 |

A correct implementation should match these to about 0.1% with RK4 at dt ≤ 1 ms. A larger difference means one of the two implementations differs from this specification; resolve it before any study. The prototype is a check, not a reference solution for timestep selection (Section 11.2).

## 5. Canonical constitutive model

Use SI units internally. Reserve gamma for flight-path angle and gamma_he for the gas heat-capacity ratio.

### 5.1 Calibrated force and propellant flow

$$
K_F=\frac{N F_{\mathrm{engine,ref}}}{P_{\mathrm{cal}}},\qquad
F_{\mathrm{eng}}=K_FP_u,\qquad
T_{\mathrm{ax}}=F_{\mathrm{eng}}\cos\delta,
\qquad q_p=\frac{F_{\mathrm{eng}}}{I_{\rm sp}g_0}.
$$

The PDF's expression is recovered with K_F = N Cf At/(1+k_inj). Calibrating K_F avoids pretending that the separately unknown At and k_inj are identified by one thrust datum. With P_cal = Pset, K_F = 8 × 71,000 / 8.0e6 = 0.071 N/Pa.

Reference checks at Pu = Pset:

- Summed engine force 568,000 N; axial force ≈ 533,758 N.
- Propellant consumption ≈ 246.47 kg/s.
- At constant Pu = Pset, ≈1,849 kg of the 2,562 kg would be consumed over 7.5 s. The reference runs retain ample propellant; this estimate alone is not a proof for all controller settings, so assert mp > 0 for every stage and accepted state.

### 5.2 Thermodynamic and environment quantities

$$
V_u=V_{\rm tank}-m_p/\rho_p,\quad
P_u=m_uR_{\rm He}T_u/V_u,\quad
M=m_{\rm dry}+m_p+m_s+m_u,
$$
$$
P_s=P_{s0}(m_s/m_{s0})^{\gamma_{\rm He}},\quad
T_s=T_{s0}(m_s/m_{s0})^{\gamma_{\rm He}-1},
$$
$$
\rho(h)=\rho_0e^{-h/H},\quad
g(h)=g_0\left(\frac{R_E}{R_E+h}\right)^2,\quad
D=\tfrac12\rho(h)V^2C_DA_{\rm ref}.
$$

Validate finite states and parameters, Vu > 0, mp > 0, ms > 0, mu > 0, M > 0, and V > 0 before evaluating singular expressions. Require 0 ≤ a ≤ 1, allowing only a documented small roundoff tolerance at the bounds; do not clamp a failed step into range. A failed check marks the run physical_model_invalid; it is never silently patched.

### 5.3 Regulator command

$$
a_{\rm cmd}=\operatorname{clip}(K_p(P_{\rm set}-P_u),0,1),\qquad
\dot a=(a_{\rm cmd}-a)/\tau_v.
$$

Proportional control needs a pressure error to hold the valve open, so a finite droop is expected (2.9% at cutoff in the nominal reference run).

### 5.4 Choked and subcritical regulator flow

For Ps > Pu define r = Pu/Ps and

$$
r_*=\left(\frac{2}{\gamma_{\rm He}+1}\right)^{\gamma_{\rm He}/(\gamma_{\rm He}-1)}\approx 0.4871.
$$

Keeping C_reg as the choked-flow coefficient, the quasi-steady ideal-gas orifice law is

$$
q_r=aC_{\rm reg}\frac{P_s}{\sqrt{T_s}}\Phi(r),
$$
$$
\Phi(r)=
\begin{cases}
1,&0\le r\le r_*,\\
\dfrac{\sqrt{\dfrac{2}{\gamma_{\rm He}-1}
\left(r^{2/\gamma_{\rm He}}-r^{(\gamma_{\rm He}+1)/\gamma_{\rm He}}\right)}}
{\left(2/(\gamma_{\rm He}+1)\right)^{(\gamma_{\rm He}+1)/(2(\gamma_{\rm He}-1))}},
&r_*<r<1,\\
0,&r\ge1.
\end{cases}
$$

The check valve prevents reverse flow. The initial ratio 8.28/35 = 0.237 is choked; at Pu ≈ Pset the flow unchokes once ms/ms0 falls to about 0.635, which happens late in the nominal burn. Record the actual ratio. Phi is continuous with a continuous slope at r_*, so no event is needed there.

Near r = 1 the slope of Phi grows without bound. The sizing target uses r_ref = 0.8, but the nominal simulation reaches only about r = 0.71 at cutoff; verify each sweep case rather than treating the sizing target as an observed trajectory limit. Only if a retained case causes an actual solver problem there may an optional linear ramp above r_lin be activated; state that it changes the model and compare r_lin = 0.98, 0.99, 0.995. Prefer deferring the optional gas-starved case over adding this machinery.

### 5.5 Sizing of C_reg

No hardware flow coefficient is available, so size an effective one. The ullage pressure obeys

$$
\dot P_u=\frac{R_{\rm He}T_u}{V_u}q_r
-\frac{P_u}{V_u}\frac{q_p}{\rho_p}.
$$

Holding Pu at a reference pressure requires

$$
q_{r,\rm req}=\frac{P_{u,\rm ref}q_{p,\rm ref}}
{\rho_p R_{\rm He}T_u},\qquad
C_{\rm reg}=
\frac{q_{r,\rm req}\sqrt{T_{s,\rm ref}}}
{a_{\rm ref}P_{s,\rm ref}\Phi(r_{\rm ref})}.
$$

Size at end-of-burn conditions, where the supply is weakest: Pu,ref = Pset, Ps,ref = Ps,f = 10 MPa, Ts,ref ≈ 178 K, r_ref = 0.8 (Phi ≈ 0.787), a_ref = 0.8. Then q_r,req ≈ 2.79 kg/s and C_reg ≈ 5.90e-6 kg sqrt(K)/(s Pa). At t = 0 the same demand needs a ≈ 0.231. This is a sizing target, not a prediction of the valve's operating point; the nominal reference run reaches a = 0.579 at cutoff.

## 6. RHS and phases

### 6.1 Powered phase (0 ≤ t < t_cut)

$$
\dot{\mathbf u}=
\begin{bmatrix}
V\cos\gamma\\
V\sin\gamma\\
(T_{\rm ax}-D)/M-g\sin\gamma\\
-g\cos\gamma/V\\
-q_p\\
-q_r\\
q_r\\
(a_{\rm cmd}-a)/\tau_v
\end{bmatrix}.
$$

In the fixed-orifice scenario a_dot = 0. Use one pure RHS function that every method calls.

### 6.2 Coast phase (t ≥ t_cut)

- F_eng = T_ax = q_p = 0 and q_r = 0.
- mp, ms, mu, and a stay at their cutoff values (a_dot = 0). Letting the valve close through a_dot = -a/tau_v would keep a fast, flight-irrelevant mode alive for the whole coast.
- Only x, h, V, and gamma change, under drag and gravity, until apogee.

The pressure subsystem does not depend on h, V, or gamma. Changing Cd or gamma0 alone must not change the pressure or propellant histories; this is a useful implementation check.


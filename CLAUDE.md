# CLAUDE.md

Pressure-fed launch abort IVP (AE 370 Group Project 1). Eight states, four self-written fixed-step integrators (FE, RK4, AB3, BE).

## Spec

The lean coding spec is in `docs/spec/`; section numbers match the full team README (Revision 4). Open only the file the current milestone needs, and search it by section number rather than reading it whole.

| Milestone | Work | Spec file |
|---|---|---|
| M1 | `src/parameters.py` | `model.md` |
| M2 | `src/physics.py` (derived quantities, rhs, FD Jacobian) | `model.md` |
| M3 | `step_fe`, `step_rk4` | `numerics.md` §8 |
| M4 | `simulate`, Hermite apogee, `compute_metrics` core; reproduce §3.8 | `numerics.md` §7–8, `model.md` §3.8 |
| M5 | AB3 (RK4 start/restart), BE | `numerics.md` §8.3–8.4 |
| M6 | Verification, Jacobian, stability | `verification_studies.md` §10, `numerics.md` §9 |
| M7 | Convergence/cost, timestep selection, sweeps, figures | `verification_studies.md` §11–12 |

Gate: no studies until `validation/test_reference_case.py` matches §3.8 within ~0.1%.

## Fixed conventions

- State `u = [x, h, V, gamma, mp, ms, mu, a]`, float64, SI units. Indices are the `IX..IA` constants in `src/physics.py`.
- Time first: `rhs(t, u, params, phase)`, `jacobian(t, u, params, phase)`, `derived_quantities(u, params, phase)`; `phase` is `"powered"` or `"coast"`.
- Integrators never see `params` or `phase`. The driver builds `f = lambda t, u: rhs(t, u, params, phase)` per step interval; steps are `step_xx(f, t, u, dt, ...)` and also run on scalar test problems.
- The phase is chosen per step by the driver, never inside an RK stage or BE residual. `t_cut / dt` is an integer; times are `n * dt`.
- `params` is the frozen `Params` dataclass from `build_params`; derived quantities at full precision. `REFERENCE_CASE` never changes; study changes go in `STUDY_BASELINE`.
- Never clip or patch states to rescue a step; invalid states raise/flag `physical_model_invalid`.
- No external time integrators (no `scipy.integrate`), including for references. SciPy is used only for the BE algebraic solve.

## Commands

Run from the repo root with the venv active (`.\.venv\Scripts\Activate.ps1`):

- Tests: `python -m pytest`
- Study/figure scripts: `python -m validation.<module>`, `python -m results.<module>`
- Everything: `python run_all.py`

## Git

Branches: `model`, `integrator`, `validation`, `results`, `report`; `main` gets reviewed work via PR. Commits and PRs show only the user as author: no Co-Authored-By or "Generated with Claude" lines.

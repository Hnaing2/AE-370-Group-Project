# AE-370-Group-Project

AE 370 (Aerospace Numerical Methods, Fall 2026) — Group Project 1: a numerical method for a dynamical system.

We simulate a rocket whose thrust comes from a pressure-fed, actively regulated propulsion system
(tank pressures, regulator valve and vehicle dynamics are integrated together as one initial value problem).
The time integrator is written from scratch; no `scipy.integrate` time marchers are used.

## Repository layout

```
src/          Model (right-hand side, parameters) and the from-scratch time integrator
validation/   Exact-solution checks, conservation checks, and convergence studies
results/      Scripts that run the parameter studies and produce the report figures
report/       LaTeX technical report; generated figures go in report/figures/
```

The split mirrors the grading rubric: the numerical method lives in `src/`, and everything that
*uses* it (convergence tests, results) lives in separate scripts.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Branches

`main` holds reviewed, working work only. Each piece of the project has its own branch, which is
merged into `main` through a pull request:

| Branch       | Rubric item                  | Scope                                                        |
|--------------|------------------------------|--------------------------------------------------------------|
| `model`      | 1. Premise                   | State equations, parameter table, initial conditions         |
| `integrator` | 2. Method                    | From-scratch time integrator                                 |
| `validation` | 3. Correct implementation    | Exact-solution comparisons, convergence and time-step study  |
| `results`    | 4. Results                   | Parameter sweeps and report figures                          |
| `report`     | All (write-up)               | LaTeX report, contributions, AI acknowledgment               |

Pull `main` into your branch often so the pieces stay compatible.

## Implementation notes

Choices made while coding that the spec leaves open; cite these in the report where relevant.

- **Pressure extrema use stored states only** (`compute_metrics`, M4). Min/max powered Pu is taken
  over the accepted step states, with no refinement between steps. Missing the true peak between
  steps costs about 10–200 Pa at dt = 0.5–2.5 ms, well under the 8 kPa (0.1% of Pset) tolerance.
  Revisit if a much coarser dt is ever selected.
- **`Pu_overshoot` keeps its sign** (`compute_metrics`, M4). It is max Pu − Pset after the first
  time Pu reaches Pset. Negative means Pu never climbed back above Pset: both reference runs give
  negative values (about −6 kPa regulated, −140 Pa fixed orifice). The 3.5% initial overpressure
  is reported separately as `Pu_initial_overpressure`.
- **BE uses a self-written Newton iteration, not `scipy.optimize.root`** (`step_be`, M5). Section
  8.4 allows either. MINPACK's `hybr` stops on its trust-region radius, not on the residual. When
  one Newton correction reaches round-off before that radius shrinks, `hybr` exits with "not
  making good progress", and the step counts as a failure. This happened on the scalar decay test
  u' = −u, and at dt = 0.156 ms in two regulator corners with the residual already at 1e-16. The
  default `xtol` instead lets some steps stop above the 1e-10 residual target. No single `xtol`
  avoided both. `hybr` also failed outright at the τ_v = 5 ms, κ = 100 corner at dt = 5 and
  2.5 ms. Newton passed all 36 checks: nominal, fixed orifice, and the four τ_v × κ corners, each
  at dt = 5 ms to 0.156 ms, powered phase. Its max scaled residual was 2.5e-13 and it used ~20
  RHS calls per step versus ~38 for `hybr`. `hybr` remains available as a cross-check through
  `be_solver_options={"method": "hybr", ...}`.
- **BE Newton settings** (`step_be`, M5). Unknowns and residuals are scaled by
  S = (h0, h0, V0, 1 rad, mp0, ms0, mu0, 1), the same floors as the finite-difference Jacobian. x
  uses h0 because x0 = 0. Each step starts from the current state and evaluates the
  `physics.jacobian` matrix there. It re-evaluates the Jacobian only when an iteration fails to
  reduce the residual. The step stops once the scaled correction is ≤ 1e-12, and is accepted
  only if the max scaled residual is ≤ 1e-10. A typical step costs one Jacobian (16 RHS calls)
  and 2–4 iterations; `nonlinear_iters` counts the final round-off-level correction too.
  Iterates are never clipped. An iterate that leaves the physical domain ends the run as
  `solver_failed`, not `physical_model_invalid`, because no accepted state was invalid.
- **BE helium drift is round-off, not accumulated residual** (M5). The ms and mu rows of the
  Jacobian are exact negatives, so each Newton correction preserves ms + mu. The measured drift
  is ~1e-14, far inside the 1e-8 limit. The residual bound alone would allow up to N × 1e-10.
- **AB3 history is per segment** (`simulate`, M5). Each segment stores f at its accepted states.
  The first two steps of each segment are RK4 steps that reuse the stored f as k1, so the restart
  at t_cut uses only coast-side derivatives. A run reaching coast costs n_steps + 14 RHS calls:
  each of the 2 segments has 1 initial f, and 2 RK4 steps that each need 3 calls beyond the
  stored f.

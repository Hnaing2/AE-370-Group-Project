# AE 370 Group Project: Pressure-Fed Launch Abort Simulation

AE 370 (Aerospace Numerical Methods, Fall 2026), Group Project 1: a numerical method for a dynamical system.
Team: Brendan Aoki, Eduardo Gutierrez, Drew Hisel, Htat Naing.

The code simulates an idealized, pressure-fed in-flight abort as an eight-state initial value problem that couples
the flight path (x, h, V, γ), the propellant and helium inventories, and a regulator valve. The studies ask how
the valve time constant τ_v and gain κ shape ullage pressure, impulse, downrange displacement and apogee; how the
regulated supply compares with an unregulated fixed orifice; and how sensitive the flight is to the drag
coefficient and the initial flight-path angle.

All four time integrators (forward Euler, classical RK4, third-order Adams–Bashforth, backward Euler) are written
from scratch; no `scipy.integrate` routine is used anywhere. SciPy is used only as an optional cross-check of the
backward-Euler algebraic solve. The selected method is **AB3 with Δt = 1.25 ms**, chosen by a convergence and cost
study against refined RK4 references and verified on all 34 parameter cases.

The written report is submitted separately; this repository generates its data and figures.

## Repository layout

```
src/
  parameters.py           Inputs with provenance labels, derived quantities, initial state
  physics.py              Constitutive model, right-hand side, finite-difference and analytic Jacobians
  integrators.py          Single steps of FE, RK4, AB3 and BE (Newton solve); stability functions
  simulation.py           Time-marching driver: powered/coast phases, AB3 restart at cutoff, apogee, metrics
validation/
  test_*.py               Tests: exact solutions, physical benchmarks, conservation, Jacobian, mission reference
  scalar_convergence.py   Convergence order of the steppers on scalar problems with exact solutions
  stability.py            Frozen-Jacobian eigenvalues and step limits along pilot trajectories
  convergence.py          Reference solutions, error targets, convergence and cost study
  timestep_selection.py   Selection of one method and time step; verification on the full parameter grid
  atmosphere_check.py     Exponential atmosphere versus the 1976 U.S. Standard Atmosphere
results/
  manifest.py             Input manifest and run environment
  sweeps.py               Parameter studies with the selected method and time step
  figures.py              Report figures from the saved data
  data/                   Generated data (JSON, CSV)
report/figures/           Generated figures (PNG and PDF)
run_all.py                Runs the complete workflow in order
```

The numerical method is in `src/integrators.py` and `src/simulation.py`. Everything that uses it (tests,
convergence studies, results) is in `validation/` and `results/`.

## Setup

Python 3.12 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate it (Windows PowerShell: `.\.venv\Scripts\Activate.ps1`; macOS/Linux: `source .venv/bin/activate`), then:

```bash
pip install -r requirements.txt
```

## Running

Run every command from the repository root.

```bash
python -m pytest
```

```bash
python run_all.py
```

`run_all.py` runs the tests, writes the input manifest, then runs scalar convergence, stability, the convergence
and cost study, timestep selection, the atmosphere check, the parameter sweeps, and the figures. Each stage refuses
missing, failed, or stale upstream outputs, and the workflow stops at the first failed check. It takes about
25 minutes with 11 worker processes (half the logical cores are used).

A single stage runs as `python -m validation.<module>` or `python -m results.<module>`, in the order above.
`python -m results.figures` redraws all figures from the saved data in a few seconds.

## Outputs

The data and figures in this repository were produced by `python run_all.py` at the commit recorded in
`results/data/environment.json`.

| File in `results/data/` | Written by | Contents |
|---|---|---|
| `inputs.json` | `results.manifest` | Every input with its unit and provenance label, derived quantities, swept grids |
| `environment.json` | `results.manifest` | Python and package versions, platform, git commit, worker count |
| `scalar_convergence.json` | `validation.scalar_convergence` | Exact-solution errors and observed orders |
| `stability.json` | `validation.stability` | Frozen-Jacobian eigenvalues and step limits |
| `convergence.json` | `validation.convergence` | References, trial errors, error scores and cost for every method and step |
| `timestep_selection.json` | `validation.timestep_selection` | Screening, selected method and step, full-grid verification |
| `atmosphere_check.json` | `validation.atmosphere_check` | Density and apogee comparison |
| `sweeps.json`, `*_grid.csv`, `regulated_vs_fixed_orifice.csv`, `*_histories.csv` | `results.sweeps` | Parameter-study metrics and histories |

| Figure in `report/figures/` | Shows |
|---|---|
| `fig1_regulator_heatmaps` | Minimum ullage pressure and apogee over τ_v × κ |
| `fig2_flight_vs_tau_v` | Downrange displacement at 15 s and apogee versus τ_v |
| `fig3_regulated_vs_fixed_orifice` | Regulated versus fixed-orifice histories and output differences |
| `fig4_error_vs_cost` | Error score versus RHS evaluations and versus time step, for each method |
| `fig5_cd_gamma0_sweep` | Downrange displacement and apogee versus drag coefficient for three initial flight-path angles |
| `fig6_pressure_transients` | Ullage pressure and valve opening over the first 0.5 s |
| `fig7_scalar_convergence` | Exact-solution convergence of the four steppers |

## Conventions

- State `u = [x, h, V, gamma, mp, ms, mu, a]`, float64, SI units.
- The burn is powered for 0 ≤ t < t_cut = 7.5 s, then coasts to apogee. Every time step divides t_cut, so no step
  straddles cutoff, and the phase is chosen per step by the driver.
- A state that leaves the physical domain ends the run as `physical_model_invalid`; states are never clipped.
- `REFERENCE_CASE` in `src/parameters.py` is the baseline; studies change inputs through `STUDY_BASELINE` and
  per-case overrides.

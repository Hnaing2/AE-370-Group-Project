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

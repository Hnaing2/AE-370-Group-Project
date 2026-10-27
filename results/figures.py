"""The report figures from saved results/data/ files: the regulator-grid, regulated-versus-fixed-orifice,
error-versus-cost and drag/flight-path-angle figures, the start-up pressure transients, and the scalar
exact-solution convergence.

Run: python -m results.figures (after results.sweeps)
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

FIGURE_DIR = "report/figures"
DATA_DIR = os.path.join("results", "data")

# Reference data-viz palette, light mode (print). Categorical slots in fixed order; one-hue blue ramp for magnitude.
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
METHOD_COLOR = {"fe": SERIES[0], "rk4": SERIES[1], "ab3": SERIES[2], "be": SERIES[3]}
METHOD_LABEL = {"fe": "FE", "rk4": "RK4", "ab3": "AB3", "be": "BE"}
FIG_WIDTH = 6.5  # in, report text width


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _load_json(name: str) -> dict[str, Any]:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run the study scripts first (python run_all.py)")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURE_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"wrote {FIGURE_DIR}/{name}.pdf/.png")


def _marker_kw(color: str) -> dict[str, Any]:
    """Filled end-marker with a surface-colored ring."""
    return {"marker": "o", "markersize": 4.5, "markerfacecolor": color, "markeredgecolor": SURFACE,
            "markeredgewidth": 1.0}


def _end_label(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", ha="left",
                fontsize=7.5, color=INK_2)


def _text_on(fill: tuple[float, float, float]) -> str:
    r, g, b = fill
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else SURFACE


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7, color=MUTED, wrap=True)


def fig_regulator_heatmaps() -> None:
    """1. tau_v x kappa heatmaps of minimum pressure and of apogee."""
    sweeps = _load_json("sweeps.json")
    rows = sweeps["regulator_grid"]
    taus, kappas = sweeps["grids"]["tau_v"], sweeps["grids"]["kappa"]
    nominal = (0.02, 20.0)
    grids = {}
    for key, scale in (("Pu_min", 1e-6), ("apogee_h", 1e-3)):
        Z = np.full((len(kappas), len(taus)), np.nan)
        for row in rows:
            i, j = kappas.index(row["inputs"]["kappa"]), taus.index(row["inputs"]["tau_v"])
            Z[i, j] = row["metrics"][key] * scale
        grids[key] = Z
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.9), constrained_layout=True)
    panels = (("Pu_min", "Minimum powered ullage pressure", "MPa", "{:.3f}"),
              ("apogee_h", "Apogee altitude", "km", "{:.2f}"))
    for ax, (key, title, unit, fmt) in zip(axes, panels):
        Z = grids[key]
        im = ax.imshow(Z, origin="lower", cmap=cmap, aspect="auto")
        ax.grid(False)
        lo, hi = np.nanmin(Z), np.nanmax(Z)
        for i in range(len(kappas)):
            for j in range(len(taus)):
                fill = to_rgb(cmap((Z[i, j] - lo) / (hi - lo) if hi > lo else 0.5))
                ax.text(j, i, fmt.format(Z[i, j]), ha="center", va="center", fontsize=6.5, color=_text_on(fill))
        j0, i0 = taus.index(nominal[0]), kappas.index(nominal[1])
        ax.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.2))
        ax.set_xticks(range(len(taus)), [f"{t * 1e3:g}" for t in taus])
        ax.set_yticks(range(len(kappas)), [f"{k:g}" for k in kappas])
        ax.set_xlabel(r"valve time constant $\tau_v$ [ms]")
        ax.set_ylabel(r"regulator gain $\kappa$")
        ax.set_title(f"{title} [{unit}]")
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, color=MUTED)
    _caption(fig, f"{sweeps['method'].upper()}, dt = {sweeps['dt'] * 1e3:g} ms; nominal case outlined "
                  f"(Cd = 0.6, gamma0 = 1.10 rad). Pset = 8 MPa.")
    _save(fig, "fig1_regulator_heatmaps")


def fig_apogee_vs_tau_v() -> None:
    """2. Downrange displacement or apogee versus tau_v for selected gains."""
    sweeps = _load_json("sweeps.json")
    rows = sweeps["regulator_grid"]
    gains = (5.0, 20.0, 100.0)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.6), constrained_layout=True)
    panels = (("x_eval", f"Downrange displacement x(t_eval = 15 s) [m]", 1.0),
              ("apogee_h", "Apogee altitude [km]", 1e-3))
    for ax, (key, title, scale) in zip(axes, panels):
        for color, kappa in zip(SERIES, gains):
            sel = sorted((r for r in rows if r["inputs"]["kappa"] == kappa), key=lambda r: r["inputs"]["tau_v"])
            tau = [r["inputs"]["tau_v"] * 1e3 for r in sel]
            y = [r["metrics"][key] * scale for r in sel]
            ax.plot(tau, y, color=color, label=rf"$\kappa$ = {kappa:g}", **_marker_kw(color))
            _end_label(ax, tau[-1], y[-1], rf"$\kappa$ = {kappa:g}")
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
        ax.minorticks_off()
        ax.set_xlim(4, 190)
        ax.set_xlabel(r"valve time constant $\tau_v$ [ms]")
        ax.set_title(title)
    axes[0].legend(loc="best")
    _caption(fig, f"{sweeps['method'].upper()}, dt = {sweeps['dt'] * 1e3:g} ms; nominal Cd and gamma0. "
                  "x is displacement from the abort point, not clearance from the booster.")
    _save(fig, "fig2_flight_vs_tau_v")


def _read_histories(name: str = "powered_histories.csv") -> dict[str, np.ndarray]:
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        data = np.array([[float(v) for v in row] for row in reader])
    return {name: data[:, i] for i, name in enumerate(header)}


def fig_regulated_vs_fixed_orifice() -> None:
    """3. Regulated versus fixed-orifice metrics."""
    sweeps = _load_json("sweeps.json")
    comp = sweeps["regulated_vs_fixed_orifice"]
    hist = _read_histories()
    pset = sweeps["regulator_grid"][0]["inputs"]["Pset"]

    fig = plt.figure(figsize=(FIG_WIDTH, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    ax_p, ax_a, ax_bar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    cases = (("regulated", "Regulated (nominal)", SERIES[0]), ("fixed_orifice", "Fixed orifice", SERIES[1]))
    for tag, label, color in cases:
        ax_p.plot(hist["t"], hist[f"Pu_{tag}"] * 1e-6, color=color, label=label)
        ax_a.plot(hist["t"], hist[f"a_{tag}"], color=color, label=label)
    ax_p.axhline(pset * 1e-6, color=MUTED, linewidth=0.8)
    ax_p.annotate("Pset", (hist["t"][-1], pset * 1e-6), xytext=(2, 2), textcoords="offset points",
                  ha="right", va="bottom", fontsize=7, color=MUTED)
    ax_p.set_title("Ullage pressure [MPa]")
    ax_a.set_title("Valve opening a [-]")
    for ax in (ax_p, ax_a):
        ax.set_xlabel("time [s]")
        ax.set_xlim(0, hist["t"][-1])
    ax_a.set_ylim(0, 1)
    ax_p.legend(loc="lower left")

    metrics = (("I_ax", "Total axial impulse"), ("mp_used", "Propellant used"), ("Pu_min", "Minimum Pu"),
               ("V_cut", "Speed at cutoff"), ("x_eval", "x(t_eval)"), ("h_eval", "h(t_eval) gain"),
               ("apogee_h", "Apogee altitude"), ("apogee_x", "Downrange at apogee"),
               ("peak_specific_force_g", "Peak axial specific force"))
    h0 = sweeps["regulator_grid"][0]["inputs"]["h0"]
    rel = []
    for key, _ in metrics:
        reg, fix = comp[key]["regulated"], comp[key]["fixed_orifice"]
        if key == "h_eval":
            reg, fix = reg - h0, fix - h0
        rel.append(100.0 * (fix - reg) / reg)
    y = np.arange(len(metrics))[::-1]
    ax_bar.barh(y, rel, height=0.55, color=SERIES[1])
    ax_bar.axvline(0.0, color=AXIS, linewidth=0.8)
    for yi, r in zip(y, rel):
        ax_bar.annotate(f"{r:+.1f}%", (r, yi), xytext=(-3 if r < 0 else 3, 0), textcoords="offset points",
                        ha="right" if r < 0 else "left", va="center", fontsize=7, color=INK_2)
    ax_bar.set_yticks(y, [label for _, label in metrics])
    ax_bar.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax_bar.grid(axis="y", visible=False)
    lo = min(rel)
    ax_bar.set_xlim(lo * 1.25 if lo < 0 else -1, max(max(rel) * 1.25, 1))
    ax_bar.set_xlabel("fixed orifice relative to regulated [%]")
    ax_bar.set_title("Fixed orifice versus regulated: same hardware, helium, and initial wet mass")
    _caption(fig, f"{sweeps['method'].upper()}, dt = {sweeps['dt'] * 1e3:g} ms. Fixed opening a_fix ~ 0.231 "
                  "matches nominal demand at the initial supply state; the regulated valve starts closed.")
    _save(fig, "fig3_regulated_vs_fixed_orifice")


def fig_error_vs_cost() -> None:
    """4. Error versus measured cost, showing why the method/dt was chosen."""
    conv = _load_json("convergence.json")
    sel = _load_json("timestep_selection.json")
    candidates = sel["screening"]["candidates"]
    chosen = sel["selected"]
    e_max = sel["e_score_max"]
    finite = [c["e_score"] for c in candidates if c["e_score"] is not None]
    ceiling = 10 ** math.ceil(math.log10(max(finite) * 3))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.9), constrained_layout=True)
    for ax, xkey in zip(axes, ("rhs_calls", "dt")):
        for method in conv["methods"]:
            cs = sorted((c for c in candidates if c["method"] == method), key=lambda c: c["dt"])
            color = METHOD_COLOR[method]
            x = np.array([c[xkey] * (1e3 if xkey == "dt" else 1.0) for c in cs])
            e = np.array([math.nan if c["e_score"] is None else c["e_score"] for c in cs])
            ok = np.isfinite(e)
            ax.plot(x[ok], e[ok], color=color, label=METHOD_LABEL[method], **_marker_kw(color))
            ax.plot(x[~ok], np.full((~ok).sum(), ceiling), linestyle="none", marker="x", markersize=5,
                    markeredgewidth=1.5, color=color)
            margin = np.array([not c["checks"]["stability_margin"] for c in cs]) & ok
            ax.plot(x[margin], e[margin], linestyle="none", marker="o", markersize=8, markerfacecolor="none",
                    markeredgecolor=MUTED, markeredgewidth=0.8)
        screening = next(c for c in candidates if (c["method"], c["n_cut"]) == (chosen["method"], chosen["n_cut"]))
        cx = screening[xkey] * (1e3 if xkey == "dt" else 1.0)
        ax.plot([cx], [screening["e_score"]], linestyle="none", marker="o", markersize=11, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=1.2)
        # Label clear of the method's own line, which falls steeply to the right in cost and to the left in dt.
        by_cost = xkey == "rhs_calls"
        ax.annotate(f"selected: {METHOD_LABEL[chosen['method']]}, {chosen['dt'] * 1e3:g} ms",
                    (cx, screening["e_score"]), xytext=(10, 0) if by_cost else (8, -14), textcoords="offset points",
                    va="center" if by_cost else "baseline", fontsize=7, color=INK)
        ax.axhline(e_max, color=MUTED, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(top=ceiling * 2)
        ax.set_ylabel("E_score (max error / tolerance)")
    axes[0].set_xlabel("RHS evaluations, summed over the screening cases")
    axes[0].set_title("Error versus cost")
    axes[1].set_xlabel("dt [ms]")
    axes[1].set_title("Error versus step")
    axes[0].legend(loc="upper right", ncols=2)
    _caption(fig, f"Screening cases: {', '.join(sel['screening']['cases'])}. Gray line: acceptance limit "
                  f"E_score = {e_max:g}. x at the top: a run left the physical domain. Gray ring: dt beyond "
                  "0.8 x the frozen-Jacobian stability limit.")
    _save(fig, "fig4_error_vs_cost")


def fig_cd_gamma0_sweep() -> None:
    """5. x(t_eval) and apogee versus Cd, one line per gamma0."""
    sweeps = _load_json("sweeps.json")
    rows = sweeps["cd_gamma0_grid"]
    gammas = sweeps["grids"]["gamma0"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.6), constrained_layout=True)
    panels = (("x_eval", "Downrange displacement x(t_eval = 15 s) [km]", 1e-3),
              ("apogee_h", "Apogee altitude [km]", 1e-3))
    for ax, (key, title, scale) in zip(axes, panels):
        for color, gamma0 in zip(SERIES, gammas):
            sel = sorted((r for r in rows if r["inputs"]["gamma0"] == gamma0), key=lambda r: r["inputs"]["Cd"])
            cd = [r["inputs"]["Cd"] for r in sel]
            y = [r["metrics"][key] * scale for r in sel]
            label = rf"$\gamma_0$ = {gamma0:.2f} rad ({math.degrees(gamma0):.0f}°)"
            ax.plot(cd, y, color=color, label=label, **_marker_kw(color))
            _end_label(ax, cd[-1], y[-1], f"{math.degrees(gamma0):.0f}°")
        ax.set_xticks(sweeps["grids"]["Cd"])
        ax.set_xlim(0.37, 0.9)
        ax.set_xlabel("drag coefficient Cd")
        ax.set_title(title)
    axes[1].legend(loc="best")
    _caption(fig, f"{sweeps['method'].upper()}, dt = {sweeps['dt'] * 1e3:g} ms; nominal regulator. "
                  "Pressure and propellant histories are identical across the 9 runs.")
    _save(fig, "fig5_cd_gamma0_sweep")


def fig_pressure_transients() -> None:
    """6. Start-up Pu and valve opening for every tau_v at the nominal and the highest gain: the lag mechanism
    behind the first-crossing dip and the later overshoot."""
    sweeps = _load_json("sweeps.json")
    hist = _read_histories("transient_histories.csv")
    pset = sweeps["regulator_grid"][0]["inputs"]["Pset"]
    taus = sweeps["grids"]["tau_v"]
    gains = sorted({float(k.rsplit("kappa", 1)[1]) for k in hist if k.startswith("Pu_")})
    colors = BLUE_RAMP[2:2 + len(taus)]  # light (fast valve) to dark (slow valve)
    fig, axes = plt.subplots(2, len(gains), figsize=(FIG_WIDTH, 4.2), sharex=True, sharey="row",
                             constrained_layout=True)
    for col, kappa in enumerate(gains):
        ax_p, ax_a = axes[0, col], axes[1, col]
        for color, tau in zip(colors, taus):
            name = f"tau{round(tau * 1e3, 6):g}_kappa{kappa:g}"
            label = rf"$\tau_v$ = {tau * 1e3:g} ms"
            ax_p.plot(hist["t"], hist[f"Pu_{name}"] * 1e-6, color=color, linewidth=1.2, label=label)
            ax_a.plot(hist["t"], hist[f"a_{name}"], color=color, linewidth=1.2, label=label)
        ax_p.axhline(pset * 1e-6, color=MUTED, linewidth=0.8, linestyle="--")
        ax_p.set_title(rf"Ullage pressure [MPa], $\kappa$ = {kappa:g}")
        ax_a.set_title(rf"Valve opening a [-], $\kappa$ = {kappa:g}")
        ax_a.set_xlabel("time [s]")
        ax_a.set_xlim(0, hist["t"][-1])
        ax_a.set_ylim(0, 1)
    axes[0, 0].annotate("Pset", (hist["t"][-1], pset * 1e-6), xytext=(-2, 2), textcoords="offset points",
                        ha="right", va="bottom", fontsize=7, color=MUTED)
    axes[1, 0].legend(loc="upper right", ncols=2)
    _caption(fig, f"{sweeps['method'].upper()}, dt = {sweeps['dt'] * 1e3:g} ms, stored states; nominal Cd and gamma0. "
                  "The valve starts closed, so Pu falls from the 3.5% initial overpressure through Pset before the "
                  "valve opens.")
    _save(fig, "fig6_pressure_transients")


def fig_scalar_convergence() -> None:
    """7. Global error at t = 1 against dt for the four production steppers on the exact scalar problems."""
    data = _load_json("scalar_convergence.json")
    problems = data["problems"]
    fig, axes = plt.subplots(1, len(problems), figsize=(FIG_WIDTH, 2.8), sharey=True, constrained_layout=True)
    for ax, (name, problem) in zip(axes, problems.items()):
        for method, table in problem["methods"].items():
            color = METHOD_COLOR[method]
            label = f"{METHOD_LABEL[method]} (p = {table['observed_order'][-1]:.2f})"
            ax.plot(table["dt"], table["error"], color=color, label=label, **_marker_kw(color))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("dt")
        ax.set_title(problem["equation"])
        ax.legend(loc="lower right")
    axes[0].set_ylabel(f"|u_n - u(t)| at t = {data['t_end']:g}")
    _caption(fig, "p: observed order log2(E(dt)/E(dt/2)) at the finest pair; theory FE 1, BE 1, AB3 3, RK4 4. AB3 "
                  "takes two RK4 start-up steps. FE and BE errors have nearly equal size and opposite sign, so their "
                  "lines overlap.")
    _save(fig, "fig7_scalar_convergence")


def main() -> None:
    _style()
    fig_regulator_heatmaps()
    fig_apogee_vs_tau_v()
    fig_regulated_vs_fixed_orifice()
    fig_error_vs_cost()
    fig_cd_gamma0_sweep()
    fig_pressure_transients()
    fig_scalar_convergence()


if __name__ == "__main__":
    main()

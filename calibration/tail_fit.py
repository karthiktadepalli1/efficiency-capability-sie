"""Tail index of the task-length distribution.

The capability leg needs k in  1 - alpha = Pr(task length > h) ~ h^(-k), where the
relevant measure of tasks is work-hours: a task that takes a human T hours occupies T
units of research labor. For a length distribution with density f, the hours measure
has density proportional to x f(x), and a Pareto tail of index k in that measure is
estimated by the Pareto maximum-likelihood (Hill) estimator with observations weighted
by their length:

    1 / k_hat = sum_{x_i > T} x_i ln(x_i / T) / sum_{x_i > T} x_i .

The count-weighted Hill estimator and a log-log slope of the survival function are
reported alongside. Writes drafts/tables/tail_fits.tex.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE.parent / "drafts" if (HERE.parent / "drafts").is_dir() else HERE.parent / "output"
TABLES = OUTPUT / "tables"


def _read(path: Path, col: str) -> np.ndarray:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r[col]) for r in rows if r[col] not in ("", "nan")])
    return x[x > 0]


def th11_baselined() -> np.ndarray:
    return _read(DATA / "metr_th11_baselined_tasks.csv", "human_minutes")


def rct_early_actual() -> np.ndarray:
    return _read(DATA / "metr_early2025_rct_issues.csv", "initial_implementation_time")


def rct_early_forecast() -> np.ndarray:
    return _read(DATA / "metr_early2025_rct_issues.csv", "predicted_time_no_ai")


def hill_count(x: np.ndarray, T: float) -> float:
    t = x[x > T]
    return 1.0 / np.mean(np.log(t / T))


def hill_time(x: np.ndarray, T: float) -> float:
    t = x[x > T]
    return 1.0 / np.average(np.log(t / T), weights=t)


def loglog_time(x: np.ndarray, T: float) -> float:
    grid = np.geomspace(T, x.max() * 0.999, 12)
    S = np.array([x[x > g].sum() / x.sum() for g in grid])
    ok = S > 0
    return -np.polyfit(np.log(grid[ok]), np.log(S[ok]), 1)[0]


THRESHOLDS_H = (1.0, 2.0, 4.0)


def fits(x: np.ndarray) -> list[tuple[float, int, float, float, float]]:
    out = []
    for h in THRESHOLDS_H:
        T = 60.0 * h
        n = int((x > T).sum())
        out.append((h, n, hill_count(x, T), hill_time(x, T), loglog_time(x, T)))
    return out


def baseline_k() -> float:
    """Length-weighted Pareto MLE above four hours on the baselined TH1.1 tasks."""
    return float(hill_time(th11_baselined(), 240.0))


def baseline_k_loglog() -> float:
    """Log-log slope of the hours survival function above four hours on the baselined TH1.1 tasks."""
    return float(loglog_time(th11_baselined(), 240.0))


def thin_k() -> float:
    """Same estimator with a one-hour threshold."""
    return float(hill_time(th11_baselined(), 60.0))


def rct_k() -> float:
    return float(hill_time(rct_early_actual(), 240.0))


def gdpval_k() -> float:
    """Count-weighted fit between the 75th percentile (8 h) and the maximum (605 h) of
    GDPval's published expert completion times, n = 1320."""
    return math.log(0.25 * 1320) / math.log(605 / 8)


def write_table() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, x in [("TH1.1 baselined tasks", th11_baselined()),
                    ("Early-2025 RCT issues", rct_early_actual())]:
        first = True
        for h, n, kc, kt, kl in fits(x):
            label = f"{name} ($n = {len(x)}$)" if first else ""
            first = False
            rows.append(f"{label} & {h:g} h & {n} & {kc:.2f} & {kt:.2f} & {kl:.2f} \\\\")
    rows.append(f"GDPval quantiles ($n = 1320$) & 8 h & 330 & {gdpval_k():.2f} & --- & --- \\\\")
    body = "\n".join(rows)
    (TABLES / "tail_fits.tex").write_text(
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        " & & & \\multicolumn{1}{c}{count-weighted} & \\multicolumn{2}{c}{length-weighted} \\\\\n"
        "\\cmidrule(lr){4-4}\\cmidrule(lr){5-6}\n"
        "Data & Threshold & Tasks above & Hill & Hill & log-log \\\\\n\\midrule\n"
        f"{body}\n" + "\\bottomrule\n\\end{tabular}\n"
    )


def summary_table() -> None:
    """Descriptive statistics of the task-length data, in hours."""
    TABLES.mkdir(parents=True, exist_ok=True)

    def row(name, x):
        h = x / 60.0
        q = np.quantile(h, [0.25, 0.5, 0.75, 0.9])
        sh4 = h[h > 4].sum() / h.sum()
        sh8 = h[h > 8].sum() / h.sum()
        return (f"{name} & {len(h)} & {q[0]:.1f} & {q[1]:.1f} & {q[2]:.1f} & {q[3]:.1f} & {h.max():.1f} "
                f"& {sh4:.2f} & {sh8:.2f} \\\\")

    rows = [
        row("TH1.1 baselined tasks", th11_baselined()),
        row("Early-2025 RCT, actual", rct_early_actual()),
        row("Early-2025 RCT, forecast", rct_early_forecast()),
        "GDPval, full set (published) & 1320 & 2.0 & 4.0 & 8.0 & --- & 605.0 & --- & --- \\\\",
    ]
    body = "\n".join(rows)
    (TABLES / "task_length_summary.tex").write_text(
        "\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
        " & & \\multicolumn{5}{c}{task length, hours} & \\multicolumn{2}{c}{share of hours in tasks} \\\\\n"
        "\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}\n"
        "Data & $n$ & p25 & p50 & p75 & p90 & max & $>4$ h & $>8$ h \\\\\n\\midrule\n"
        f"{body}\n" + "\\bottomrule\n\\end{tabular}\n"
    )


def plot_survival(threshold_h: float = 4.0) -> None:
    """Log-log survival of task length: share of hours (solid) and share of tasks (dashed) in
    tasks longer than T, with the fitted Pareto tail above the threshold."""
    import ktviz
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    from style import save_panels_pdf

    FIG = OUTPUT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    fig.subplots_adjust(wspace=0.55)  # room for the direct labels at the left panel's line ends
    panels = [("METR Time Horizon 1.1, baselined tasks", th11_baselined()),
              ("METR early-2025 developer study, actual issue times", rct_early_actual())]
    for ax, (_, x) in zip(axes, panels):
        h = np.sort(x) / 60.0
        grid = np.geomspace(h.min(), h.max(), 400)
        S_count = np.array([(h > g).mean() for g in grid])
        S_time = np.array([h[h > g].sum() / h.sum() for g in grid])
        T = threshold_h
        k = hill_time(x, 60.0 * T)
        S_T = h[h > T].sum() / h.sum()
        gg = np.geomspace(T, h.max(), 50)
        keep = S_count > 0  # survival is zero past the longest task, off a log axis
        ax.plot(grid[keep], S_time[keep], color=ktviz.INK, lw=1.6, label="Share of hours")
        ax.plot(grid[keep], S_count[keep], color=ktviz.CONTEXT, lw=1.3, ls="--", label="Share of tasks")
        ax.plot(gg, S_T * (gg / T) ** (-k), color=ktviz.ACCENT, lw=2.2, label=f"Pareto fit, k = {k:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.005, 1.1)
        ax.set_xlabel("Task length T (human hours)")
        ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ktviz.label_lines(ax)
    axes[0].set_ylabel("Share longer than T")
    save_panels_pdf(fig, axes, FIG / "tail_survival", [name for name, _ in panels])
    plt.close(fig)


if __name__ == "__main__":
    write_table()
    summary_table()
    plot_survival()
    print(f"baseline k (TH1.1 baselined, length-weighted, >4h) = {baseline_k():.3f}")
    print(f"thin k (same, >1h) = {thin_k():.3f};  RCT >4h = {rct_k():.3f};  GDPval = {gdpval_k():.3f}")
    for name, x in [("TH1.1", th11_baselined()), ("RCT", rct_early_actual())]:
        for h, n, kc, kt, kl in fits(x):
            print(f"  {name} >{h:g}h n={n:3d} count-Hill={kc:.2f} time-Hill={kt:.2f} time-loglog={kl:.2f}")

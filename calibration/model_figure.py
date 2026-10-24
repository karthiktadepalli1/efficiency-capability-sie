"""Figure 1: research capacity (Proposition 1) over efficiency and capability.

Research capacity relative to the human researchers working alone, R*/L, as a function
of effective compute per researcher X/L ("AI agents per human researcher") and the
automated task share alpha, at the baseline task elasticity of substitution. It
illustrates the model rather than the data; the explainer reuses it with a headline.
"""
from __future__ import annotations

from pathlib import Path

import ktviz
import matplotlib.pyplot as plt
import numpy as np

import style  # noqa: F401  (applies the paper figure style)


def capacity(x, a, sigma):
    """Research capacity R*/L (Proposition 1) at agents per human x and automated share a; arrays allowed."""
    x, a = np.broadcast_arrays(np.asarray(x, float), np.asarray(a, float))
    s = sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        ces = (a ** (1 / s) * x ** ((s - 1) / s) + (1 - a) ** (1 / s)) ** (s / (s - 1))
    return np.where(a >= x / (1 + x), 1 + x, ces)


def plot_output_map(stem: Path, sigma: float, title: str | None = None, formats=("pdf",)) -> None:
    """Shaded research capacity over agents per human and the automated share.

    x is log10(agents per human) on a linear axis; y is the automated share. The colour scale is
    log10 of research output relative to the humans working alone.
    """
    xs = np.linspace(0, np.log10(3000), 700)
    ys = np.linspace(0, 1, 700)
    XX, YY = np.meshgrid(xs, ys)
    R = capacity(10 ** XX, YY, sigma)
    fig, ax = ktviz.figure((7.0, 5.2))
    cf = ax.contourf(XX, YY, np.log10(R), levels=np.linspace(0, 3.6, 73), cmap=ktviz.SEQUENTIAL, extend="max",
                     zorder=0)
    if "pdf" in formats:  # PDF viewers draw hairline seams between filled contour bands
        cf.set_rasterized(True)
    lab = dict(fontsize=11, fontweight=600, ha="center", va="center", zorder=3)
    ax.text(np.log10(3), 0.85, "Slow:\ntoo few agents", color=ktviz.INK, **lab)
    ax.text(np.log10(200), 0.12, "Slow: too little automation", color=ktviz.INK, **lab)
    ax.text(np.log10(900), 0.94, "Fast", color=ktviz.BG, **lab)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, 1)
    xt = [1, 10, 100, 1000]
    ax.set_xticks(np.log10(xt), [f"{v:,}×" for v in xt])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("AI agents per human researcher (efficiency)")
    ax.set_ylabel("Share of research tasks automated (capability)")
    # Colour scale, as research output relative to the humans working alone.
    cax = ax.inset_axes([1.05, 0, 0.035, 1])
    cb = fig.colorbar(cf, cax=cax, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["1×", "10×", "100×", "1,000×"])
    cb.ax.tick_params(length=0, labelsize=ktviz.LABEL_SIZE, labelcolor=ktviz.MUTED)
    cb.outline.set_visible(False)
    cax.text(0, 1.09, "Research output,\nvs. humans alone", transform=cax.transAxes, ha="left", va="bottom",
             fontsize=ktviz.LABEL_SIZE, color=ktviz.MUTED, linespacing=1.2)
    ktviz.save(fig, stem, title=title, formats=formats)
    plt.close(fig)

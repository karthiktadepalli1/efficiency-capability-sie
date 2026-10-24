"""Shared figure style for the paper: ktviz on a white background, exported as untitled PDFs.

Paper figures carry no in-figure title (the LaTeX caption does that job), so the
framing below is ktviz's without the headline.
"""
from __future__ import annotations

from pathlib import Path

import ktviz
import matplotlib.pyplot as plt

ktviz.use()
BG = "#ffffff"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
ktviz.BG = BG  # also used for label halos


def save_pdf(fig, stem: Path) -> None:
    """One-axes figure: ktviz framing (horizontal y-label above the axis), PDF only."""
    ktviz.save(fig, stem, formats=("pdf",))


def save_panels_pdf(fig, axes, stem: Path, headings: list[str]) -> None:
    """Multi-panel figure: a heading and the horizontal y-label above each panel, PDF only."""
    for ax, heading in zip(axes, headings):
        if ax.get_yscale() == "log" or ax.get_xscale() == "log":
            ax.grid(True, which="major", color=ktviz.LOG_GRID, lw=0.7, zorder=0)
        label = ax.get_ylabel()
        ax.set_ylabel("")
        ax.annotate(heading, (0, 1), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points",
                    ha="left", va="bottom", fontsize=ktviz.LABEL_SIZE + 1, fontweight=600, color=ktviz.INK)
        if label:
            ax.annotate(label, (0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                        ha="left", va="bottom", fontsize=plt.rcParams["axes.labelsize"], color=ktviz.MUTED)
    fig.savefig(Path(stem).with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)

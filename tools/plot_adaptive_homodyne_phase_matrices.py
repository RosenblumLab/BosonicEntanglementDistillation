#!/usr/bin/env python
"""Plot adaptive-homodyne phase matrices for several Fock dimensions.

Run from the repository root:

    python tools/plot_adaptive_homodyne_phase_matrices.py
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from other_measurements import adaptive_homodyne_phase_matrix


DIMENSIONS = (10, 20, 40, 80)
OUTPUT_STEM = ROOT / "plots" / "adaptive_homodyne_phase_matrix_heatmaps"
FONT_SIZE = 12


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
        }
    )

    phase_matrices = {
        dimension: adaptive_homodyne_phase_matrix(dimension)
        for dimension in DIMENSIONS
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 6.2),
        constrained_layout=True,
    )
    image = None
    for axis, dimension in zip(axes.flat, DIMENSIONS):
        image = axis.imshow(
            phase_matrices[dimension],
            origin="lower",
            interpolation="nearest",
            extent=(-0.5, dimension - 0.5, -0.5, dimension - 0.5),
            vmin=0,
            vmax=1,
            cmap="viridis",
            rasterized=True,
        )
        ticks = [0, dimension // 2, dimension - 1]
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.set_xlabel(r"$m$")
        axis.set_ylabel(r"$n$")
        axis.set_title(rf"$N={dimension}$")
        axis.set_aspect("equal")

    colorbar = fig.colorbar(image, ax=axes, pad=0.03, shrink=0.94)
    colorbar.set_label(r"$H_{nm}^{\mathrm{ahd}}$")

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

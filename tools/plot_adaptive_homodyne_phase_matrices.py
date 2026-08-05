#!/usr/bin/env python
"""Plot adaptive-homodyne phase matrices for several Fock dimensions.

Run from the repository root:

    python tools/plot_adaptive_homodyne_phase_matrices.py
"""

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from other_measurements import adaptive_homodyne_phase_matrix


DIMENSIONS = (10, 20, 40, 80, 220)
OUTPUT_STEM = ROOT / "plots" / "adaptive_homodyne_phase_matrix_heatmaps"
HIGH_PRECISION_CACHE = (
    ROOT / "results" / "adaptive_homodyne_phase_matrix_N=220.npz"
)
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

    phase_matrices = {}
    for dimension in DIMENSIONS:
        print(f"Computing adaptive-homodyne phase matrix for N={dimension}...")
        options = {}
        if dimension == 220:
            options = {
                "cache_path": HIGH_PRECISION_CACHE,
                "calculation_method": "mpmath",
                "mp_dps": 50,
                "mp_workers": min(64, os.cpu_count() or 1),
            }
        phase_matrices[dimension] = adaptive_homodyne_phase_matrix(
            dimension, **options
        )

    fig = plt.figure(figsize=(10.2, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        7,
        width_ratios=(1, 1, 1, 1, 1, 1, 0.08),
    )
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    colorbar_axis = fig.add_subplot(grid[:, 6])
    image = None
    for axis, dimension in zip(axes, DIMENSIONS):
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

    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"$H_{nm}^{\mathrm{ahd}}$")

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

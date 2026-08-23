"""Dark-frame diagnostic plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def darkplot(dark, mask, darkfile, *, badmask=0):
    """Create histogram, ratio, cumulative, and image diagnostics."""
    cube = np.asarray(dark, dtype=float)
    mask = np.asarray(mask)
    if cube.ndim != 3 or mask.shape != cube.shape[:2]:
        raise ValueError("dark must be (ny,nx,nread) and mask must be (ny,nx)")
    nread = cube.shape[-1]
    middle, final = (nread - 1) // 2, nread - 1
    differences = [cube[..., middle] - cube[..., 1],
                   cube[..., final] - cube[..., 1]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for values in differences:
        values = values[np.isfinite(values)]
        axes[0, 0].hist(values, bins=200, range=(0, 100), histtype="step")
        positive = values[values > 0]
        axes[1, 0].hist(positive, bins=np.logspace(-1, 4, 150),
                        histtype="step")
    axes[0, 0].set(xlabel="total dark", ylabel="N pixels")
    axes[1, 0].set(xscale="log", yscale="log", xlabel="dark rate",
                   ylabel="N pixels")

    d1, d2 = differences[1], differences[0]
    good = (d2 > 100) & ((mask & badmask) == 0) & np.isfinite(d1 / d2)
    axes[0, 1].hist((d1 / d2)[good], bins=40, range=(0, 4))
    axes[0, 1].set(xlabel="full / first-half dark")
    for values in differences:
        values = values[np.isfinite(values)]
        axes[1, 1].hist(values, bins=200, range=(0, 100), cumulative=True,
                        density=True, histtype="step")
    axes[1, 1].set(xlim=(0, 50), ylim=(0.5, 1.1),
                   xlabel="total dark", ylabel="cumulative fraction")
    fig.tight_layout()

    base = Path(darkfile)
    base.parent.mkdir(parents=True, exist_ok=True)
    plot_file = base.with_suffix(".png")
    image_file = base.with_name(base.name + "2.jpg")
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    plt.imsave(image_file, np.clip(d1, -20, 100), cmap="gray",
               vmin=-20, vmax=100, origin="lower")
    return plot_file, image_file


"""Flat-field diagnostic image."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def flatplot(flat, file):
    """Save a clipped 0.5--1.5 flat-field image as JPEG."""
    output = Path(file).with_suffix(".jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output, np.clip(np.asarray(flat), 0.5, 1.5), cmap="gray",
               vmin=0.5, vmax=1.5, origin="lower")
    return output


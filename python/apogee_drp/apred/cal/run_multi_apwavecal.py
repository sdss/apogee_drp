"""Construct multi-night wavelength-calibration index lines."""

from __future__ import annotations

import numpy as np


def build_multiwave_lines(wave_ids, mjds, psf_by_wave, *,
                          shutdown_mjds=(55800, 56130, 56512, 56876,
                                         57230, 57600, 57966, 58335),
                          max_solutions=10, max_days=15):
    """Group wave solutions using the rules in ``run_multi_apwavecal.pro``."""
    ids = np.asarray(wave_ids, dtype=int)
    dates = np.asarray(mjds, dtype=int)
    order = np.argsort(dates)
    ids, dates = ids[order], dates[order]
    lines = []
    boundaries = np.asarray(shutdown_mjds)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        period = np.flatnonzero((dates >= lo) & (dates <= hi))
        cursor = 0
        while cursor < period.size:
            start = dates[period[cursor]]
            candidates = period[cursor : cursor + max_solutions]
            selected = candidates[dates[candidates] <= start + max_days]
            if selected.size == 0:
                break
            frames = ids[selected]
            psfid = psf_by_wave.get(int(frames[0]))
            if psfid is not None:
                name = f"{int(frames[0]):08d}"[:4] + "0000"
                frame_text = ",".join(f"{value:08d}" for value in frames)
                lines.append(
                    f"wave   {dates[selected].min()} {dates[selected].max()} "
                    f"{name} {frame_text} {int(psfid)}"
                )
            cursor += selected.size
    return lines


def run_multi_apwavecal(*args, outfile="mwave.par", **kwargs):
    """Build groups and write them to a calibration parameter file."""
    from pathlib import Path

    lines = build_multiwave_lines(*args, **kwargs)
    Path(outfile).write_text("\n".join(lines) + ("\n" if lines else ""),
                             encoding="utf-8")
    return lines


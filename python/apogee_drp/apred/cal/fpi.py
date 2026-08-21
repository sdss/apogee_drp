"""Build nightly APOGEE Fabry-Pérot wavelength calibrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...utils import apload
from .utils import product_build_lock

CHIPS = ("a", "b", "c")

__all__ = ["CHIPS", "build_fpi", "fpi_exposures"]


def fpi_exposures(rows):
    """Return sorted, unique FPI exposure numbers from exposure metadata."""
    if rows is None or len(rows) == 0:
        return []
    names = getattr(getattr(rows, "dtype", None), "names", None)
    if names is None and hasattr(rows, "colnames"):
        names = tuple(rows.colnames)
    if not names or "num" not in names or "exptype" not in names:
        raise ValueError("exposure metadata must contain num and exptype")
    numbers = [
        int(row["num"]) for row in rows
        if str(row["exptype"]).strip().upper() == "FPI"
    ]
    return sorted(set(numbers))


def _discover_exposures(*, observatory, mjd, verbose=False):
    from ...utils import info
    return fpi_exposures(info.expinfo(
        observatory=observatory, mjd5=int(mjd), verbose=verbose,
        fieldinfo=False))


def _select_library_psf(number, *, mjd, telescope, unlock=False):
    from .getpsfcal import getpsfcal
    selected = getpsfcal(
        number, mjd=mjd, telescope=telescope, psflibrary=True,
        unlock=unlock)
    if selected is None or int(selected) <= 0:
        raise RuntimeError(f"No library PSF was found for exposure {int(number):08d}")
    return int(selected)


def _process_exposures(exposures, *, load, darkid, flatid, psfid,
                       modelpsf, clobber, unlock, verbose):
    from ..process import process
    return process(exposures, load=load, darkid=darkid,
                   flatid=flatid, psfid=psfid, modelpsf=modelpsf, fluxid=None,
                   doproc=True, clobber=clobber, onedclobber=clobber,
                   unlock=unlock, verbose=verbose)


def _run_fpi_solution(mjd, *, observatory, apred, number, clobber, verbose):
    from .. import fpi
    return fpi.dailyfpiwave(int(mjd), observatory=observatory,
                            apred=apred, num=str(int(number)), clobber=clobber,
                            verbose=verbose, dependencies=False)


def build_fpi(fpiid, *, name=None, apred="daily", telescope="apo25m",
              darkid=None, flatid=None, psfid=None, modelpsf=None,
              fiberid=None, librarypsf=False, clobber=False, unlock=False,
              verbose=False, night_exposures=None):
    """Reduce a night's FPI frames and derive the requested WaveFPI product.

    Dependencies are not built here. In particular, the daily wavelength
    solution and selected PSF product must already exist; ``makecal`` builds
    them first only when called with ``dependencies=True``.
    """
    requested = [int(value) for value in np.atleast_1d(fpiid)]
    if not requested:
        raise ValueError("fpiid must contain at least one exposure")
    number = requested[0]
    output_name = number if name is None else int(name)
    if output_name != number:
        raise ValueError("WaveFPI output name must match the selected FPI exposure")
    load = apload.ApLoad(apred=apred, telescope=telescope)
    mjd = int(load.cmjd(number))
    observatory = telescope[:3].lower()
    with product_build_lock(load, "fpi", output_name, clobber=clobber,
                            unlock=unlock, verbose=verbose) as (build, outputs):
        if not build:
            return

        exposures = (_discover_exposures(
            observatory=observatory, mjd=mjd, verbose=verbose)
            if night_exposures is None else
            sorted(set(int(value) for value in night_exposures)))
        if not exposures:
            raise ValueError(f"No FPI exposures found for MJD {mjd}")
        if number not in exposures:
            raise ValueError(
                f"Requested exposure {number:08d} is not an FPI exposure on MJD {mjd}")
        if verbose:
            print(f"Found {len(exposures)} FPI exposures for MJD {mjd}: "
                  + ", ".join(f"{value:08d}" for value in exposures))

        selected_psf = psfid
        if librarypsf:
            if psfid is not None or modelpsf is not None:
                raise ValueError(
                    "librarypsf cannot be combined with psfid or modelpsf")
            selected_psf = _select_library_psf(
                number, mjd=mjd, telescope=telescope, unlock=unlock)
        if selected_psf is None and modelpsf is None:
            raise ValueError("psfid, modelpsf, or librarypsf is required")

        _process_exposures(
            exposures, load=load, darkid=darkid, flatid=flatid,
            psfid=selected_psf, modelpsf=modelpsf, clobber=clobber,
            unlock=unlock, verbose=verbose)
        _run_fpi_solution(
            mjd, observatory=observatory, apred=apred, number=number,
            clobber=clobber, verbose=verbose)
        missing = [filename for filename in outputs
                   if not Path(filename).is_file() or Path(filename).stat().st_size == 0]
        if missing:
            raise RuntimeError(
                "FPI wavelength solution did not create: " + ", ".join(missing))

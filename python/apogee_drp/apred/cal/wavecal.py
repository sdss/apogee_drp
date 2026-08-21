"""Build single, multi-epoch, and daily APOGEE wavelength calibrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from ...utils import apload
from .utils import calibration_lock, product_build_lock

CHIPS = ("a", "b", "c")

__all__ = [
    "arc_flux_metric", "build_dailywave", "build_multiwave", "build_wave",
]


def _marker_file(load, number, suffix=".dat"):
    template = load.filename("Wave", num=number)
    return str(Path(template).with_suffix(suffix))


def _lines_file(load, number):
    template = Path(load.filename("Wave", num=int(number)))
    return str(template.with_name(
        template.name.replace("Wave-", "Lines-", 1)))


def arc_flux_metric(image, header, *, spatial_bin=8, smooth_width=7):
    """Return counts/read for the diagnostic line region in an ap2D image."""
    data = np.asarray(image, dtype=float)
    if data.ndim != 2:
        raise ValueError("arc image must be two-dimensional")
    nx = data.shape[1]
    if bool(header.get("LAMPUNE", False)):
        center, threshold = 1452, 40.0
    elif bool(header.get("LAMPTHAR", False)):
        center, threshold = 1566, 1000.0
    else:
        center, threshold = 1000, 10.0
    lo, hi = max(center - 100, 0), min(center + 101, nx)
    if hi - lo < 3:
        raise ValueError("arc image does not cover the diagnostic line region")
    nread = int(header.get("NREAD", 0))
    if nread <= 0:
        raise ValueError("arc image has no positive NREAD value")
    subsection = median_filter(
        np.nan_to_num(data[:, lo:hi], nan=0.0),
        size=(1, int(smooth_width)), mode="nearest")
    usable_rows = subsection.shape[0] // int(spatial_bin) * int(spatial_bin)
    if usable_rows == 0:
        raise ValueError("arc image is too short for the spatial bin")
    rebinned = subsection[:usable_rows].reshape(
        -1, int(spatial_bin), subsection.shape[1]).sum(axis=1)
    metric = float(np.nanmedian(np.nanmax(rebinned, axis=1)) / nread)
    return metric, threshold


def _check_arc(load, number):
    filename = load.filename("2D", num=int(number), chip="b")
    if not Path(filename).is_file():
        return False, f"{filename} NOT FOUND"
    frame = load.frame(int(number), chip="b")
    metric, threshold = arc_flux_metric(frame["flux"], frame["header"])
    return metric >= threshold, (
        f"{number:08d}: arc flux/read={metric:.1f}, required={threshold:.1f}")


def _process_frames(frames, *, load, darkid, flatid, psfid, modelpsf,
                    clobber, unlock, verbose):
    from ..process import process
    return process(
        frames, load=load, darkid=darkid, flatid=flatid, psfid=psfid,
        modelpsf=modelpsf, fluxid=None, doproc=True, clobber=clobber,
        onedclobber=clobber, unlock=unlock, verbose=verbose)


def _run_wavecal(frames, *, name, load, npoly, nofit, plot, clobber,
                 verbose, init, dependencies):
    from .. import wave
    return wave.wavecal(
        frames, rows=np.arange(300), name=name, npoly=int(npoly),
        inst=load.instrument, plot=plot, hard=plot, nofit=nofit,
        verbose=verbose, clobber=clobber, init=init, vers=load.apred,
        dependencies=dependencies)


def build_wave(waveid, *, name=None, apred="daily", telescope="apo25m",
               darkid=None, flatid=None, psfid=None, modelpsf=None,
               fiberid=None, npoly=4, clobber=False, nowait=False,
               nofit=False, unlock=False, plot=False, verbose=False):
    """Build one wavelength solution or only its line measurements."""
    del fiberid  # trace information is carried by the selected PSF product
    frames = [int(value) for value in np.atleast_1d(waveid)]
    if not frames:
        raise ValueError("waveid must contain at least one exposure")
    output_name = frames[0] if name is None else name
    load = apload.ApLoad(apred=apred, telescope=telescope)
    waittime = 0 if nowait else 10
    if nofit:
        lockfile = Path(load.filename("Wave", num=output_name))
        lockfile.parent.mkdir(parents=True, exist_ok=True)
        lock_context = calibration_lock(
            lockfile, waittime=waittime, unlock=unlock)
    else:
        lock_context = product_build_lock(
            load, "wave", output_name, clobber=clobber, unlock=unlock,
            waittime=waittime, verbose=verbose)

    with lock_context as state:
        if not nofit:
            build, _ = state
            if not build:
                return
        if psfid is None and modelpsf is None:
            raise ValueError("psfid or modelpsf is required to process arc exposures")
        _process_frames(
            frames, load=load, darkid=darkid, flatid=flatid,
            psfid=psfid, modelpsf=modelpsf, clobber=clobber,
            unlock=unlock, verbose=verbose)
        usable = []
        for frame in frames:
            okay, message = _check_arc(load, frame)
            if verbose:
                print(message)
            if okay:
                usable.append(frame)
        if not usable:
            raise ValueError("No input arc exposure passed the flux check")
        _run_wavecal(
            usable, name=output_name, load=load, npoly=npoly, nofit=nofit,
            plot=plot, clobber=clobber, verbose=verbose, init=False,
            dependencies=True)
        if nofit:
            missing = [frame for frame in usable
                       if not Path(_lines_file(load, frame)).is_file()]
            if missing:
                raise RuntimeError("Line measurement failed for: " +
                                   ", ".join(map(str, missing)))
            return
        if not load.product_exists("wave", output_name):
            raise RuntimeError("Wavelength solver did not create all chip products")
        Path(_marker_file(load, output_name)).touch()


def build_multiwave(waveid, *, name=None, apred="daily", telescope="apo25m",
                    npoly=4, clobber=False, nowait=False, unlock=False,
                    plot=False, verbose=False, dependencies=False,
                    single_builder_options=None):
    """Fit one simultaneous wavelength solution across multiple nights."""
    frames = [int(value) for value in np.atleast_1d(waveid)]
    if not frames:
        raise ValueError("waveid must contain at least one exposure")
    output_name = frames[0] if name is None else name
    load = apload.ApLoad(apred=apred, telescope=telescope)
    with product_build_lock(
        load, "multiwave", output_name, clobber=clobber, unlock=unlock,
        waittime=(0 if nowait else 10), verbose=verbose,
    ) as (build, _):
        if not build:
            return
        if dependencies:
            options = dict(single_builder_options or {})
            for index in range(0, len(frames), 2):
                build_wave(
                    frames[index:index + 2], name=frames[index], apred=apred,
                    telescope=telescope, nofit=True, unlock=unlock,
                    verbose=verbose, **options)
        available = [frame for frame in frames
                     if Path(_lines_file(load, frame)).is_file()]
        if not available:
            raise ValueError("No individual arc-line measurements are available")
        _run_wavecal(
            frames, name=output_name, load=load, npoly=npoly, nofit=False,
            plot=plot, clobber=clobber, verbose=verbose, init=False,
            dependencies=False)
        if not load.product_exists("multiwave", output_name):
            raise RuntimeError("Multiwave solver did not create all chip products")
        Path(_marker_file(load, output_name, suffix=".multidat")).touch()


def _run_dailywave(mjd, *, observatory, apred, npoly, clobber, verbose,
                   init, dependencies):
    from .. import wave
    return wave.dailywave(
        int(mjd), observatory=observatory, apred=apred, npoly=int(npoly),
        init=init, clobber=clobber, verbose=verbose,
        dependencies=dependencies)


def build_dailywave(mjd, *, apred="daily", telescope="apo25m", darkid=None,
                    flatid=None, psfid=None, modelpsf=None, fiberid=None,
                    npoly=4, clobber=False, nowait=False, nofit=False,
                    unlock=False, librarypsf=False, verbose=False,
                    dependencies=False):
    """Build a nightly solution from available nearby arc measurements.

    Missing individual solutions are acceptable. They are generated only
    when ``dependencies=True``; otherwise ``wave.dailywave`` skips them.
    """
    del darkid, flatid, psfid, modelpsf, fiberid, librarypsf
    if nofit:
        raise ValueError("nofit is meaningful for individual wave solutions only")
    mjd = int(mjd)
    load = apload.ApLoad(apred=apred, telescope=telescope)
    with product_build_lock(
        load, "dailywave", mjd, clobber=clobber, unlock=unlock,
        waittime=(0 if nowait else 10), verbose=verbose,
    ) as (build, _):
        if not build:
            return
        _run_dailywave(
            mjd, observatory=telescope[:3].lower(), apred=apred, npoly=npoly,
            clobber=clobber, verbose=verbose, init=False,
            dependencies=dependencies)
        if not load.product_exists("dailywave", mjd):
            raise RuntimeError("Daily wavelength solver did not create all chip products")
        Path(_marker_file(load, mjd)).touch()

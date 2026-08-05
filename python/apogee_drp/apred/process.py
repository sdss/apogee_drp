"""Calibration-exposure processing wrapper.

This module is the Python counterpart of ``process.pro``.  It reduces a
list of raw ramps to ap2D images and, optionally, extracts the three ap1D
spectra.  The numerical work is delegated to :mod:`apogee_drp.apred.ap3d`
and :mod:`apogee_drp.apred.ap2d`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from ..utils.apload import ApLoad
from .ap2d import ap2dproc
from .ap3d import process_file

__all__ = ["ProcessRecord", "process"]


@dataclass(frozen=True)
class ProcessRecord:
    """Result of one exposure/chip or one exposure extraction operation."""

    exposure: int
    stage: str
    chip: str | None
    status: str
    input_file: str
    output_file: str | None
    elapsed: float


def _id(current: int | None, legacy: int | None, name: str) -> int | None:
    """Combine a Python ``*id`` argument with its legacy IDL spelling."""

    current = None if current in (None, 0) else int(current)
    legacy = None if legacy in (None, 0) else int(legacy)
    if current is not None and legacy is not None and current != legacy:
        raise ValueError(f"Conflicting {name} calibration IDs: {current} and {legacy}")
    return current if current is not None else legacy


def _chip_file(base: str, root: str, chip: str) -> str:
    """Expand an ``ApLoad.filename(..., chips=True)`` template."""

    marker = f"{root}-"
    if marker not in base:
        raise ValueError(f"Cannot insert chip into APOGEE filename {base!r}")
    return base.replace(marker, f"{root}-{chip}-", 1)


def process(
    nums: int | Sequence[int] | np.ndarray,
    *,
    apred: str = "daily",
    telescope: str = "apo25m",
    load: Any | None = None,
    cmjd: int | str | None = None,
    clobber: bool = False,
    onedclobber: bool = False,
    detid: int | None = None,
    darkid: int | None = None,
    flatid: int | None = None,
    traceid: int | None = None,
    psfid: int | None = None,
    modelpsf: int | None = None,
    fluxid: int | None = None,
    waveid: int | None = None,
    littrowid: int | None = None,
    persistid: int | None = None,
    detector: int | None = None,
    dark: int | None = None,
    flat: int | None = None,
    trace: int | None = None,
    psf: int | None = None,
    flux: int | None = None,
    wave: int | None = None,
    littrow: int | None = None,
    persist: int | None = None,
    nocr: bool = False,
    jchip: int | str | None = None,
    nfs: int = 0,
    doproc: bool = False,
    doap3dproc: bool = False,
    doap2dproc: bool = False,
    outdir: str | Path | None = None,
    maxread: int | Sequence[int] | None = None,
    skywave: bool = False,
    unlock: bool = False,
    verbose: bool = False,
    process_3d: Callable[..., Any] | None = None,
    process_2d: Callable[..., Any] | None = None,
    **process_options: Any,
) -> list[ProcessRecord]:
    """Reduce calibration exposures from raw ramps through ap1D spectra.

    ``doproc`` performs both stages, while ``doap3dproc`` and
    ``doap2dproc`` select them independently.  Legacy calibration argument
    names are accepted because several translated calibration builders still
    use the original IDL call spelling.
    """

    numbers = [int(value) for value in np.atleast_1d(nums)]
    if not numbers:
        return []
    if load is None:
        load = ApLoad(apred=apred, telescope=telescope)
    process_3d = process_file if process_3d is None else process_3d
    process_2d = ap2dproc if process_2d is None else process_2d

    ids = {
        "Detector": _id(detid, detector, "detector"),
        "Dark": _id(darkid, dark, "dark"),
        "Flat": _id(flatid, flat, "flat"),
        "PSF": _id(psfid if psfid is not None else traceid,
                   psf if psf is not None else trace, "PSF"),
        "PSFModel": None if modelpsf in (None, 0) else int(modelpsf),
        "Flux": _id(fluxid, flux, "flux"),
        "Wave": _id(waveid, wave, "wave"),
        "Littrow": _id(littrowid, littrow, "Littrow"),
        "Persist": _id(persistid, persist, "persistence"),
    }
    do_3d = bool(doproc or doap3dproc)
    do_2d = bool(doproc or doap2dproc)
    if not do_3d and not do_2d:
        return []

    chip_names = ("a", "b", "c")
    if jchip is not None:
        chip = chip_names[int(jchip)] if isinstance(jchip, (int, np.integer)) else str(jchip)
        if chip not in chip_names:
            raise ValueError("jchip must be 0, 1, 2, 'a', 'b', or 'c'")
        chips = (chip,)
    else:
        chips = chip_names
    if maxread is None:
        max_reads = (None, None, None)
    elif np.ndim(maxread) == 0:
        max_reads = (int(maxread),) * 3
    else:
        values = tuple(int(value) for value in maxread)
        if len(values) != 3:
            raise ValueError("maxread must be a scalar or a three-element sequence")
        max_reads = values

    records: list[ProcessRecord] = []
    onedclobber = bool(onedclobber or clobber)
    for number in numbers:
        mjd = str(cmjd if cmjd is not None else load.cmjd(number))
        raw_base = load.filename("R", num=number, mjd=mjd, chips=True)
        two_d_base = load.filename("2D", num=number, mjd=mjd, chips=True)
        if do_3d:
            for chip in chips:
                started = perf_counter()
                raw_file = _chip_file(raw_base, "R", chip)
                output_file = _chip_file(two_d_base, "2D", chip)
                if outdir is not None:
                    output_file = str(Path(outdir) / Path(output_file).name)
                if Path(output_file).exists() and not clobber:
                    status = "skipped"
                else:
                    calibration_files = {}
                    mapping = {
                        "detector": "Detector", "bpm": "BPM",
                        "dark": "Dark", "flat": "Flat",
                        "littrow": "Littrow", "persistence_mask": "Persist",
                    }
                    for keyword, root in mapping.items():
                        # As in approcess.pro, the BPM product is keyed by
                        # darkid rather than by a separate calibration ID.
                        cal_id = ids["Dark"] if root == "BPM" else ids[root]
                        if cal_id is None or (root == "Littrow" and chip != "b"):
                            calibration_files[keyword] = None
                        else:
                            base = load.filename(root, num=cal_id, chips=True)
                            calibration_files[keyword] = _chip_file(base, root, chip)
                    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                    process_3d(
                        raw_file, output_file, overwrite=clobber,
                        detect_cosmic_rays=not nocr, fix_cosmic_rays=not nocr,
                        up_the_ramp=(int(nfs) == 0), nfowler=int(nfs),
                        max_read=max_reads[chip_names.index(chip)],
                        verbose=verbose, **calibration_files, **process_options,
                    )
                    status = "processed"
                records.append(ProcessRecord(number, "3d", chip, status,
                                             raw_file, output_file,
                                             perf_counter() - started))

        if do_2d:
            all_two_d = [_chip_file(two_d_base, "2D", chip) for chip in chip_names]
            if outdir is not None:
                all_two_d = [str(Path(outdir) / Path(filename).name)
                             for filename in all_two_d]
            missing = [filename for filename in all_two_d if not Path(filename).exists()]
            if missing:
                raise FileNotFoundError(
                    f"Cannot extract exposure {number:08d}; missing ap2D files: "
                    + ", ".join(missing)
                )
            psf_id = ids["PSF"]
            if psf_id is None:
                raise ValueError("psfid (or traceid) is required for 2D-to-1D extraction")
            started = perf_counter()
            inpfile = str(Path(all_two_d[0]).parent / f"{number:08d}")
            psf_dir = Path(load.filename("PSF", num=psf_id, chip="c", dir=True))
            psf_file = str(psf_dir / f"{psf_id:08d}")
            model_file = None
            if ids["PSFModel"] is not None:
                model_file = str(Path(load.filename("PSFModel", num=ids["PSFModel"],
                                                    chip="c", dir=True)) /
                                 f"{ids['PSFModel']:08d}")
            flux_file = None
            if ids["Flux"] is not None:
                flux_file = load.filename("Flux", num=ids["Flux"], chips=True)
            wave_file = None
            if ids["Wave"] is not None:
                wave_file = load.filename("Wave", num=ids["Wave"], chips=True)
            process_2d(
                inpfile, psf_file, extract_type=4, load=load,
                modelpsffile=model_file, fluxcalfile=flux_file,
                wavefile=wave_file, skywave=skywave,
                clobber=onedclobber, unlock=unlock, verbose=verbose,
            )
            records.append(ProcessRecord(number, "2d", None, "processed",
                                         inpfile, None, perf_counter() - started))
    return records

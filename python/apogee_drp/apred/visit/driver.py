"""Testable Python driver for the APOGEE visit-level reduction.

This module translates the control flow in ``ap1dvisit.pro``.  The numerical
reduction stages are deliberately supplied by a backend: most of those stages
are independent IDL programs and have not yet been translated on the ``daily``
branch.  Keeping them behind :class:`VisitBackend` makes the driver executable
with either compatibility wrappers or native Python implementations, and makes
the orchestration testable without real APOGEE files.

The important array convention is the native Python convention used elsewhere
in ``apogee_drp``: spectra have shape ``(nfiber, npix)``.  A backend wrapping
IDL must transpose the legacy ``(npix, nfiber)`` representation at its boundary.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol, Sequence

import numpy as np

__all__ = [
    "CHIPS",
    "FrameFailure",
    "PlanFailure",
    "ShiftRecord",
    "VisitBackend",
    "VisitResult",
    "ap1dvisit",
    "sanitize_frame",
    "select_visit_objects",
]

CHIPS = ("a", "b", "c")
ALLOWED_PLATE_TYPES = {"normal", "twilight", "sky", "single", "cal"}


class PlanFailure(RuntimeError):
    """A plan cannot be reduced."""


class FrameFailure(RuntimeError):
    """A single exposure cannot be reduced."""


@dataclass
class ShiftRecord:
    """Python equivalent of the IDL ``shiftstr`` structure."""

    index: int = -1
    framenum: str = ""
    shift: float = 999999.0
    shifterr: float = 999999.0
    shiftfit: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    chipshift: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 2), dtype=np.float32)
    )
    chipfit: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )
    pixshift: float = 0.0
    sn: float = -1.0


@dataclass
class VisitResult:
    """Result for one input plan."""

    planfile: str
    plate_file: str | None = None
    visit_summary_file: str | None = None
    processed_frames: int = 0
    failed_frames: int = 0
    skipped_plate_reduction: bool = False
    errors: list[str] = field(default_factory=list)


class VisitBackend(Protocol):
    """Operations required by :func:`ap1dvisit`.

    Methods use ordinary dictionaries for plan metadata and backend-defined
    frame/plug-map objects.  This is intentional: current APOGEE Python code
    uses a mixture of dictionaries, Astropy tables, and ``ApData`` objects.
    """

    # Plan, paths, calibrations, and plug maps.
    def load_plan(self, planfile: str, *, verbose: bool) -> MutableMapping[str, Any]: ...
    def directories(self, plan: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def filename(self, kind: str, **kwargs: Any) -> str | Sequence[str]: ...
    def load_plugmap(
        self,
        plan: Mapping[str, Any],
        *,
        mapper_data: Any = None,
    ) -> Any: ...
    def make_lsf(self, lsfid: Any) -> None: ...
    def calibration_files(
        self, plan: Mapping[str, Any], chips: Sequence[str]
    ) -> tuple[Sequence[str], Sequence[str]]: ...
    def read_relflux(self, filename: str) -> np.ndarray: ...

    # Frame I/O and numerical reduction stages.
    def frame_number(self, exposure: Any, plan: Mapping[str, Any]) -> str: ...
    def one_d_files(
        self, plan: Mapping[str, Any], framenum: str, chips: Sequence[str]
    ) -> Sequence[str]: ...
    def cframe_files(
        self, plan: Mapping[str, Any], framenum: str, chips: Sequence[str]
    ) -> Sequence[str]: ...
    def validate_files(
        self, files: Sequence[str], *, mjd: int, kind: str
    ) -> None: ...
    def load_frame(self, files: Sequence[str], *, kind: str) -> Any: ...
    def prepare_frame(
        self,
        frame: Any,
        *,
        plan: Mapping[str, Any],
        wavefiles: Sequence[str],
        lsffiles: Sequence[str],
        plate_dir: str,
        newwave: bool,
    ) -> Any: ...
    def header_value(
        self, frame: Any, keyword: str, default: Any = None, chip: int = 0
    ) -> Any: ...
    def add_header(
        self,
        frame: Any,
        keyword: str,
        value: Any,
        comment: str | None = None,
    ) -> None: ...
    def add_history(self, frame: Any, text: str) -> None: ...
    def dither_shift(
        self,
        reference: Any,
        frame: Any,
        *,
        plugmap: Any,
        plan: Mapping[str, Any],
        plotfile: str,
        nofit: bool,
    ) -> Any: ...
    def wavelength_calibrate(
        self,
        frame: Any,
        *,
        plugmap: Any,
        plan: Mapping[str, Any],
        plotfile: str,
        dithonly: bool,
    ) -> Any: ...
    def sky_subtract(
        self, frame: Any, *, plugmap: Any, force: bool
    ) -> Any: ...
    def telluric_correct(
        self,
        frame: Any,
        *,
        plugmap: Any,
        plan: Mapping[str, Any],
        plots_dir: str,
        test: bool,
        force: bool,
    ) -> tuple[Any, Any]: ...
    def write_cframes(
        self, frame: Any, plugmap: Any, files: Sequence[str]
    ) -> None: ...
    def shift_result(self, frame: Any) -> Mapping[str, Any]: ...
    def chip_pixel_arrays(
        self, frame: Any, chip: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def chip_arrays(self, frame: Any, chip: int) -> tuple[np.ndarray, np.ndarray]: ...
    def combine_dithers(
        self,
        frames: Sequence[Any],
        shifts: Sequence[ShiftRecord],
        *,
        plugmap: Any,
        nodither: bool,
    ) -> tuple[Any, Any]: ...
    def flux_calibrate(self, frame: Any, *, plugmap: Any) -> Any: ...
    def write_visit_products(
        self,
        frame: Any,
        *,
        plan: Mapping[str, Any],
        plugmap: Any,
        shifts: Sequence[ShiftRecord],
        pairs: Any,
        survey: str,
        relflux: np.ndarray,
    ) -> str: ...

    # Summary/RV metadata.  RV measurement itself is performed downstream,
    # matching the current IDL routine, which only assembles visit metadata.
    def plate_file(self, plan: Mapping[str, Any]) -> str: ...
    def visit_summary_file(self, plan: Mapping[str, Any], plugmap: Any) -> str: ...
    def build_visit_rows(
        self,
        *,
        plan: Mapping[str, Any],
        plugmap: Any,
        object_indices: np.ndarray,
        survey: str,
        nframes: int,
        relflux: np.ndarray,
    ) -> Any: ...
    def write_visit_summary(
        self,
        filename: str,
        rows: Any,
        *,
        plan: Mapping[str, Any],
        source_frame: Any,
    ) -> None: ...
    def ingest_visits(self, rows: Any) -> None: ...

    # Small compatibility hooks for heterogeneous legacy containers.
    def fiber_data(self, plugmap: Any) -> Any: ...
    def set_plugmap_mjd(self, plugmap: Any, mjd: int) -> None: ...
    def write_tellstar_summary(
        self, plan: Mapping[str, Any], tellstars: Sequence[Any]
    ) -> None: ...


def _value(row: Any, name: str, default: Any = None) -> Any:
    """Get a field from a mapping, structured row, or attribute object."""

    if isinstance(row, Mapping):
        return row.get(name, default)
    try:
        return row[name]
    except (IndexError, KeyError, TypeError, ValueError):
        return getattr(row, name, default)


def _column(rows: Any, name: str, default: Any = None) -> np.ndarray:
    """Return a field from an Astropy table, structured array, or row list."""

    try:
        return np.asarray(rows[name])
    except (IndexError, KeyError, TypeError, ValueError):
        return np.asarray([_value(row, name, default) for row in rows])


def _string_column(rows: Any, name: str, default: str = "") -> np.ndarray:
    values = _column(rows, name, default)
    return np.char.strip(values.astype(str))


def select_visit_objects(fiberdata: Any, *, fps: bool) -> np.ndarray:
    """Select rows that receive apVisit products.

    This is the direct equivalent of the ``objind=where(...)`` block in IDL.
    FPS additionally requires a valid, assigned, on-target fiber.
    """

    spectrograph = _column(fiberdata, "spectrographid", 0).astype(int)
    holetype = _string_column(fiberdata, "holetype")
    objtype = _string_column(fiberdata, "objtype")
    apogee_id = _string_column(fiberdata, "tmass_style")

    keep = (
        (spectrograph == 2)
        & (holetype == "OBJECT")
        & np.isin(objtype, ("STAR", "HOT_STD"))
        & (objtype != "SKY")
        & (objtype != "none")
        & (apogee_id != "2MNone")
    )
    if fps:
        keep &= (
            (_column(fiberdata, "assigned", 0).astype(int) == 1)
            & (_column(fiberdata, "on_target", 0).astype(int) == 1)
            & (_column(fiberdata, "valid", 0).astype(int) == 1)
        )
    return np.flatnonzero(keep)


def sanitize_frame(
    frame: Any,
    *,
    chip_arrays: Any,
    bad_error: float = 1.0e30,
) -> int:
    """Replace non-finite flux/error and non-positive errors in all chips.

    ``chip_arrays(frame, chip)`` must return ``(flux, error, mask)`` arrays.
    Arrays are modified in place.  The return value is the number of pixels
    marked bad, counting each pixel once per chip.
    """

    nbad = 0
    for chip in range(3):
        flux, error, mask = chip_arrays(frame, chip)
        bad = ~np.isfinite(flux) | ~np.isfinite(error) | (error <= 0)
        count = int(np.count_nonzero(bad))
        if count:
            flux[bad] = 0.0
            error[bad] = bad_error
            mask[bad] |= 1
            nbad += count
    return nbad


def _science_indices(fiberdata: Any, *, single: bool) -> np.ndarray:
    objtype = _string_column(fiberdata, "objtype")
    spectrograph = _column(fiberdata, "spectrographid", 0).astype(int)
    keep = (objtype != "SKY") & (spectrograph == 2)
    if not single:
        mag = np.asarray([np.asarray(x)[1] for x in _column(fiberdata, "mag")])
        fiberid = _column(fiberdata, "fiberid", 0).astype(int)
        keep &= (mag > 7.5) & (fiberid != 195)
    return np.flatnonzero(keep)


def _frame_score(
    backend: VisitBackend,
    frame: Any,
    fiberdata: Any,
    objects: np.ndarray,
    *,
    single: bool,
) -> float:
    """Return the IDL sorting statistic (S/N for single, zeropoint otherwise)."""

    if objects.size == 0:
        return -1.0
    hmags = np.asarray(
        [float(np.asarray(_value(fiberdata[i], "mag"))[1]) for i in objects]
    )
    fiberids = np.asarray(
        [int(_value(fiberdata[i], "fiberid")) for i in objects]
    )
    flux, error = backend.chip_arrays(frame, 1)
    locations = 300 - fiberids
    if single:
        location = locations[np.argmin(hmags)]
        ratio = np.divide(
            flux[location],
            error[location],
            out=np.full_like(flux[location], np.nan, dtype=float),
            where=error[location] > 0,
        )
        return float(np.nanmedian(ratio))

    medflux = np.nanmedian(flux[locations], axis=1)
    valid = np.isfinite(medflux) & (medflux > 0)
    if not np.any(valid):
        return -1.0
    return float(np.nanmedian(hmags[valid] + 2.5 * np.log10(medflux[valid])))


def _shift_record(
    backend: VisitBackend,
    frame: Any,
    index: int,
    framenum: str,
    score: float,
) -> ShiftRecord:
    result = backend.shift_result(frame)
    shiftfit = np.asarray(result.get("shiftfit", (0.0, 0.0)), dtype=np.float32)
    return ShiftRecord(
        index=index,
        framenum=str(framenum),
        shift=float(backend.header_value(frame, "DITHSH", shiftfit[0])),
        shifterr=float(backend.header_value(frame, "EDITHSH", 0.0)),
        shiftfit=shiftfit,
        chipshift=np.asarray(
            result.get("chipshift", np.zeros((3, 2))), dtype=np.float32
        ),
        chipfit=np.asarray(result.get("chipfit", np.zeros(4)), dtype=np.float32),
        pixshift=float(backend.header_value(frame, "MEDWSH", 0.0)),
        sn=score,
    )


def ap1dvisit(
    planfiles: str | Path | Sequence[str | Path],
    backend: VisitBackend,
    *,
    clobber: bool = False,
    verbose: bool = False,
    newwave: bool = False,
    test: bool = False,
    mapper_data: Any = None,
    halt: bool = False,
    dithonly: bool = False,
    ap1dwavecal: bool = False,
    force: bool | None = None,
    logger: logging.Logger | None = None,
) -> list[VisitResult]:
    """Reduce one or more APOGEE visit plan files.

    Parameters mirror the IDL keywords.  Unlike IDL, errors are explicit:
    plan-level failures are recorded in the returned :class:`VisitResult`;
    with ``halt=True`` they are raised immediately.  A failed telluric
    correction rejects only that exposure unless ``force=True``.
    """

    log = logger or logging.getLogger(__name__)
    if isinstance(planfiles, (str, Path)):
        plans = [str(planfiles)]
    else:
        plans = [str(path) for path in planfiles]
    if not plans:
        return []
    if ap1dwavecal:
        newwave = True

    started = time.monotonic()
    results: list[VisitResult] = []
    log.info("RUNNING AP1DVISIT: %d plan file(s)", len(plans))

    for planfile in plans:
        outcome = VisitResult(planfile=planfile)
        results.append(outcome)
        try:
            plan = backend.load_plan(planfile, verbose=verbose)
            plate_type = str(plan.get("platetype", "normal")).strip().lower()
            if plate_type not in ALLOWED_PLATE_TYPES:
                raise PlanFailure(f"unsupported plate type {plate_type!r}")
            plan["platetype"] = plate_type
            plan.setdefault("field", "")
            fps = int(plan["mjd"]) >= 59556
            survey = str(plan.get("survey", "mwm" if
                                  int(plan["plateid"]) >= 15000 or int(plan["plateid"])
                                  == 0 else "apogee"))
            use_force = bool(plan.get("force", False) if force is None else force)
            dirs = backend.directories(plan)
            plugmap = None if plate_type == "cal" else backend.load_plugmap(
                plan, mapper_data=mapper_data)
            if plugmap is not None:
                backend.set_plugmap_mjd(plugmap, int(plan["mjd"]))
                fiberdata = backend.fiber_data(plugmap)
                science = _science_indices(fiberdata,single=plate_type == "single")
            else:
                fiberdata = None
                science = np.array([], dtype=int)

            if plugmap is not None and int(plan.get("fluxid", 0)) != 0:
                fluxfile = backend.filename("Flux", chip="b", num=plan["fluxid"])
                relflux = backend.read_relflux(str(fluxfile))
            else:
                relflux = np.ones(300, dtype=np.float32)

            backend.make_lsf(plan["lsfid"])
            wavefiles, lsffiles = backend.calibration_files(plan, CHIPS)
            backend.validate_files(wavefiles, mjd=int(plan["mjd"]), kind="Wave")
            backend.validate_files(lsffiles, mjd=int(plan["mjd"]), kind="LSF")
            plate_dir = str(backend.filename("Plate",mjd=plan["mjd"],
                                             plate=plan["plateid"], chip="a",
                                             field=plan["field"], directory=True))
            Path(plate_dir).mkdir(parents=True, exist_ok=True)
            plots_dir = str(Path(plate_dir) / "plots")
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            outcome.plate_file = backend.plate_file(plan)

            exposures = list(plan["APEXP"])
            nframes = len(exposures)
            allframes: list[Any] = []
            tellstars: list[Any] = []
            shifts: list[ShiftRecord] = []
            nodither = True
            reference_frame = None
            reference_command = None
            telluric_errors = 0

            if Path(outcome.plate_file).exists() and not clobber:
                outcome.skipped_plate_reduction = True
                source_frame = backend.load_frame([outcome.plate_file],kind="Plate")
            else:
                for j, exposure in enumerate(exposures):
                    frame_start = time.monotonic()
                    framenum = backend.frame_number(exposure, plan)
                    cfiles = list(backend.cframe_files(plan, framenum, CHIPS))
                    try:
                        if clobber or not all(Path(f).exists() for f in cfiles):
                            files = list(backend.one_d_files(plan,framenum,CHIPS))
                            backend.validate_files(files,mjd=int(plan["mjd"]),kind="1D")
                            raw = backend.load_frame(files, kind="1D")
                            sanitize_frame(raw,chip_arrays=backend.chip_pixel_arrays)
                            frame = backend.prepare_frame(raw, plan=plan,
                                                          wavefiles=wavefiles,
                                                          lsffiles=lsffiles,
                                                          plate_dir=plate_dir,
                                                          newwave=newwave)
                            commanded = float(backend.header_value(frame, "DITHPIX",0.0))
                            if (j > 0 and commanded != 0 and reference_command is not None and
                                 abs(commanded - reference_command) > 0.002 ):
                                nodither = False
                            nofit = plate_type == "single"
                            if j > 0 and reference_frame is not None:
                                shiftout = backend.dither_shift(
                                    reference_frame, frame,
                                    plugmap=plugmap, plan=plan,
                                    plotfile=str( Path(plots_dir) /
                                                  f"dithershift-{framenum}" ),
                                    nofit=nofit)
                            else:
                                shiftout = {"shiftfit": np.zeros(2),
                                            "shifterr": 0.0,
                                            "chipshift": np.zeros((3, 2)),
                                            "chipfit": np.zeros(4)}
                            if j == 0 or nodither:
                                reference_frame = frame
                                if commanded != 0:
                                    reference_command = commanded
                                shiftout = dict(shiftout)
                                shiftout["shiftfit"] = np.zeros(2)
                                shiftout["shifterr"] = 0.0
                            shiftfit = np.asarray(shiftout["shiftfit"])
                            backend.add_history(
                                frame, "APDITHERSHIFT: Measuring dither shift"
                            )
                            if shiftfit[0] == 0:
                                backend.add_history(frame,
                                        "APDITHERSHIFT: This is the reference frame")
                            backend.add_header(frame, "DITHSH", float(shiftfit[0]))
                            backend.add_header(frame, "DITHSLOP", float(shiftfit[1]))
                            backend.add_header(frame, "EDITHSH",
                                               float(shiftout.get("shifterr", 0.0)))
                            if ap1dwavecal:
                                frame = backend.wavelength_calibrate(
                                    frame, plugmap=plugmap, plan=plan,
                                    plotfile=str( Path(plots_dir) /
                                                  f"pixshift-{framenum}" ),
                                    dithonly=dithonly)
                            frame = backend.sky_subtract(frame, plugmap=plugmap,
                                                         force=use_force)
                            if plate_type in {"sky", "cal"}:
                                continue
                            frame, tellstar = backend.telluric_correct(frame,
                                                      plugmap=plugmap, plan=plan,
                                                      plots_dir=plots_dir, test=test,
                                                      force=use_force)
                            tellstars.append(tellstar)
                            backend.write_cframes(frame, plugmap, cfiles)

                        backend.validate_files(cfiles,
                                               mjd=int(plan["mjd"]), kind="Cframe" )
                        frame = backend.load_frame(cfiles, kind="Cframe")
                        commanded = float(backend.header_value(frame,"DITHPIX",0.0))
                        if reference_command is None:
                            reference_command = commanded
                            reference_frame = frame
                        elif (
                            commanded != 0
                            and abs(commanded - reference_command) > 0.002
                        ):
                            nodither = False
                        score = _frame_score(backend, frame, fiberdata,
                                             science, single=plate_type=="single")
                        shifts.append(_shift_record(backend,frame,j,framenum,score))
                        allframes.append(frame)
                        outcome.processed_frames += 1
                        log.info("%s frame %s completed in %.1f s",
                                 planfile, framenum, time.monotonic() -
                                 frame_start)
                    except Exception as exc:
                        outcome.failed_frames += 1
                        message = f"frame {framenum}: {exc}"
                        outcome.errors.append(message)
                        log.exception("%s %s", planfile, message)
                        if "tellur" in str(exc).lower():
                            telluric_errors += 1
                        if halt:
                            raise

                if dithonly:
                    continue
                if tellstars and plate_type in {"single", "normal"}:
                    backend.write_tellstar_summary(plan, tellstars)
                minframes = 1 if nodither else 2
                if len(allframes) < minframes:
                    raise PlanFailure(
                        f"only {len(allframes)} good frame(s); need {minframes}"
                    )
                if telluric_errors and halt:
                    raise PlanFailure(
                        f"{telluric_errors} frame(s) had telluric errors"
                    )
                combined, pairs = backend.combine_dithers(allframes,shifts,
                                                          plugmap=plugmap,
                                                          nodither=nodither)
                if pairs is None and not nodither:
                    raise PlanFailure("no dither pairs")
                final = backend.flux_calibrate(combined, plugmap=plugmap)
                backend.write_visit_products(final, plan=plan, plugmap=plugmap,
                                             shifts=shifts, pairs=pairs,
                                             survey=survey, relflux=relflux)
                source_frame = final

            if plate_type not in {"normal", "single"}:
                continue
            object_indices = select_visit_objects(fiberdata, fps=fps)
            summary = backend.visit_summary_file(plan, plugmap)
            outcome.visit_summary_file = summary
            if Path(summary).exists() and not clobber:
                continue
            rows = backend.build_visit_rows(plan=plan, plugmap=plugmap,
                                            object_indices=object_indices,
                                            survey=survey, nframes=nframes,
                                            relflux=relflux)
            backend.write_visit_summary(summary, rows, plan=plan,
                                        source_frame=source_frame)
            if object_indices.size:
                backend.ingest_visits(rows)
        except Exception as exc:
            outcome.errors.append(str(exc))
            log.exception("Failed plan %s", planfile)
            if halt:
                raise

    log.info("AP1DVISIT finished in %.1f s", time.monotonic() - started)
    return results

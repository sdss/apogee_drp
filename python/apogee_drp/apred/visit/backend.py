"""Native numerical backend methods for :func:`ap1dvisit`.

``NativeVisitBackendMixin`` deliberately implements the portable reduction
stages only. A site backend subclasses it and supplies plan parsing, SDSS file
resolution, calibration discovery, plug-map loading, and summary/database I/O.
This keeps filesystem policy separate from the translated IDL numerics.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .combine import dither_combine
from .flux import flux_calibrate as native_flux_calibrate
from .io import BADERR, read_cframes, write_cframes as native_write_cframes
from .products import write_visit_products as native_write_visit_products
from .shift import dither_shift as native_dither_shift


_MISSING = object()


def _get(value: Any, name: str, default: Any = _MISSING) -> Any:
    """Retrieve a field case-insensitively from common table-like objects."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() == name.lower():
                return item
    names = getattr(value, "colnames", None)
    if names is None:
        names = getattr(getattr(value, "dtype", None), "names", None)
    if names is not None:
        for key in names:
            if str(key).lower() == name.lower():
                return value[key]
    for candidate in (name, name.lower(), name.upper()):
        try:
            return value[candidate]
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    for candidate in (name, name.lower(), name.upper()):
        if hasattr(value, candidate):
            return getattr(value, candidate)
    if default is not _MISSING:
        return default
    raise KeyError(name)


def _set(value: Any, name: str, item: Any) -> None:
    if isinstance(value, dict):
        for key in value:
            if str(key).lower() == name.lower():
                value[key] = item
                return
        value[name] = item
    else:
        setattr(value, name, item)


def _chip(frame: Any, index: int) -> Any:
    return _get(frame, f"chip{'abc'[index]}")


class NativeVisitBackendMixin:
    """Concrete implementations of the translated visit numerical stages.

    Required plan keys for the native wrappers are:

    - ``telluric_files``: three preconvolved calibration filenames, unless
      ``telluric_models`` already contains the sampled model array;
    - ``plate_files`` and ``visit_directory`` for product writing.

    A subclass may override :meth:`resolve_telluric_files` and
    :meth:`visit_product_options` to use the standard SDSS tree instead.
    """

    def load_frame(self, files: Sequence[str], *, kind: str) -> Any:
        if kind.lower() == "cframe":
            return read_cframes(files)
        raise NotImplementedError(
            "site backend must implement native ap1D loading"
        )

    def header_value(
        self, frame: Any, keyword: str, default: Any = None, chip: int = 0
    ) -> Any:
        header = _get(_chip(frame, chip), "header")
        try:
            return header.get(keyword, default)
        except AttributeError:
            return _get(header, keyword, default)

    def add_header(
        self,
        frame: Any,
        keyword: str,
        value: Any,
        comment: str | None = None,
    ) -> None:
        for index in range(3):
            header = _get(_chip(frame, index), "header")
            header[keyword] = (value, comment) if comment else value

    def add_history(self, frame: Any, text: str) -> None:
        for index in range(3):
            header = _get(_chip(frame, index), "header")
            try:
                header.add_history(text)
            except AttributeError:
                history = header.setdefault("HISTORY", [])
                if isinstance(history, str):
                    history = [history]
                    header["HISTORY"] = history
                history.append(text)

    def chip_pixel_arrays(
        self, frame: Any, chip: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = _chip(frame, chip)
        error = _get(data, "err", _MISSING)
        if error is _MISSING:
            error = _get(data, "error")
        return (
            np.asarray(_get(data, "flux")),
            np.asarray(error),
            np.asarray(_get(data, "mask")),
        )

    def chip_arrays(
        self, frame: Any, chip: int
    ) -> tuple[np.ndarray, np.ndarray]:
        data = _chip(frame, chip)
        return np.asarray(_get(data, "flux")), np.asarray(_get(data, "err"))

    def dither_shift(
        self,
        reference: Any,
        frame: Any,
        *,
        plugmap: Any,
        plan: Mapping[str, Any],
        plotfile: str,
        nofit: bool,
    ) -> dict[str, Any]:
        mode = str(plan.get("dither_shift_mode", "lines")).lower()
        result = native_dither_shift(reference, frame,
                                     xcorr=mode in {"xcorr", "correlation"},
                                     object_spectra=bool(plan.get("dither_object_spectra",
                                                                  False)), plugmap=plugmap, nofit=nofit,
                                     mjd=int(plan.get("mjd", 999999)))
        return asdict(result)

    def wavelength_calibrate(self, frame: Any, **kwargs: Any) -> Any:
        """Use wavelength calibration already written into the ap1D files.

        Current production reductions run the Python ``bin/ap1dwavecal``
        command before ``ap1dvisit``.  This compatibility hook therefore
        validates the wavelength products instead of rerunning the retired
        IDL visit-stage routine.
        """

        for index in range(3):
            chip = _chip(frame, index)
            flux = np.asarray(_get(chip, "flux"))
            wavelength = np.asarray(_get(chip, "wavelength"))
            wcoef = np.asarray(_get(chip, "wcoef"))
            if wavelength.shape != flux.shape:
                raise ValueError(
                    "ap1D wavelength array must match [fiber,pixel] flux shape"
                )
            if wcoef.ndim != 2 or wcoef.shape[0] != flux.shape[0]:
                raise ValueError(
                    "ap1D wavelength coefficients must have one row per fiber"
                )
        return frame

    def sky_subtract(
            self, frame: Any, *, plugmap: Any, force: bool
    ) -> tuple[Any, Any]:
        from ..sky.subtract import sky_subtract as native_sky_subtract
        return native_sky_subtract(frame, plugmap, force=force, return_metrics=True)

    
    def resolve_telluric_files(
        self, frame: Any, plan: Mapping[str, Any]
    ) -> Sequence[str]:
        files = plan.get("telluric_files")
        if files is None:
            raise ValueError(
                "plan requires telluric_files or pre-sampled telluric_models"
            )
        if len(files) != 3:
            raise ValueError("telluric_files must contain three chip files")
        return files

    def telluric_correct(
        self,
        frame: Any,
        *,
        plugmap: Any,
        plan: Mapping[str, Any],
        plots_dir: str,
        test: bool,
        force: bool,
    ) -> tuple[Any, Any]:
        from ..sky.telluric import (
            load_preconvolved_telluric,
            telluric_correct_frame,
        )

        models = plan.get("telluric_models")
        if models is None:
            models = load_preconvolved_telluric(
                self.resolve_telluric_files(frame, plan), frame
            )
        result = telluric_correct_frame(
            frame,
            plugmap,
            models,
            starfitopt=int(plan.get("starfitopt", 1)),
            force=force,
            single=str(plan.get("platetype", "")).lower() == "single",
            telescope=str(plan.get("telescope", "apo25m")),
        )
        return result.frame, result.tellstar

    def write_cframes(
        self, frame: Any, plugmap: Any, files: Sequence[str]
    ) -> None:
        native_write_cframes(
            frame,
            plugmap,
            files,
            pipeline_version=getattr(self, "pipeline_version", None),
        )

    def shift_result(self, frame: Any) -> Mapping[str, Any]:
        shift = _get(frame, "shift", {})
        if isinstance(shift, Mapping):
            return shift
        if hasattr(shift, "__dataclass_fields__"):
            return asdict(shift)
        return {
            name: getattr(shift, name)
            for name in ("shiftfit", "shifterr", "chipshift", "chipfit")
            if hasattr(shift, name)
        }

    def combine_dithers(
        self,
        frames: Sequence[Any],
        shifts: Sequence[Any],
        *,
        plugmap: Any,
        nodither: bool,
    ) -> tuple[Any, Any]:
        fiberdata = _get(plugmap, "fiberdata")
        fiberids = np.asarray(_get(fiberdata, "fiberid"), dtype=int)
        objtypes = np.asarray(_get(fiberdata, "objtype")).astype(str)
        nfibers = np.asarray(_get(_chip(frames[0], 0), "flux")).shape[0]
        fiber_types = np.full(nfibers, "", dtype=object)
        rows = 300 - fiberids
        valid = (rows >= 0) & (rows < nfibers)
        fiber_types[rows[valid]] = objtypes[valid]
        return dither_combine(list(frames),shifts,fiber_types=fiber_types,no_dither=nodither)

    def flux_calibrate(self, frame: Any, *, plugmap: Any) -> Any:
        return native_flux_calibrate(frame, plugmap)

    def visit_product_options(
        self, plan: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "plate_files": plan.get("plate_files"),
            "visit_directory": plan["visit_directory"],
            "single": str(plan.get("platetype", "")).lower() == "single",
            "mjdfrac": plan.get("mjdfrac"),
            "telescope": plan.get("telescope", "apo25m"),
            "reduction_version": plan.get("reduction_version", ""),
            "software_version": plan.get("software_version", ""),
        }

    def write_visit_products(
        self,
        frame: Any,
        *,
        plan: Mapping[str, Any],
        plugmap: Any,
        shifts: Sequence[Any],
        pairs: Any,
        survey: str,
        relflux: np.ndarray,
    ) -> Any:
        options = self.visit_product_options(plan)
        return native_write_visit_products(
            frame,
            plugmap,
            shifts,
            pairs,
            survey=survey,
            relflux=relflux,
            **options,
        )

    def fiber_data(self, plugmap: Any) -> Any:
        return _get(plugmap, "fiberdata")

    def set_plugmap_mjd(self, plugmap: Any, mjd: int) -> None:
        _set(plugmap, "mjd", int(mjd))

    @property
    def bad_error(self) -> float:
        return float(BADERR)

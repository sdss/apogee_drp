"""FITS I/O used by the Python ``ap1dvisit`` backend.

This is a faithful translation of ``apvisit_outcframe.pro``.  Internal spectral
arrays use ``(nfiber, npix)`` so they can be written directly with Astropy:
FITS NAXIS1 is then pixel and NAXIS2 is fiber, matching the IDL product.
"""

from __future__ import annotations

import copy
import getpass
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .models import ChipFrame, VisitFrame

BADERR = np.float32(1.0e10)

IMAGE_EXTENSIONS = (
    ("flux", np.float32, "FLUX", "Flux (ADU)"),
    ("err", np.float32, "ERROR", "Flux Error (ADU)"),
    ("mask", np.int16, "MASK", "Flag Mask (bitwise)"),
    ("wavelength", np.float64, "WAVELENGTH", "Wavelength (Ang)"),
    ("sky", np.float32, "SKY FLUX", "Sky (ADU)"),
    ("skyerr", np.float32, "SKY ERROR", "Sky Error (ADU)"),
    ("telluric", np.float32, "TELLURIC", "Telluric"),
    ("telluricerr", np.float32, "TELLURIC ERROR", "Telluric Error"),
    (
        "wcoef",
        np.float64,
        "WAVE COEFFICIENTS",
        "Wavelength Coefficients",
    ),
    ("lsfcoef", np.float64, "LSF COEFFICIENTS", "LSF Coefficients"),
)

_OUTPUT_HISTORY = (
    "Output File:",
    " HDU0 - Header only",
    " HDU1 - Flux (ADU)",
    " HDU2 - Error (ADU)",
    " HDU3 - flag mask (bitwise OR combined)",
    "        1 - bad pixels",
    "        2 - cosmic ray",
    "        4 - saturated",
    "        8 - unfixable",
    " HDU4 - Wavelength (Ang)",
    " HDU5 - Sky (ADU)",
    " HDU6 - Sky Error (ADU)",
    " HDU7 - Telluric",
    " HDU8 - Telluric Error",
    " HDU9 - Wavelength coefficients",
    " HDU10 - LSF coefficients",
    " HDU11 - Plugmap structure",
    " HDU12 - Plugmap header",
    " HDU13 - Telluric structure",
    " HDU14 - Shift structure",
)


def _get(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        if name in container:
            return container[name]
        lower = name.lower()
        for key in container:
            if str(key).lower() == lower:
                return container[key]
    if hasattr(container, name):
        return getattr(container, name)
    lower = name.lower()
    for candidate in dir(container):
        if candidate.lower() == lower:
            return getattr(container, candidate)
    raise KeyError(name)


def _chip(frame: Any, index: int) -> Any:
    for key in (f"chip{'abc'[index]}", f"CHIP{'ABC'[index]}"):
        try:
            return _get(frame, key)
        except KeyError:
            pass
    try:
        return frame[index]
    except (IndexError, KeyError, TypeError):
        raise KeyError(f"chip {'abc'[index]}") from None


def _as_header(value: Any) -> fits.Header:
    if isinstance(value, fits.Header):
        return value.copy()
    if isinstance(value, Mapping):
        header = fits.Header()
        for key, item in value.items():
            if str(key).upper() in {"COMMENT", "HISTORY"}:
                for line in np.atleast_1d(item):
                    header[str(key).upper()] = str(line)
            elif item is not None:
                header[str(key).upper()] = item
        return header
    return fits.Header.fromstring(str(value), sep="\n")


def _clean_error(error: Any) -> np.ndarray:
    out = np.asarray(error, dtype=np.float32).copy()
    bad = ~np.isfinite(out) | (out <= 0) | (out == BADERR)
    out[bad] = BADERR
    return out


def _image_hdu(
    data: Any,
    *,
    dtype: Any,
    extname: str,
    bunit: str,
) -> fits.ImageHDU:
    array = np.asarray(data, dtype=dtype)
    hdu = fits.ImageHDU(array, name=extname)
    hdu.header["CTYPE1"] = "Pixel"
    hdu.header["CTYPE2"] = "Fiber"
    hdu.header["BUNIT"] = bunit
    if extname == "MASK":
        hdu.header.add_history("Explanation of BITWISE flag mask (OR combined)")
        for line in (
            " 1 - bad pixels",
            " 2 - cosmic ray",
            " 4 - saturated",
            " 8 - unfixable",
        ):
            hdu.header.add_history(line)
    elif extname == "WAVE COEFFICIENTS":
        for line in (
            "Wavelength Coefficients to be used with PIX2WAVE.PRO:",
            " 1 Global additive pixel offset",
            " 4 Sine Parameters",
            " 7 Polynomial parameters",
        ):
            hdu.header.add_history(line)
    elif extname == "LSF COEFFICIENTS":
        hdu.header.add_history(
            "LSF Coefficients to be used with LSF_GH.PRO"
        )
    return hdu


def _table_hdu(value: Any, name: str) -> fits.BinTableHDU:
    if isinstance(value, fits.BinTableHDU):
        hdu = value.copy()
        hdu.name = name
        return hdu
    if isinstance(value, Table):
        table = value
    elif isinstance(value, np.ndarray) and value.dtype.names:
        table = Table(value)
    elif isinstance(value, Mapping):
        columns = {}
        for key, item in value.items():
            array = np.asarray(item)
            if array.ndim == 0:
                array = array.reshape(1)
            columns[str(key)] = array
        table = Table(columns)
    elif hasattr(value, "__dataclass_fields__"):
        table = Table(
            {key: [getattr(value, key)] for key in value.__dataclass_fields__}
        )
    else:
        table = Table(rows=np.atleast_1d(value))
    return fits.BinTableHDU(table, name=name)


def _plugmap_metadata(plugmap: Any) -> dict[str, Any]:
    if not isinstance(plugmap, Mapping):
        values = vars(plugmap)
    else:
        values = plugmap
    output: dict[str, Any] = {}
    for key, value in values.items():
        if str(key).lower() in {"fiberdata", "guidedata"}:
            continue
        array = np.asarray(value)
        if array.ndim <= 1:
            output[str(key)] = array.reshape(1, -1) if array.ndim == 1 else [value]
    return output


def write_cframes(
    frame: Any,
    plugmap: Any,
    outfiles: Sequence[str | Path],
    *,
    overwrite: bool = True,
    pipeline_version: str | None = None,
) -> list[str]:
    """Write the three 15-HDU apCframe files.

    Parameters match the IDL routine, with explicit output filenames.  Image
    dtypes and HDU ordering are fixed to the legacy data model.
    """

    if len(outfiles) != 3:
        raise ValueError("outfiles must contain exactly three chip filenames")
    tellstar = _get(frame, "tellstar")
    shift = _get(frame, "shift")
    fiberdata = _get(plugmap, "fiberdata")
    plugmeta = _plugmap_metadata(plugmap)
    written: list[str] = []

    for index, filename in enumerate(outfiles):
        chip = _chip(frame, index)
        for field, *_ in IMAGE_EXTENSIONS:
            _get(chip, field)

        header = _as_header(_get(chip, "header"))
        lead = "AP1DVISIT: "
        header.add_history(
            lead + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        header.add_history(lead + f"{getpass.getuser()} on {platform.node()}")
        header.add_history(lead + f"Python {platform.python_version()}")
        if pipeline_version:
            header.add_history(
                lead + " APOGEE Reduction Pipeline Version: " + pipeline_version
            )
        for line in _OUTPUT_HISTORY:
            header.add_history(lead + line)

        hdus: list[fits.hdu.base.ExtensionHDU] = [fits.PrimaryHDU(header=header)]
        for field, dtype, extname, bunit in IMAGE_EXTENSIONS:
            data = _get(chip, field)
            if field == "err":
                data = _clean_error(data)
            hdus.append(
                _image_hdu(data, dtype=dtype, extname=extname, bunit=bunit)
            )
        hdus.append(_table_hdu(fiberdata, "PLUGMAP"))
        hdus.append(_table_hdu(plugmeta, "PLUGMAP HEADER"))
        hdus.append(_table_hdu(tellstar, "TELLURIC"))
        hdus.append(_table_hdu(shift, "SHIFT"))

        path = str(filename)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(hdus).writeto(
            path, overwrite=overwrite, output_verify="silentfix"
        )
        written.append(path)
    return written


def read_cframes(files: Sequence[str | Path]) -> VisitFrame:
    """Read three files written by :func:`write_cframes`."""

    if len(files) != 3:
        raise ValueError("files must contain exactly three chip filenames")
    chips = []
    tellstar = None
    shift = None
    plugmap = None
    for index, filename in enumerate(files):
        with fits.open(filename, memmap=False) as hdul:
            values: dict[str, Any] = {"filename": str(filename),
                                      "header": hdul[0].header.copy()}
            for hdu_index, (field, dtype, _, _) in enumerate(
                IMAGE_EXTENSIONS, start=1
            ):
                values[field] = np.asarray(
                    hdul[hdu_index].data, dtype=dtype
                ).copy()
            chips.append(ChipFrame(**values))
            if index == 0:
                plugmap = {
                    "fiberdata": Table(hdul[11].data),
                    "metadata": Table(hdul[12].data),
                }
                tellstar = Table(hdul[13].data)
                shift = Table(hdul[14].data)
    frame = VisitFrame(*chips, tellstar=tellstar, shift=shift)
    frame.metadata["plugmap"] = plugmap
    return frame

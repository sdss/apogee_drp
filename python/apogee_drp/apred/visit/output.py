"""apPlate and apVisit products translated from ``apvisit_output.pro``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ...utils.bitmask import PixelBitMask, StarBitMask
from .io import (
    BADERR,
    _as_header,
    _chip,
    _get,
    _plugmap_metadata,
    _table_hdu,
)

FLUX_SCALE = 1.0e-17
BADMASK = 16639
PIXEL_BITS = PixelBitMask()
STAR_BITS = StarBitMask()


@dataclass
class VisitProductResult:
    plate_files: list[str]
    visit_files: list[str]
    skipped_fibers: list[int]


def _optional(value: Any, name: str, default: Any = None) -> Any:
    try:
        return _get(value, name)
    except (KeyError, AttributeError):
        return default


def _column(fiberdata: Any, name: str, default: Any = None) -> np.ndarray:
    value = _optional(fiberdata, name, default)
    if value is None:
        raise ValueError(f"required plugmap column {name!r} is missing")
    return np.asarray(value)


def _validate_frame(frame: Any) -> tuple[list[Any], int, int]:
    chips = [_chip(frame, index) for index in range(3)]
    required = (
        "header",
        "flux",
        "err",
        "mask",
        "wavelength",
        "sky",
        "skyerr",
        "telluric",
        "telluricerr",
        "lsfcoef",
        "wcoef",
    )
    shape = None
    for chip in chips:
        for name in required:
            _get(chip, name)
        flux = np.asarray(_get(chip, "flux"))
        if flux.ndim != 2:
            raise ValueError("flux must have [fiber, pixel] shape")
        if shape is None:
            shape = flux.shape
        elif flux.shape != shape:
            raise ValueError("all chip flux arrays must have the same shape")
        for name in (
            "err",
            "mask",
            "wavelength",
            "sky",
            "skyerr",
            "telluric",
            "telluricerr",
        ):
            if np.asarray(_get(chip, name)).shape != flux.shape:
                raise ValueError(f"{name} must have the same shape as flux")
    assert shape is not None
    return chips, shape[0], shape[1]


def _image_hdu(
    data: Any,
    extname: str,
    unit: str,
    *,
    axis2: str,
    dtype: Any,
) -> fits.ImageHDU:
    hdu = fits.ImageHDU(np.asarray(data, dtype=dtype), name=extname)
    hdu.header["CTYPE1"] = "Pixel"
    hdu.header["CTYPE2"] = axis2
    hdu.header["BUNIT"] = unit
    return hdu


def _error_array(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bad = (values == BADERR) | ~np.isfinite(values)
    output = values / np.float32(FLUX_SCALE)
    output[bad] = BADERR
    return output


def _flux_array(values: Any) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(values, dtype=np.float32) / np.float32(FLUX_SCALE)
    bad = ~np.isfinite(output)
    output[bad] = 0.0
    return output, bad


def _coefficient_hdu(data: Any, extname: str, unit: str) -> fits.ImageHDU:
    hdu = fits.ImageHDU(np.asarray(data, dtype=np.float64), name=extname)
    hdu.header["CTYPE1"] = "Fiber"
    hdu.header["CTYPE2"] = "Parameters"
    hdu.header["BUNIT"] = unit
    return hdu


def _primary_header(
    source: Any,
    *,
    plate: Any,
    mjd: int,
    locid: Any,
    telescope: str,
    reduction_version: str,
    software_version: str,
) -> fits.Header:
    header = _as_header(source)
    # IDL removes inherited AP2D/AP3D history from the plate products.
    retained = []
    for card in header.cards:
        if card.keyword == "HISTORY" and (
            str(card.value).startswith("AP2D:")
            or str(card.value).startswith("AP3D:")
        ):
            continue
        retained.append(card)
    header = fits.Header(retained)
    header["V_APRED"] = software_version
    header["APRED"] = reduction_version
    header["PLATE"] = plate
    header["MJD5"] = int(mjd)
    header["LOCID"] = locid
    header["TELESCOP"] = telescope
    header.add_history("AP1DVISIT: Python output")
    return header


def _plugmap_header_hdu(plugmap: Any) -> fits.BinTableHDU:
    return _table_hdu(_plugmap_metadata(plugmap), "PLUGMAP HEADER")


def _structure_hdu(value: Any, name: str) -> fits.BinTableHDU:
    if value is None:
        return fits.BinTableHDU(Table(), name=name)
    return _table_hdu(value, name)


def write_plate_products(
    frame: Any,
    plugmap: Any,
    shiftstr: Any,
    pairstr: Any,
    files: Sequence[str | Path],
    *,
    telescope: str = "apo25m",
    reduction_version: str = "",
    software_version: str = "",
    overwrite: bool = True,
) -> list[str]:
    """Write the three apPlate files and their legacy HDU sequence."""

    if len(files) != 3:
        raise ValueError("files must contain one apPlate filename per chip")
    chips, _, _ = _validate_frame(frame)
    plate = _optional(plugmap, "plateid", _optional(plugmap, "plate"))
    mjd = int(_get(plugmap, "mjd"))
    locid = _optional(plugmap, "locationid", -1)
    fiberdata = _get(plugmap, "fiberdata")
    fluxcorr = _optional(frame, "fluxcorr")
    written: list[str] = []

    for chip, filename in zip(chips, files):
        primary = _primary_header(
            _get(chip, "header"),
            plate=plate,
            mjd=mjd,
            locid=locid,
            telescope=telescope,
            reduction_version=reduction_version,
            software_version=software_version,
        )
        flux, nonfinite = _flux_array(_get(chip, "flux"))
        mask = np.asarray(_get(chip, "mask"), dtype=np.int16).copy()
        mask[nonfinite] |= np.int16(PIXEL_BITS.getval("BADPIX"))
        sky = np.asarray(_get(chip, "sky"), dtype=np.float32) / np.float32(
            FLUX_SCALE
        )
        skyerr = _error_array(_get(chip, "skyerr"))
        hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [
            fits.PrimaryHDU(header=primary),
            _image_hdu(
                flux,
                "FLUX",
                "Flux (10^-17 ergs/s/cm^2/Ang)",
                axis2="Fiber",
                dtype=np.float32,
            ),
            _image_hdu(
                _error_array(_get(chip, "err")),
                "ERROR",
                "Flux Error (10^-17 ergs/s/cm^2/Ang)",
                axis2="Fiber",
                dtype=np.float32,
            ),
            _image_hdu(
                mask,
                "MASK",
                "Flag Mask (bitwise)",
                axis2="Fiber",
                dtype=np.int16,
            ),
            _image_hdu(
                _get(chip, "wavelength"),
                "WAVELENGTH",
                "Wavelength (Ang)",
                axis2="Fiber",
                dtype=np.float64,
            ),
            _image_hdu(
                sky,
                "SKY FLUX",
                "Sky (10^-17 ergs/s/cm^2/Ang)",
                axis2="Fiber",
                dtype=np.float32,
            ),
            _image_hdu(
                skyerr,
                "SKY ERROR",
                "Sky Error (10^-17 ergs/s/cm^2/Ang)",
                axis2="Fiber",
                dtype=np.float32,
            ),
            _image_hdu(
                _get(chip, "telluric"),
                "TELLURIC",
                "Telluric",
                axis2="Fiber",
                dtype=np.float32,
            ),
            _image_hdu(
                _get(chip, "telluricerr"),
                "TELLURIC ERROR",
                "Telluric Error",
                axis2="Fiber",
                dtype=np.float32,
            ),
            _coefficient_hdu(
                _get(chip, "wcoef"),
                "WAVE COEFFICIENTS",
                "Wavelength Coefficients",
            ),
            _coefficient_hdu(
                _get(chip, "lsfcoef"), "LSF COEFFICIENTS", "LSF Coefficients"
            ),
            _table_hdu(fiberdata, "PLUGMAP"),
            _plugmap_header_hdu(plugmap),
            _structure_hdu(shiftstr, "SHIFT"),
            _structure_hdu(pairstr, "PAIR"),
        ]
        if fluxcorr is not None:
            hdus.append(
                fits.ImageHDU(
                    np.asarray(fluxcorr, dtype=np.float32), name="FLUX CONVERSION"
                )
            )
        output = Path(filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(hdus).writeto(output, overwrite=overwrite)
        written.append(str(output))
    return written


def _fiber_value(fiberdata: Any, name: str, index: int, default: Any = None) -> Any:
    values = _optional(fiberdata, name)
    if values is None:
        return default
    value = np.asarray(values)[index]
    return value.item() if np.ndim(value) == 0 and hasattr(value, "item") else value


def _magnitude(fiberdata: Any, index: int, band: int) -> float:
    direct = ("jmag", "hmag", "kmag")[band]
    value = _fiber_value(fiberdata, direct, index)
    if value is not None:
        return float(value)
    magnitudes = np.asarray(_get(fiberdata, "mag"))
    return float(magnitudes[index, band])


def _neighbor_delta(
    fiberdata: Any,
    spectrograph: np.ndarray,
    fiberids: np.ndarray,
    index: int,
    neighbor_id: int,
) -> float:
    found = np.flatnonzero((spectrograph == 2) & (fiberids == neighbor_id))
    if found.size == 0:
        return 99.99
    neighbor_h = _magnitude(fiberdata, int(found[0]), 1)
    return 99.99 if neighbor_h < 0.01 else neighbor_h - _magnitude(fiberdata, index, 1)


def _visit_flag(
    flux: np.ndarray,
    error: np.ndarray,
    mask: np.ndarray,
    *,
    mjd: int,
    hplus: float,
    hminus: float,
    mtpflux: float | None,
) -> tuple[int, float, np.ndarray]:
    middle_good = (
        (flux[1] > 0)
        & (error[1] > 0)
        & ((mask[1].astype(np.int64) & BADMASK) == 0)
    )
    snr = (
        float(np.median(flux[1, middle_good] / error[1, middle_good]))
        if np.any(middle_good)
        else 0.0
    )
    flag = 0
    if snr < 5:
        flag |= int(STAR_BITS.getval("LOW_SNR"))
    if mjd <= 55761:
        flag |= int(STAR_BITS.getval("COMMISSIONING"))
    if min(hplus, hminus) < -5:
        flag |= int(STAR_BITS.getval("VERY_BRIGHT_NEIGHBOR"))
    elif min(hplus, hminus) < -2.5:
        flag |= int(STAR_BITS.getval("BRIGHT_NEIGHBOR"))
    if mtpflux is not None:
        if mtpflux < 0.75:
            flag |= int(STAR_BITS.getval("MTPFLUX_LT_75"))
        if mtpflux < 0.5:
            flag |= int(STAR_BITS.getval("MTPFLUX_LT_50"))

    persistence = (
        ("PERSIST_HIGH", "PERSIST_HIGH"),
        ("PERSIST_MED", "PERSIST_MED"),
        ("PERSIST_LOW", "PERSIST_LOW"),
    )
    cumulative = np.zeros(mask.shape, dtype=bool)
    for pixel_name, star_name in persistence:
        cumulative |= (
            mask.astype(np.int64) & int(PIXEL_BITS.getval(pixel_name))
        ) > 0
        if np.count_nonzero(cumulative) / mask.size > 0.2:
            flag |= int(STAR_BITS.getval(star_name))
            break

    blue_median = np.median(flux[2])
    green_median = np.median(flux[1])
    expected_ratio = (1.55 / 1.62) ** -4
    if blue_median > 1.5 * green_median * expected_ratio:
        flag |= int(STAR_BITS.getval("PERSIST_JUMP_POS"))
    if blue_median < 0.667 * green_median * expected_ratio:
        flag |= int(STAR_BITS.getval("PERSIST_JUMP_NEG"))
    if flag & int(STAR_BITS.getval("PERSIST_JUMP_POS")):
        persistence_pixels = np.zeros(mask.shape, dtype=bool)
        for name in ("PERSIST_LOW", "PERSIST_MED", "PERSIST_HIGH"):
            persistence_pixels |= (
                mask.astype(np.int64) & int(PIXEL_BITS.getval(name))
            ) > 0
        mask[persistence_pixels] |= np.int16(PIXEL_BITS.getval("BADPIX"))
    mask[~np.isfinite(flux)] |= np.int16(PIXEL_BITS.getval("BADPIX"))
    if np.count_nonzero(mask.astype(np.int64) & BADMASK) / mask.size > 0.2:
        flag |= int(STAR_BITS.getval("BAD_PIXELS"))
    return flag, snr, mask


def _relative_fluxes(relflux: Any | None, nfibers: int) -> tuple[np.ndarray, np.ndarray] | None:
    if relflux is None:
        return None
    relative = np.asarray(relflux, dtype=np.float64).copy()
    if relative.shape != (nfibers,):
        raise ValueError("relflux must contain one value per detector fiber")
    mtp = relative.copy()
    for start in range(0, nfibers, 30):
        mtp[start : start + 30] = np.median(relative[start : start + 30])
    relative /= np.max(relative)
    mtp /= np.max(mtp)
    return relative, mtp


def build_visit_hdul(
    frame: Any,
    plugmap: Any,
    fiber_row: int,
    *,
    survey: str | None = None,
    relflux: Any | None = None,
    telescope: str = "apo25m",
    reduction_version: str = "",
    software_version: str = "",
) -> fits.HDUList:
    """Build one in-memory apVisit file for a detector-row index."""

    chips, nfibers, _ = _validate_frame(frame)
    if not 0 <= fiber_row < nfibers:
        raise IndexError("fiber_row is outside the frame")
    fiberdata = _get(plugmap, "fiberdata")
    fiberids = _column(fiberdata, "fiberid").astype(int)
    spectrograph = _column(fiberdata, "spectrographid").astype(int)
    fiber_id = 300 - fiber_row
    match = np.flatnonzero((spectrograph == 2) & (fiberids == fiber_id))
    if match.size == 0:
        raise ValueError(f"fiber {fiber_id} has no spectrograph-2 plugmap row")
    index = int(match[0])
    plate = _optional(plugmap, "plateid", _optional(plugmap, "plate"))
    mjd = int(_get(plugmap, "mjd"))
    locid = _optional(plugmap, "locationid", -1)
    hplus = _neighbor_delta(fiberdata, spectrograph, fiberids, index, fiber_id + 1)
    hminus = _neighbor_delta(fiberdata, spectrograph, fiberids, index, fiber_id - 1)
    relative = _relative_fluxes(relflux, nfibers)
    this_mtp = None if relative is None else float(relative[1][fiber_row])

    raw_flux = np.stack(
        [np.asarray(_get(chip, "flux"))[fiber_row] for chip in chips]
    ).astype(np.float32)
    raw_error = np.stack(
        [np.asarray(_get(chip, "err"))[fiber_row] for chip in chips]
    ).astype(np.float32)
    mask = np.stack(
        [np.asarray(_get(chip, "mask"))[fiber_row] for chip in chips]
    ).astype(np.int16)
    visit_flag, snr, mask = _visit_flag(
        raw_flux.copy(),
        raw_error,
        mask,
        mjd=mjd,
        hplus=hplus,
        hminus=hminus,
        mtpflux=this_mtp,
    )

    source_header = _as_header(_get(chips[0], "header"))
    header = fits.Header()
    for key in ("DATE-OBS", "EXPTIME", "JD-MID", "UT-MID", "NPAIRS"):
        if key in source_header:
            header[key] = source_header[key]
    ncombine = int(source_header.get("NCOMBINE", 1))
    header["NCOMBINE"] = ncombine
    for number in range(1, ncombine + 1):
        key = f"FRAME{number}"
        if key in source_header:
            header[key] = source_header[key]
    header["LOCID"] = locid
    header["PLATE"] = plate
    header["TELESCOP"] = telescope
    header["MJD5"] = mjd
    header["FIBERID"] = fiber_id
    header["OBJID"] = str(_fiber_value(fiberdata, "tmass_style", index, "")).strip()
    header["OBJTYPE"] = str(_fiber_value(fiberdata, "objtype", index, "")).strip()
    header["RA"] = float(_fiber_value(fiberdata, "ra", index, np.nan))
    header["DEC"] = float(_fiber_value(fiberdata, "dec", index, np.nan))
    header["JMAG"] = _magnitude(fiberdata, index, 0)
    header["HMAG"] = _magnitude(fiberdata, index, 1)
    header["KMAG"] = _magnitude(fiberdata, index, 2)
    header["SNR"] = snr
    if survey is not None:
        header["SURVEY"] = str(survey).strip()
    for output_key, source_key in (
        ("TARG1", "target1"),
        ("TARG2", "target2"),
        ("TARG3", "target3"),
        ("TARG4", "target4"),
        ("SVAPTRG0", "sdssv_apogee_target0"),
        ("CATID", "catalogid"),
        ("SDSSID", "sdss_id"),
        ("GRELEASE", "gaia_release"),
        ("GSRCID", "gaia_sourceid"),
        ("GPLX", "gaia_plx"),
        ("GPMRA", "gaia_pmra"),
        ("GPMDEC", "gaia_pmdec"),
        ("GGMAG", "gaia_gmag"),
        ("GBPMAG", "gaia_bpmag"),
        ("GRPMAG", "gaia_rpmag"),
    ):
        value = _fiber_value(fiberdata, source_key, index)
        if value is not None:
            header[output_key] = value
    header["HPLUS"] = hplus
    header["HMINUS"] = hminus
    if mjd >= 59556:
        for output_key, source_key in (
            ("ASSIGNED", "assigned"),
            ("ONTARGET", "on_target"),
            ("VALID", "valid"),
            ("SVTARG0", "sdssv_apogee_target0"),
            ("CARTON1", "firstcarton"),
            ("CADENCE", "cadence"),
            ("PROGRAM", "program"),
            ("CATEGORY", "category"),
        ):
            value = _fiber_value(fiberdata, source_key, index)
            if value is not None:
                header[output_key] = value
    if relative is not None:
        header["RELFLUX"] = float(relative[0][fiber_row])
        header["MTPFLUX"] = float(relative[1][fiber_row])
    header["VISITFLG"] = visit_flag
    fluxcorr = _optional(frame, "fluxcorr")
    if fluxcorr is not None:
        array = np.asarray(fluxcorr)
        header["FLUXFLAM"] = (
            float(array[fiber_row])
            if array.ndim == 1
            else float(np.median(array[..., fiber_row]))
        )
    header["V_APRED"] = software_version
    header["APRED"] = reduction_version
    header.add_history("AP1DVISIT: Python individual visit product")

    flux, _ = _flux_array(raw_flux)
    wave = np.stack(
        [np.asarray(_get(chip, "wavelength"))[fiber_row] for chip in chips]
    )
    sky = np.stack(
        [np.asarray(_get(chip, "sky"))[fiber_row] for chip in chips]
    ) / np.float32(FLUX_SCALE)
    skyerr = np.stack(
        [np.asarray(_get(chip, "skyerr"))[fiber_row] for chip in chips]
    ) / np.float32(FLUX_SCALE)
    telluric = np.stack(
        [np.asarray(_get(chip, "telluric"))[fiber_row] for chip in chips]
    )
    telluricerr = np.stack(
        [np.asarray(_get(chip, "telluricerr"))[fiber_row] for chip in chips]
    )
    wcoef = np.stack(
        [np.asarray(_get(chip, "wcoef"))[fiber_row] for chip in chips]
    )
    lsfcoef = np.stack(
        [np.asarray(_get(chip, "lsfcoef"))[fiber_row] for chip in chips]
    )
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [
        fits.PrimaryHDU(header=header),
        _image_hdu(
            flux,
            "FLUX",
            "Flux (10^-17 erg/s/cm^2/Ang)",
            axis2="Chip",
            dtype=np.float32,
        ),
        _image_hdu(
            _error_array(raw_error),
            "ERROR",
            "Flux Error (10^-17 erg/s/cm^2/Ang)",
            axis2="Chip",
            dtype=np.float32,
        ),
        _image_hdu(
            mask, "MASK", "Flag Mask (bitwise)", axis2="Chip", dtype=np.int16
        ),
        _image_hdu(
            wave,
            "WAVELENGTH",
            "Wavelength (Ang)",
            axis2="Chip",
            dtype=np.float64,
        ),
        _image_hdu(
            sky,
            "SKY FLUX",
            "Sky (10^-17 erg/s/cm^2/Ang)",
            axis2="Chip",
            dtype=np.float32,
        ),
        _image_hdu(
            skyerr,
            "SKY ERROR",
            "Sky Error (10^-17 erg/s/cm^2/Ang)",
            axis2="Chip",
            dtype=np.float32,
        ),
        _image_hdu(
            telluric, "TELLURIC", "Telluric", axis2="Chip", dtype=np.float32
        ),
        _image_hdu(
            telluricerr,
            "TELLURIC ERROR",
            "Telluric Error",
            axis2="Chip",
            dtype=np.float32,
        ),
        _coefficient_hdu(wcoef, "WAVE COEFFICIENTS", "Wavelength Coefficients"),
        _coefficient_hdu(lsfcoef, "LSF COEFFICIENTS", "LSF Coefficients"),
    ]
    if fluxcorr is not None and np.asarray(fluxcorr).ndim > 1:
        hdus.append(
            _image_hdu(
                np.asarray(fluxcorr)[..., fiber_row],
                "FLUX CONVERSION",
                "ADU to flux units conv factor (ergs/s/cm^2/A)",
                axis2="Chip",
                dtype=np.float32,
            )
        )
    return fits.HDUList(hdus)


def write_visit_products(
    frame: Any,
    plugmap: Any,
    shiftstr: Any,
    pairstr: Any,
    *,
    plate_files: Sequence[str | Path] | None = None,
    visit_directory: str | Path,
    visit_filename: Callable[[int, str], str] | None = None,
    single: bool = False,
    mjdfrac: float | None = None,
    survey: str | None = None,
    relflux: Any | None = None,
    telescope: str = "apo25m",
    reduction_version: str = "",
    software_version: str = "",
    overwrite: bool = True,
) -> VisitProductResult:
    """Write plate products and all eligible STAR/HOT_STD visit spectra."""

    _, nfibers, _ = _validate_frame(frame)
    plate_written: list[str] = []
    if not single:
        if plate_files is None:
            raise ValueError("plate_files are required unless single=True")
        plate_written = write_plate_products(
            frame,
            plugmap,
            shiftstr,
            pairstr,
            plate_files,
            telescope=telescope,
            reduction_version=reduction_version,
            software_version=software_version,
            overwrite=overwrite,
        )
    fiberdata = _get(plugmap, "fiberdata")
    fiberids = _column(fiberdata, "fiberid").astype(int)
    spectrograph = _column(fiberdata, "spectrographid").astype(int)
    holetype = np.char.upper(_column(fiberdata, "holetype").astype(str))
    objtype = np.char.upper(_column(fiberdata, "objtype").astype(str))
    object_id = _column(fiberdata, "tmass_style").astype(str)
    directory = Path(visit_directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: list[int] = []
    plate = _optional(plugmap, "plateid", _optional(plugmap, "plate"))
    mjd = int(_get(plugmap, "mjd"))

    for fiber_row in range(nfibers):
        fiber_id = 300 - fiber_row
        match = np.flatnonzero((spectrograph == 2) & (fiberids == fiber_id))
        if match.size == 0:
            skipped.append(fiber_id)
            continue
        index = int(match[0])
        eligible = (
            holetype[index] == "OBJECT"
            and objtype[index] in {"STAR", "HOT_STD"}
            and object_id[index] not in {"-", "2MNone"}
        )
        if not eligible:
            continue
        if visit_filename is None:
            mjd_text = f"{mjdfrac:.2f}" if mjdfrac is not None else str(mjd)
            filename = f"apVisit-{plate}-{mjd_text}-{fiber_id:03d}.fits"
        else:
            filename = visit_filename(fiber_id, object_id[index].strip())
        output = directory / filename
        hdul = build_visit_hdul(
            frame,
            plugmap,
            fiber_row,
            survey=survey,
            relflux=relflux,
            telescope="apo1m" if single else telescope,
            reduction_version=reduction_version,
            software_version=software_version,
        )
        hdul.writeto(output, overwrite=overwrite)
        written.append(str(output))
    return VisitProductResult(plate_written, written, skipped)

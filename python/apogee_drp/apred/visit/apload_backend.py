"""Production filesystem adapter for the Python :mod:`ap1dvisit` port.

The spectroscopy lives in :class:`~.backend.NativeVisitBackendMixin`.  This
module only connects that implementation to the existing APOGEE ``ApLoad``,
plan, and platedata APIs.  Imports of the full DRP are intentionally lazy so
the numerical translation remains usable as a standalone package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

from .backend import NativeVisitBackendMixin
from .models import ChipFrame, VisitFrame

_CHIPS = ("a", "b", "c")


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        for key, value in row.items():
            if str(key).lower() == name.lower():
                return value
        return default
    try:
        return row[name]
    except (KeyError, IndexError, TypeError, ValueError):
        return getattr(row, name, default)


def _hdu_by_name(hdul: fits.HDUList, name: str, fallback: int | None = None) -> Any:
    wanted = name.replace(" ", "").upper()
    for hdu in hdul[1:]:
        extname = str(hdu.header.get("EXTNAME", "")).replace(" ", "").upper()
        if extname == wanted:
            return hdu.data
    if fallback is not None and fallback < len(hdul):
        return hdul[fallback].data
    return None


class ApLoadVisitBackend(NativeVisitBackendMixin):
    """Connect the translated visit reducer to a normal APOGEE reduction tree.

    Parameters other than ``apred`` and ``telescope`` are dependency-injection
    hooks used by tests and site deployments.  Normal users do not need them.
    Database ingestion is disabled unless ``ingest=True`` or an explicit
    ``visit_ingester`` is supplied.
    """

    def __init__(
        self,
        apred: str,
        telescope: str,
        *,
        verbose: bool = False,
        ingest: bool = False,
        load: Any | None = None,
        plan_loader: Callable[..., MutableMapping[str, Any]] | None = None,
        platedata_loader: Callable[..., Any] | None = None,
        lsf_builder: Callable[[Any], Any] | None = None,
        visit_ingester: Callable[[Any], Any] | None = None,
    ) -> None:
        if load is None:
            from ...utils import apload

            load = apload.ApLoad(
                apred=apred, telescope=telescope, verbose=verbose
            )
        self.load = load
        self.apred = apred
        self.telescope = telescope
        self.verbose = verbose
        self.ingest = bool(ingest)
        self._plan_loader = plan_loader
        self._platedata_loader = platedata_loader
        self._lsf_builder = lsf_builder
        self._visit_ingester = visit_ingester
        self._last_products = None
        self.pipeline_version = ""

    def load_plan(
        self, planfile: str, *, verbose: bool
    ) -> MutableMapping[str, Any]:
        loader = self._plan_loader
        if loader is None:
            from ...utils import plan as plan_utils

            loader = plan_utils.load
        plan = dict(loader(planfile, verbose=verbose))
        plan.setdefault("apred_vers", self.apred)
        plan.setdefault("telescope", self.telescope)
        plan.setdefault("platetype", "normal")
        plan["mjd"] = int(plan["mjd"])
        plan["plateid"] = int(plan["plateid"])
        plan.setdefault("field", plan.get("fieldid", ""))
        plan["reduction_version"] = str(plan.get("apred_vers", self.apred))
        plan["software_version"] = str(plan.get("gitvers", ""))
        plan["visit_directory"] = str(
            Path(
                self.filename(
                    "Plate",
                    plate=plan["plateid"],
                    mjd=plan["mjd"],
                    field=plan["field"],
                    chip="a",
                    directory=True,
                )
            )
        )
        plan["plate_files"] = self._files(
            "Plate", plan=plan, chips=_CHIPS
        )
        self._current_plan = plan
        return plan

    def directories(self, plan: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "telescope": self.telescope,
            "instrument": getattr(self.load, "instrument", ""),
            "observatory": getattr(self.load, "observatory", ""),
            "prefix": getattr(self.load, "prefix", "ap"),
            "visit": plan.get("visit_directory", ""),
        }

    def filename(self, kind: str, **kwargs: Any) -> str:
        chip = kwargs.pop("chip", None)
        directory = bool(kwargs.pop("directory", False))
        return self.load.filename(kind, chip=chip, directory=directory, **kwargs)

    def _product_filename(self, kind: str, **kwargs: Any) -> str:
        """Resolve products whose sdss_access template uses ``reduction``."""

        if "reduction" in kwargs and hasattr(self.load, "allfile"):
            return str(self.load.allfile(kind, download=False, **kwargs))
        return str(self.load.filename(kind, **kwargs))

    def _files(
        self,
        kind: str,
        *,
        plan: Mapping[str, Any] | None = None,
        chips: Sequence[str] = _CHIPS,
        **kwargs: Any,
    ) -> list[str]:
        if plan is not None:
            for key in ("field", "plateid", "mjd"):
                target = "plate" if key == "plateid" else key
                if key in plan and target not in kwargs and kind in {
                    "Plate", "Cframe", "Visit", "VisitSum", "Tellstar"
                }:
                    kwargs[target] = plan[key]
        filenames = self.load.filename(kind, chip=list(chips), **kwargs)
        return [str(filenames[chip]) for chip in chips]

    def load_plugmap(
        self, plan: Mapping[str, Any], *, mapper_data: Any = None
    ) -> Any:
        loader = self._platedata_loader
        if loader is None:
            from ...utils import platedata

            loader = platedata.getdata
        exposure0 = list(plan["APEXP"])[0]
        plate_type = str(plan.get("platetype", "normal")).lower()
        kwargs: dict[str, Any] = {
            "plugid": plan.get("plugmap"),
            "mapper_data": mapper_data,
            "fixfiberid": plan.get("fixfiberid", False),
            "badfiberid": plan.get("badfiberid"),
        }
        if plate_type == "single":
            kwargs.update(
                obj1m=_field(exposure0, "singlename"),
                starfiber=_field(exposure0, "single"),
            )
        elif plate_type == "twilight":
            kwargs["twilight"] = True
        return loader(
            int(plan["plateid"]),
            int(plan["mjd"]),
            str(plan.get("apred_vers", self.apred)),
            str(plan.get("telescope", self.telescope)),
            **kwargs,
        )

    def make_lsf(self, lsfid: Any) -> None:
        if self._lsf_builder is not None:
            self._lsf_builder(lsfid)
            return
        # LSF construction belongs to makecal.  Existing files are accepted;
        # a missing file is diagnosed by validate_files below.

    def calibration_files(
        self, plan: Mapping[str, Any], chips: Sequence[str]
    ) -> tuple[Sequence[str], Sequence[str]]:
        wave = self._files("Wave", num=plan["waveid"], chips=chips)
        lsf = self._files("LSF", num=plan["lsfid"], chips=chips)
        return wave, lsf

    def read_relflux(self, filename: str) -> np.ndarray:
        with fits.open(filename, memmap=False) as hdul:
            index = 2 if len(hdul) > 2 else 1
            return np.asarray(hdul[index].data, dtype=np.float32).squeeze()

    def frame_number(self, exposure: Any, plan: Mapping[str, Any]) -> str:
        number = int(_field(exposure, "name", exposure))
        return f"{number:08d}"

    def one_d_files(
        self, plan: Mapping[str, Any], framenum: str, chips: Sequence[str]
    ) -> Sequence[str]:
        return self._files("1D", num=int(framenum), chips=chips)

    def cframe_files(
        self, plan: Mapping[str, Any], framenum: str, chips: Sequence[str]
    ) -> Sequence[str]:
        return self._files(
            "Cframe", plan=plan, num=int(framenum), chips=chips
        )

    def validate_files(
        self, files: Sequence[str], *, mjd: int, kind: str
    ) -> None:
        missing = [str(path) for path in files if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing {kind} file(s): " + ", ".join(missing)
            )
        for path in files:
            with fits.open(path, memmap=False) as hdul:
                header = hdul[0].header
                file_mjd = header.get("MJD5", header.get("MJD"))
                if file_mjd is not None and int(float(file_mjd)) != int(mjd):
                    raise ValueError(
                        f"{kind} file {path} has MJD {file_mjd}, expected {mjd}"
                    )

    def load_frame(self, files: Sequence[str], *, kind: str) -> Any:
        lower = kind.lower()
        if lower == "cframe":
            plan = self._current_plan
            loaded = self.load.cframe(self._number_from_filename(files[0]),
                                      plan["plateid"], plan["mjd"], field=plan.get("field"))
            return VisitFrame.from_mapping(loaded)
        if lower == "plate":
            if len(files) == 1:
                first = str(files[0])
                files = [
                    first.replace("Plate-a-", f"Plate-{chip}-", 1)
                    for chip in _CHIPS
                ]
            return self._read_plate_files(files)
        if lower == "1d":
            number = self._number_from_filename(files[0])
            loaded = self.load.spectrum(number)
            frame = VisitFrame.from_mapping(loaded)
            for index, chip in enumerate(frame):
                chip.filename = str(files[index])
            return frame
        raise ValueError(f"unsupported frame kind {kind!r}")

    @staticmethod
    def _number_from_filename(filename: str) -> int:
        stem = Path(filename).name.rsplit(".", 1)[0]
        return int(stem.rsplit("-", 1)[-1])

    def _read_plate_files(self, files: Sequence[str]) -> VisitFrame:
        # Plate loading is rarely reached (only when reusing an existing
        # product); reading the explicit files avoids reconstructing arguments.
        loaded = {
            chip: fits.open(path, memmap=False)
            for chip, path in zip(_CHIPS, files)
        }
        return self._convert_plate_hdus(loaded, files)

    def _convert_plate_hdus(
        self, loaded: Any, files: Sequence[str]
    ) -> VisitFrame:
        if not isinstance(loaded, Mapping) or not all(
            chip in loaded for chip in _CHIPS
        ):
            raise OSError("plate input does not contain three chip HDU lists")
        try:
            chips = []
            for index, chip_name in enumerate(_CHIPS):
                hdul = loaded[chip_name]
                flux = _hdu_by_name(hdul, "FLUX", 1)
                error = _hdu_by_name(hdul, "ERROR", 2)
                mask = _hdu_by_name(hdul, "MASK", 3)
                if flux is None or error is None or mask is None:
                    raise ValueError(f"plate chip {chip_name} lacks flux/error/mask")
                values: dict[str, Any] = {
                    "filename": str(files[index]), "header": hdul[0].header.copy(),
                    "flux": np.asarray(flux, dtype=np.float32).copy(),
                    "err": np.asarray(error, dtype=np.float32).copy(),
                    "mask": np.asarray(mask, dtype=np.int16).copy(),
                }
                optional = {
                    "wavelength": ("WAVELENGTH", 4, np.float64),
                    "sky": ("SKYFLUX", 5, np.float32),
                    "skyerr": ("SKYERROR", 6, np.float32),
                    "telluric": ("TELLURIC", 7, np.float32),
                    "telluricerr": ("TELLURICERROR", 8, np.float32),
                    "wcoef": ("WAVECOEFFICIENTS", 9, np.float64),
                    "lsfcoef": ("LSFCOEFFICIENTS", 10, np.float64),
                }
                for name, (extname, fallback, dtype) in optional.items():
                    value = _hdu_by_name(hdul, extname, fallback)
                    if value is not None:
                        values[name] = np.asarray(value, dtype=dtype).copy()
                chips.append(ChipFrame(**values))
            frame = VisitFrame(*chips)
            frame.validate()
            return frame
        finally:
            for hdul in loaded.values():
                close = getattr(hdul, "close", None)
                if close is not None:
                    close()

    def prepare_frame(
        self,
        frame: Any,
        *,
        plan: Mapping[str, Any],
        wavefiles: Sequence[str],
        lsffiles: Sequence[str],
        plate_dir: str,
        newwave: bool,
    ) -> Any:
        if isinstance(frame, Mapping):
            frame = VisitFrame.from_mapping(frame)
        for index, chip_name in enumerate(_CHIPS):
            chip = frame.chip(chip_name)
            with fits.open(lsffiles[index], memmap=False) as hdul:
                chip.lsfcoef = np.asarray(hdul[0].data, dtype=np.float64).copy()
            chip.lsffile = str(lsffiles[index])
            chip.wavefile = str(wavefiles[index])
            chip.wave_dir = str(plate_dir)
            if chip.wcoef is None or chip.wavelength is None:
                raise ValueError(
                    "ap1D file lacks Python ap1dwavecal wavelength products; "
                    "run ap1dwavecal before ap1dvisit"
                )
            shape = chip.flux.shape
            if chip.sky is None: chip.sky = np.zeros(shape, dtype=np.float32)
            if chip.skyerr is None: chip.skyerr = np.zeros(shape, dtype=np.float32)
            if chip.telluric is None: chip.telluric = np.ones(shape, dtype=np.float32)
            if chip.telluricerr is None: chip.telluricerr = np.zeros(shape, dtype=np.float32)
        return frame

    def plate_file(self, plan: Mapping[str, Any]) -> str:
        return str(plan["plate_files"][0])

    def visit_summary_file(self, plan: Mapping[str, Any], plugmap: Any) -> str:
        return str(
            self.load.filename(
                "VisitSum",
                plate=plan["plateid"],
                mjd=plan["mjd"],
                field=plan.get("field", ""),
            )
        )

    def visit_product_options(
        self, plan: Mapping[str, Any]
    ) -> dict[str, Any]:
        options = super().visit_product_options(plan)

        def visit_filename(fiber: int, object_id: str) -> str:
            path = self._product_filename(
                "Visit",
                plate=plan["plateid"],
                mjd=plan["mjd"],
                fiber=fiber,
                field=plan.get("field", ""),
                reduction=object_id,
            )
            return Path(str(path)).name

        options["visit_filename"] = visit_filename
        return options

    def write_visit_products(self, *args: Any, **kwargs: Any) -> Any:
        result = super().write_visit_products(*args, **kwargs)
        self._last_products = result
        return result

    def build_visit_rows(
        self,
        *,
        plan: Mapping[str, Any],
        plugmap: Any,
        object_indices: np.ndarray,
        survey: str,
        nframes: int,
        relflux: np.ndarray,
    ) -> Table:
        fiberdata = self.fiber_data(plugmap)
        rows: list[dict[str, Any]] = []
        files_by_fiber: dict[int, str] = {}
        if self._last_products is not None:
            for filename in self._last_products.visit_files:
                try:
                    fiber = int(Path(filename).stem.rsplit("-", 1)[-1])
                    files_by_fiber[fiber] = filename
                except ValueError:
                    pass
        for index in np.asarray(object_indices, dtype=int):
            row = fiberdata[index]
            fiber = int(_field(row, "fiberid"))
            filename = files_by_fiber.get(fiber)
            if filename is None:
                candidate = self._product_filename(
                    "Visit",
                    plate=plan["plateid"],
                    mjd=plan["mjd"],
                    fiber=fiber,
                    field=plan.get("field", ""),
                    reduction=str(_field(row, "tmass_style", "")),
                )
                filename = str(candidate)
            header = fits.getheader(filename, 0)
            names = getattr(getattr(row, "dtype", None), "names", None) or ()
            values = {name.lower(): _field(row, name) for name in names}
            values.update(
                apogee_id=str(_field(row, "tmass_style", "")).strip(),
                target_id=str(_field(row, "object", "")).strip(),
                file=Path(filename).name,
                uri=os.path.relpath(filename, os.environ.get("MWM_ROOT", "/")),
                apred_vers=self.apred,
                fiberid=fiber,
                plate=str(plan["plateid"]),
                mjd=int(plan["mjd"]),
                telescope=self.telescope,
                survey=survey,
                field=str(plan.get("field", "")),
                design=str(plan.get("designid", -999)),
                nframes=int(nframes),
                exptime=float(header.get("EXPTIME", 0.0)),
                dateobs=str(header.get("DATE-OBS", "")),
                jd=float(header.get("JD-MID", header.get("JD", 0.0))),
                snr=float(header.get("SNR", 0.0)),
                relflux=float(header.get("RELFLUX", 1.0)),
                mtpflux=float(header.get("MTPFLUX", 1.0)),
                visitflag=int(header.get("STARFLAG", header.get("VISITFLAG", 0))),
            )
            rows.append(values)
        return Table(rows=rows)

    def write_visit_summary(
        self,
        filename: str,
        rows: Any,
        *,
        plan: Mapping[str, Any],
        source_frame: Any,
    ) -> None:
        header = fits.Header()
        header["PLATEID"] = int(plan["plateid"])
        header["MJD"] = int(plan["mjd"])
        header["APRED"] = self.apred
        if rows is not None and len(rows):
            header["NSPECTRA"] = len(rows)
        hdul = fits.HDUList(
            [fits.PrimaryHDU(header=header), fits.table_to_hdu(Table(rows))]
        )
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        hdul.writeto(filename, overwrite=True)

    def ingest_visits(self, rows: Any) -> None:
        if self._visit_ingester is not None:
            self._visit_ingester(rows)
        elif self.ingest:
            raise RuntimeError(
                "database ingestion requested but no visit_ingester was supplied"
            )

    def write_tellstar_summary(
        self, plan: Mapping[str, Any], tellstars: Sequence[Any]
    ) -> None:
        if not tellstars:
            return
        filename = str(
            self.load.filename(
                "Tellstar",
                plate=plan["plateid"],
                mjd=plan["mjd"],
                field=plan.get("field", ""),
            )
        )
        tables = [Table(item) for item in tellstars]
        combined = tables[0] if len(tables) == 1 else vstack(tables)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        combined.write(filename, overwrite=True)

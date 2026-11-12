"""Focused tests for the production ApLoad visit adapter."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from apogee_drp.apred.visit.apload_backend import (
    ApLoadVisitBackend,
    _chip_filenames,
)


class FakeLoad:
    instrument = "apogee-n"
    observatory = "apo"
    prefix = "ap"

    def __init__(self, root):
        self.root = Path(root)
        self.ap1d_calls = []

    def filename(self, kind, chips=False, **kwargs):
        number = kwargs.get("num", kwargs.get("plate", 1))
        if kind in {"Plate", "Cframe", "Visit", "VisitSum", "Tellstar"}:
            directory = self.root / "visit" / str(kwargs.get("mjd", 60000))
        else:
            directory = self.root / "cal"
        return str(directory / f"ap{kind}-{number}.fits")

    def allfile(self, kind, download=False, **kwargs):
        return self.filename(kind, **kwargs)

    def ap1D(self, number):
        self.ap1d_calls.append(number)
        output = {}
        for letter in "abc":
            output[letter] = fits.HDUList(
                [
                    fits.PrimaryHDU(header=fits.Header({"DITHPIX": 0.0})),
                    fits.ImageHDU(np.ones((2, 5)), name="FLUX"),
                    fits.ImageHDU(np.ones((2, 5)), name="ERROR"),
                    fits.ImageHDU(np.zeros((2, 5), dtype=np.int16), name="MASK"),
                    fits.ImageHDU(np.ones((2, 5)), name="WAVELENGTH"),
                    fits.ImageHDU(np.zeros((2, 5)), name="SKY FLUX"),
                    fits.ImageHDU(np.zeros((2, 5)), name="SKY ERROR"),
                    fits.ImageHDU(np.ones((2, 5)), name="TELLURIC"),
                    fits.ImageHDU(np.zeros((2, 5)), name="TELLURIC ERROR"),
                    fits.ImageHDU(np.zeros((2, 14)), name="WAVE COEFFICIENTS"),
                ]
            )
        return output


def test_chip_placeholder_expansion():
    result = _chip_filenames("/x/ap1D-123.fits", "1D", ("a", "b", "c"))
    assert result == [
        "/x/ap1D-a-123.fits",
        "/x/ap1D-b-123.fits",
        "/x/ap1D-c-123.fits",
    ]


def test_load_plan_uses_apload_for_output_paths(tmp_path):
    backend = ApLoadVisitBackend(
        "daily",
        "apo25m",
        load=FakeLoad(tmp_path),
        plan_loader=lambda filename, verbose=False: {
            "mjd": 60000,
            "plateid": 1234,
            "field": "field",
            "waveid": 1,
            "lsfid": 2,
            "APEXP": [{"name": 12345678}],
        },
    )
    plan = backend.load_plan("plan.yaml", verbose=False)
    assert plan["apred_vers"] == "daily"
    assert len(plan["plate_files"]) == 3
    assert plan["plate_files"][0].endswith("apPlate-a-1234.fits")
    assert plan["visit_directory"].endswith("visit/60000")


def test_ap1d_loader_conversion_and_existing_wave_products(tmp_path):
    load = FakeLoad(tmp_path)
    backend = ApLoadVisitBackend("daily", "apo25m", load=load)
    files = [
        str(tmp_path / f"ap1D-{chip}-12345678.fits") for chip in "abc"
    ]
    frame = backend.load_frame(files, kind="1D")
    assert load.ap1d_calls == [12345678]
    assert frame["chipb"]["flux"].shape == (2, 5)
    assert frame["chipb"]["wcoef"].shape == (2, 14)
    assert frame["chipb"]["wavelength"].shape == (2, 5)


def test_platedata_arguments_are_forwarded(tmp_path):
    calls = []

    def getdata(*args, **kwargs):
        calls.append((args, kwargs))
        return {"fiberdata": []}

    backend = ApLoadVisitBackend(
        "daily",
        "apo25m",
        load=FakeLoad(tmp_path),
        platedata_loader=getdata,
    )
    plan = {
        "plateid": 123,
        "mjd": 60000,
        "apred_vers": "daily",
        "telescope": "apo25m",
        "platetype": "normal",
        "plugmap": "confSummary-123.par",
        "APEXP": [{"name": 12345678}],
    }
    backend.load_plugmap(plan, mapper_data="/mapper")
    args, kwargs = calls[0]
    assert args == (123, 60000, "daily", "apo25m")
    assert kwargs["plugid"] == "confSummary-123.par"
    assert kwargs["mapper_data"] == "/mapper"

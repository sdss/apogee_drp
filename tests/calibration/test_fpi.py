"""Tests for the numbered FPI calibration wrapper."""

from pathlib import Path

import numpy as np
import pytest

from apogee_drp.apred.cal import fpi as fv
from apogee_drp.apred.cal import makecal
from apogee_drp.apred.process import process


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"

    def __init__(self, root):
        self.root = Path(root)

    def cmjd(self, number):
        return "60000"

    def filename(self, kind, num=None, mjd=None, chips=False, chip=None,
                 dir=False):
        directory = self.root / ("exp" if kind in ("R", "2D", "1D") else "cal")
        if dir:
            return str(directory)
        return str(directory / f"ap{kind}-{int(num):08d}.fits")


def metadata(rows):
    return np.array(rows, dtype=[("num", np.int64), ("exptype", "U20")])


def test_product_files_are_three_chips(tmp_path):
    files = fv.product_files(FakeLoad(tmp_path), 123)
    assert len(files) == 3
    assert files[0].endswith("apWaveFPI-a-00000123.fits")
    assert files[2].endswith("apWaveFPI-c-00000123.fits")


def test_fpi_exposures_filters_sorts_and_deduplicates():
    rows = metadata([(30, " FPI "), (20, "DARK"), (10, "fpi"), (30, "FPI")])
    assert fv.fpi_exposures(rows) == [10, 30]


def test_fpi_exposures_empty():
    assert fv.fpi_exposures(None) == []
    assert fv.fpi_exposures(metadata([])) == []


def test_fpi_exposures_requires_columns():
    rows = np.zeros(2, dtype=[("num", int)])
    with pytest.raises(ValueError, match="num and exptype"):
        fv.fpi_exposures(rows)


def test_select_library_psf(monkeypatch):
    import apogee_drp.apred.cal.getpsfcal as module
    calls = []
    monkeypatch.setattr(module, "getpsfcal",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or 55)
    assert fv._select_library_psf(123, mjd=60000, telescope="apo25m") == 55
    assert calls[0][1]["psflibrary"] is True


def test_select_library_psf_requires_result(monkeypatch):
    import apogee_drp.apred.cal.getpsfcal as module
    monkeypatch.setattr(module, "getpsfcal", lambda *args, **kwargs: -1)
    with pytest.raises(RuntimeError, match="No library PSF"):
        fv._select_library_psf(123, mjd=60000, telescope="apo25m")


def patch_builder(monkeypatch, tmp_path, *, create_outputs=True):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(fv, "_make_load", lambda **kwargs: load)
    lock_calls = []
    monkeypatch.setattr(fv.lock, "lock",
                        lambda *args, **kwargs: lock_calls.append((args, kwargs)))
    process_calls = []
    monkeypatch.setattr(
        fv, "_process_exposures",
        lambda *args, **kwargs: process_calls.append((args, kwargs)))
    solution_calls = []

    def solution(mjd, **kwargs):
        solution_calls.append((mjd, kwargs))
        if create_outputs:
            for filename in fv.product_files(load, kwargs["number"]):
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                Path(filename).write_bytes(b"wave")

    monkeypatch.setattr(fv, "_run_fpi_solution", solution)
    return load, lock_calls, process_calls, solution_calls


def test_build_fpi_reduces_night_and_runs_solution(monkeypatch, tmp_path):
    load, lock_calls, process_calls, solution_calls = patch_builder(
        monkeypatch, tmp_path)
    outputs = fv.build_fpi(
        123, psfid=50, darkid=10, flatid=20,
        night_exposures=[125, 123, 124, 123], verbose=True)
    assert outputs == fv.product_files(load, 123)
    assert all(Path(filename).read_bytes() == b"wave" for filename in outputs)
    assert process_calls[0][0] == ([123, 124, 125],)
    assert process_calls[0][1]["psfid"] == 50
    assert solution_calls[0][0] == 60000
    assert solution_calls[0][1]["observatory"] == "apo"
    assert lock_calls[-1][1] == {"clear": True}
    assert Path(outputs[0].replace("WaveFPI-a-", "WaveFPI-")).with_suffix(".dat").is_file()


def test_build_fpi_uses_model_psf(monkeypatch, tmp_path):
    _, _, process_calls, _ = patch_builder(monkeypatch, tmp_path)
    fv.build_fpi(123, modelpsf="40-50", night_exposures=[123])
    assert process_calls[0][1]["psfid"] is None
    assert process_calls[0][1]["modelpsf"] == "40-50"


def test_build_fpi_resolves_library_psf(monkeypatch, tmp_path):
    _, _, process_calls, _ = patch_builder(monkeypatch, tmp_path)
    monkeypatch.setattr(fv, "_select_library_psf", lambda *args, **kwargs: 77)
    fv.build_fpi(123, librarypsf=True, night_exposures=[123])
    assert process_calls[0][1]["psfid"] == 77


@pytest.mark.parametrize("kwargs", [
    {"librarypsf": True, "psfid": 5},
    {"librarypsf": True, "modelpsf": "4-5"},
])
def test_build_fpi_rejects_conflicting_psf_selection(monkeypatch, tmp_path, kwargs):
    patch_builder(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="cannot be combined"):
        fv.build_fpi(123, night_exposures=[123], **kwargs)


def test_build_fpi_requires_psf_selection(monkeypatch, tmp_path):
    patch_builder(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="psfid, modelpsf, or librarypsf"):
        fv.build_fpi(123, night_exposures=[123])


def test_build_fpi_requires_exposures(monkeypatch, tmp_path):
    patch_builder(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="No FPI exposures"):
        fv.build_fpi(123, psfid=5, night_exposures=[])


def test_build_fpi_requested_frame_must_be_fpi(monkeypatch, tmp_path):
    patch_builder(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="not an FPI exposure"):
        fv.build_fpi(123, psfid=5, night_exposures=[124])


def test_build_fpi_name_must_match_exposure(monkeypatch, tmp_path):
    patch_builder(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="must match"):
        fv.build_fpi(123, name=456, psfid=5, night_exposures=[123])


def test_build_fpi_requires_requested_id(monkeypatch, tmp_path):
    patch_builder(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        fv.build_fpi([], psfid=5, night_exposures=[])


def test_build_fpi_existing_short_circuit(monkeypatch, tmp_path, capsys):
    load, _, process_calls, _ = patch_builder(monkeypatch, tmp_path)
    outputs = fv.product_files(load, 123)
    for filename in outputs:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert fv.build_fpi(123, verbose=True) == outputs
    assert not process_calls
    assert "already made" in capsys.readouterr().out


def test_build_fpi_partial_product_is_rebuilt(monkeypatch, tmp_path):
    load, _, process_calls, _ = patch_builder(monkeypatch, tmp_path)
    outputs = fv.product_files(load, 123)
    Path(outputs[0]).parent.mkdir(parents=True)
    Path(outputs[0]).write_bytes(b"partial")
    fv.build_fpi(123, psfid=5, night_exposures=[123])
    assert process_calls
    assert all(Path(filename).is_file() for filename in outputs)


def test_build_fpi_verifies_outputs(monkeypatch, tmp_path):
    _, lock_calls, _, _ = patch_builder(
        monkeypatch, tmp_path, create_outputs=False)
    with pytest.raises(RuntimeError, match="did not create"):
        fv.build_fpi(123, psfid=5, night_exposures=[123])
    assert lock_calls[-1][1] == {"clear": True}


def test_process_model_psf_uses_dense_id_and_type_five(tmp_path):
    load = FakeLoad(tmp_path)
    for chip in "abc":
        filename = Path(load.filename("2D", num=123, chips=True))
        filename = Path(str(filename).replace("2D-", f"2D-{chip}-"))
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.touch()
    calls = []
    process(
        123, load=load, modelpsf="40-50", doap2dproc=True,
        process_2d=lambda *args, **kwargs: calls.append((args, kwargs)))
    assert calls[0][0][1].endswith("cal/00000050")
    assert calls[0][1]["extract_type"] == 5
    assert calls[0][1]["modelpsffile"].endswith("cal/40-50")


def test_makecal_fpi_dispatches_numbered_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(makecal_v7, "build_fpi",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal_v7.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        modelpsf="40-50", verbose=True)
    monkeypatch.setattr(context, "calibrations", lambda mjd: {
        "darkid": 1, "flatid": 2, "fiberid": 3})
    makecal_v7.fpi("123", context)
    assert calls[0][0] == ("123",)
    assert calls[0][1]["modelpsf"] == "40-50"
    assert calls[0][1]["apred"] == "daily"

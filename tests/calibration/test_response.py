"""Focused tests for standalone Response calibration building."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import fluxcal
from apogee_drp.apred.cal import makecal


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"

    def __init__(self, root):
        self.root = Path(root)

    def cmjd(self, number):
        return "60000"

    def filename(self, kind, num=None, chips=False, **kwargs):
        return str(self.root / kind.lower() / f"ap{kind}-{int(num):08d}.fits")


def patch_response(monkeypatch, tmp_path, *, reference=None):
    load = FakeLoad(tmp_path)
    locks = []
    monkeypatch.setattr(
        fluxcal.lock, "lock",
        lambda *args, **kwargs: locks.append((args, kwargs)))
    if reference is None:
        reference = np.linspace(0.8, 1.2, 20)
    for filename in fluxcal.product_files(load, 12):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([
            fits.PrimaryHDU(), fits.ImageHDU(), fits.ImageHDU(),
            fits.ImageHDU(reference),
        ]).writeto(filename)
    wave = np.linspace(15000, 17000, len(reference))
    for chip in "abc":
        filename = Path(fluxcal._chip_filename(load, "Wave", 99, chip))
        filename.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([
            fits.PrimaryHDU(), fits.ImageHDU(),
            fits.ImageHDU(np.broadcast_to(wave, (4, len(wave)))),
        ]).writeto(filename)
    return load, locks


def test_build_response_reads_flux_without_rebuilding(monkeypatch, tmp_path):
    load, locks = patch_response(monkeypatch, tmp_path)
    outputs = fluxcal.build_response(
        12, waveid=99, temp=4000, load=load)
    assert outputs == fluxcal.product_files(load, 12, "Response")
    for filename in outputs:
        with fits.open(filename) as hdus:
            assert hdus[0].data.shape == (20,)
            assert hdus[0].header["BBTEMP"] == 4000
            assert hdus[0].header["WAVEID"] == "99"
            assert np.all(np.isfinite(hdus[0].data))
    assert locks[-1][1] == {"clear": True}


def test_response_is_normalized_at_green_chip_center(monkeypatch, tmp_path):
    load, _ = patch_response(monkeypatch, tmp_path, reference=np.ones(21))
    outputs = fluxcal.build_response(12, waveid=99, temp=4000, load=load)
    assert fits.getdata(outputs[1])[10] == pytest.approx(1)


def test_existing_response_short_circuits(monkeypatch, tmp_path, capsys):
    load, locks = patch_response(monkeypatch, tmp_path)
    outputs = fluxcal.product_files(load, 12, "Response")
    for filename in outputs:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert fluxcal.build_response(
        12, waveid=99, temp=4000, load=load, verbose=True) == outputs
    assert "already made" in capsys.readouterr().out
    assert all(kwargs != {"lock": True} for _, kwargs in locks)


def test_partial_response_is_rebuilt(monkeypatch, tmp_path):
    load, _ = patch_response(monkeypatch, tmp_path)
    outputs = fluxcal.product_files(load, 12, "Response")
    Path(outputs[0]).parent.mkdir(parents=True)
    Path(outputs[0]).write_bytes(b"partial")
    fluxcal.build_response(12, waveid=99, temp=4000, load=load)
    assert all(Path(filename).stat().st_size > 0 for filename in outputs)
    assert Path(outputs[0]).read_bytes() != b"partial"


def test_missing_flux_is_an_explicit_dependency_error(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(fluxcal.lock, "lock", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError, match="Missing Response dependency"):
        fluxcal.build_response(12, waveid=99, temp=4000, load=load)


def test_nonfinite_response_is_rejected_and_lock_cleared(monkeypatch, tmp_path):
    reference = np.ones(20)
    reference[4] = 0
    load, locks = patch_response(monkeypatch, tmp_path, reference=reference)
    with pytest.raises(ValueError, match="nonfinite"):
        fluxcal.build_response(12, waveid=99, temp=4000, load=load)
    assert locks[-1][1] == {"clear": True}


def test_makecal_response_calls_only_response_builder(monkeypatch, tmp_path):
    response_calls, flux_calls = [], []
    monkeypatch.setattr(
        makecal_v12, "build_response",
        lambda *args, **kwargs: response_calls.append((args, kwargs)))
    monkeypatch.setattr(
        makecal_v12, "build_flux",
        lambda *args, **kwargs: flux_calls.append((args, kwargs)))
    context = makecal_v12.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        clobber=True, unlock=True, verbose=True)
    monkeypatch.setattr(
        context, "row",
        lambda *args, **kwargs: {"temp": 4000, "psf": 34})
    monkeypatch.setattr(
        context, "calibrations", lambda mjd: {"waveid": 99})
    makecal_v12.response("12", context)
    assert not flux_calls
    assert response_calls == [((12,), {
        "waveid": 99, "temp": 4000, "apred": "daily",
        "telescope": "apo25m", "clobber": True, "unlock": True,
        "verbose": True})]

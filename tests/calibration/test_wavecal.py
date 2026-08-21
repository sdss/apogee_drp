"""Tests for the shared numbered wavelength-calibration builders."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import makecal
from apogee_drp.apred.cal import utils as cal_utils
from apogee_drp.apred.cal import wavecal as wv


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"
    instrument = "apogee-n"

    def __init__(self, root):
        self.root = Path(root)
        self.delete_calls = []

    def filename(self, kind, num=None, chip=None, directory=False, **kwargs):
        root = self.root / kind
        if directory:
            return str(root)
        infix = f"-{chip}" if chip is not None else ""
        return str(root / f"ap{kind}{infix}-{int(num):08d}.fits")

    def cmjd(self, number): return 60000

    def product_files(self, product, name):
        assert product in ("wave", "multiwave", "dailywave")
        return [self.filename("Wave", num=name, chip=chip) for chip in "abc"]

    def product_exists(self, product, name):
        return all(
            Path(filename).is_file() and Path(filename).stat().st_size > 0
            for filename in self.product_files(product, name)
        )

    def product_delete(self, product, name, **kwargs):
        self.delete_calls.append((product, name, kwargs))
        for filename in self.product_files(product, name):
            path = Path(filename)
            if path.exists() or path.is_symlink():
                path.unlink()


def test_product_files(tmp_path):
    files = FakeLoad(tmp_path).product_files("wave", 123)
    assert len(files) == 3
    assert files[0].endswith("apWave-a-00000123.fits")


def test_obsolete_product_filename_and_load_helpers_are_removed():
    assert not hasattr(wv, "product_files")
    assert not hasattr(wv, "_chip_filename")
    assert not hasattr(wv, "_make_load")
    assert not hasattr(wv, "_complete")


@pytest.mark.parametrize("lamp,center,threshold", [
    ("LAMPUNE", 1452, 40), ("LAMPTHAR", 1566, 1000), (None, 1000, 10)])
def test_arc_flux_metric_uses_spectral_columns(lamp, center, threshold):
    image = np.zeros((16, 1800))
    image[:, center] = threshold * 2
    header = fits.Header({"NREAD": 2})
    if lamp: header[lamp] = 1
    metric, required = wv.arc_flux_metric(image, header, smooth_width=1)
    assert metric == pytest.approx(threshold * 8)
    assert required == threshold


def test_arc_flux_metric_validates_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        wv.arc_flux_metric(np.ones(4), {"NREAD": 1})
    with pytest.raises(ValueError, match="NREAD"):
        wv.arc_flux_metric(np.ones((8, 1800)), {})


def patch_common(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(wv.apload, "ApLoad", lambda **kwargs: load)
    locks = []
    monkeypatch.setattr(cal_utils.lock, "lock",
                        lambda *args, **kwargs: locks.append((args, kwargs)))
    process_calls = []
    monkeypatch.setattr(wv, "_process_frames",
                        lambda *args, **kwargs: process_calls.append((args, kwargs)))
    monkeypatch.setattr(wv, "_check_arc", lambda load, frame: (True, "okay"))
    return load, locks, process_calls


def write_products(load, number, product="wave"):
    for filename in load.product_files(product, number):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"wave")


def test_build_wave_workflow(monkeypatch, tmp_path):
    load, locks, process_calls = patch_common(monkeypatch, tmp_path)
    calls = []
    def run(frames, **kwargs):
        calls.append((frames, kwargs)); write_products(load, kwargs["name"])
    monkeypatch.setattr(wv, "_run_wavecal", run)
    assert wv.build_wave(
        [123, 124], name=123, psfid=50, verbose=True) is None
    assert process_calls[0][0] == ([123, 124],)
    assert calls[0][0] == [123, 124]
    assert calls[0][1]["dependencies"] is True
    assert locks[-1][1] == {"clear": True}


def test_build_wave_drops_low_flux_member(monkeypatch, tmp_path):
    load, _, _ = patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(wv, "_check_arc",
                        lambda load, frame: (frame == 124, str(frame)))
    calls = []
    monkeypatch.setattr(wv, "_run_wavecal",
                        lambda frames, **kwargs: (calls.append(frames), write_products(load, 123)))
    wv.build_wave([123, 124], psfid=50)
    assert calls == [[124]]


def test_build_wave_nofit_verifies_lines(monkeypatch, tmp_path):
    load, _, _ = patch_common(monkeypatch, tmp_path)
    def run(frames, **kwargs):
        for frame in frames:
            filename = Path(wv._lines_file(load, frame))
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.write_bytes(b"lines")
    monkeypatch.setattr(wv, "_run_wavecal", run)
    assert wv.build_wave([123, 124], psfid=50, nofit=True) is None
    assert all(Path(wv._lines_file(load, frame)).is_file()
               for frame in (123, 124))


def test_build_wave_nofit_does_not_delete_wave_product(monkeypatch, tmp_path):
    load, _, _ = patch_common(monkeypatch, tmp_path)
    write_products(load, 123)

    def run(frames, **kwargs):
        path = Path(wv._lines_file(load, 123))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"lines")

    monkeypatch.setattr(wv, "_run_wavecal", run)
    wv.build_wave(123, psfid=50, nofit=True)
    assert not load.delete_calls
    assert load.product_exists("wave", 123)


def test_build_wave_existing(monkeypatch, tmp_path, capsys):
    load, locks, process_calls = patch_common(monkeypatch, tmp_path)
    write_products(load, 123)
    assert wv.build_wave(123, verbose=True) is None
    assert not process_calls
    assert not locks
    assert "wave product 123 already exists" in capsys.readouterr().out


def test_build_wave_requires_psf(monkeypatch, tmp_path):
    patch_common(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="psfid or modelpsf"):
        wv.build_wave(123)


def test_build_wave_requires_good_arc(monkeypatch, tmp_path):
    patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(wv, "_check_arc", lambda *args: (False, "bad"))
    with pytest.raises(ValueError, match="No input arc"):
        wv.build_wave(123, psfid=50)


def test_build_wave_clears_lock_on_failure(monkeypatch, tmp_path):
    _, locks, _ = patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(wv, "_run_wavecal", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="all chip"):
        wv.build_wave(123, psfid=50)
    assert locks[-1][1] == {"clear": True}


def test_build_multiwave_uses_available_lines_without_requiring_all(monkeypatch, tmp_path):
    load, _, _ = patch_common(monkeypatch, tmp_path)
    line = Path(wv._lines_file(load, 123)); line.parent.mkdir(parents=True); line.write_bytes(b"x")
    calls = []
    def run(frames, **kwargs):
        calls.append((frames, kwargs))
        write_products(load, 60000, "multiwave")
    monkeypatch.setattr(wv, "_run_wavecal", run)
    wv.build_multiwave([123, 124, 125], name=60000)
    assert calls[0][0] == [123, 124, 125]
    assert calls[0][1]["dependencies"] is False


def test_build_multiwave_requires_some_lines(monkeypatch, tmp_path):
    patch_common(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="No individual"):
        wv.build_multiwave([123, 124], name=60000)


def test_build_multiwave_existing_short_circuits(monkeypatch, tmp_path,
                                                  capsys):
    load, locks, _ = patch_common(monkeypatch, tmp_path)
    write_products(load, 60000, "multiwave")
    assert wv.build_multiwave(
        [123, 124], name=60000, verbose=True) is None
    assert not locks
    assert "multiwave product 60000 already exists" in capsys.readouterr().out


def test_build_multiwave_explicit_dependencies(monkeypatch, tmp_path):
    load, _, _ = patch_common(monkeypatch, tmp_path)
    singles = []
    def single(frames, **kwargs):
        singles.append(frames)
        for frame in frames:
            path = Path(wv._lines_file(load, frame))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    monkeypatch.setattr(wv, "build_wave", single)
    monkeypatch.setattr(wv, "_run_wavecal",
                        lambda frames, **kwargs: write_products(
                            load, 60000, "multiwave"))
    wv.build_multiwave([1, 2, 3, 4], name=60000, dependencies=True,
                       single_builder_options={"psfid": 50})
    assert singles == [[1, 2], [3, 4]]


def test_build_dailywave_forwards_dependency_policy(monkeypatch, tmp_path):
    load, _, _ = patch_common(monkeypatch, tmp_path)
    calls = []
    def run(mjd, **kwargs):
        calls.append((mjd, kwargs))
        write_products(load, mjd, "dailywave")
    monkeypatch.setattr(wv, "_run_dailywave", run)
    wv.build_dailywave(60000, dependencies=False)
    assert calls[0][1]["dependencies"] is False
    assert calls[0][1]["observatory"] == "apo"


def test_build_dailywave_dependencies_can_be_enabled(monkeypatch, tmp_path):
    load, _, _ = patch_common(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(wv, "_run_dailywave",
                        lambda mjd, **kwargs: (
                            calls.append(kwargs),
                            write_products(load, mjd, "dailywave")))
    wv.build_dailywave(60000, dependencies=True)
    assert calls[0]["dependencies"] is True


def test_build_dailywave_existing_short_circuits(monkeypatch, tmp_path,
                                                  capsys):
    load, locks, _ = patch_common(monkeypatch, tmp_path)
    write_products(load, 60000, "dailywave")
    assert wv.build_dailywave(60000, verbose=True) is None
    assert not locks
    assert "dailywave product 60000 already exists" in capsys.readouterr().out


def test_build_dailywave_clears_lock_on_failure(monkeypatch, tmp_path):
    _, locks, _ = patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(wv, "_run_dailywave", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="all chip"):
        wv.build_dailywave(60000)
    assert locks[-1][1] == {"clear": True}


def test_build_dailywave_rejects_nofit(monkeypatch, tmp_path):
    patch_common(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="individual"):
        wv.build_dailywave(60000, nofit=True)


def test_makecal_uses_current_wave_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(makecal, "build_wave",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={})
    monkeypatch.setattr(context, "row", lambda *args, **kwargs: None)
    monkeypatch.setattr(context, "calibrations", lambda mjd: {"modelpsf": "40-50"})
    makecal.wave("123", context)
    assert calls[0][0] == ([123],)
    assert calls[0][1]["modelpsf"] == "40-50"


def test_makecal_multiwave_never_builds_dependencies(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        makecal, "build_multiwave",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        dependencies=True)
    monkeypatch.setattr(
        context, "frames", lambda *args, **kwargs: [123, 124])
    makecal.multiwave("60000", context)
    assert calls[0][0] == ([123, 124],)
    assert calls[0][1]["dependencies"] is False


@pytest.mark.parametrize("dependencies", [False, True])
def test_makecal_dailywave_forwards_dependency_policy(
        monkeypatch, tmp_path, dependencies):
    calls = []
    monkeypatch.setattr(
        makecal, "build_dailywave",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        dependencies=dependencies, modelpsf="40-50")
    monkeypatch.setattr(context, "calibrations", lambda mjd: {
        "darkid": 1, "flatid": 2, "fiberid": 3})
    makecal.dailywave("60000", context)
    assert calls[0][0] == (60000,)
    assert calls[0][1]["dependencies"] is dependencies
    assert calls[0][1]["modelpsf"] == "40-50"

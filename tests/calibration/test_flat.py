"""Tests for the numbered flat-calibration implementation."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import flat as flat_module
from apogee_drp.apred.cal import utils as cal_utils


def test_make_flat_chip_does_not_modify_input():
    flat = np.ones((64, 64), dtype=float)
    flatmask = np.zeros_like(flat, dtype=np.uint32)
    original = flat.copy()
    flat_module.make_flat_chip(flat, flatmask, bad_pixel_bits=0)
    np.testing.assert_array_equal(flat, original)


def test_make_flat_chip_marks_reduction_and_zero_pixels():
    flat = np.ones((64, 64), dtype=float)
    flatmask = np.zeros_like(flat, dtype=np.uint32)
    flatmask[10, 10] = 4
    flat[20, 20] = 0

    output, spectral, mask = flat_module.make_flat_chip(
        flat,
        flatmask,
        bad_pixel_bits=4,
    )

    assert output[10, 10] == 0
    assert mask[10, 10] & 1
    assert output[20, 20] == 0
    assert mask[20, 20] & 8
    assert output.shape == spectral.shape == mask.shape


def test_make_flat_chip_flags_outlier_and_neighbor():
    flat = np.ones((64, 64), dtype=float)
    flat[30, 30] = 2.0
    flat[30, 31] = 1.10
    flatmask = np.zeros_like(flat, dtype=np.uint32)

    _, _, mask = flat_module.make_flat_chip(
        flat,
        flatmask,
        bad_pixel_bits=0,
    )

    assert mask[30, 30] & 2
    assert mask[30, 31] & 4


def test_make_flat_chip_validates_shapes():
    with pytest.raises(ValueError, match="identical"):
        flat_module.make_flat_chip(
            np.ones((10, 10)),
            np.zeros((9, 10)),
            bad_pixel_bits=0,
        )


def test_non_dithered_spectral_flat_is_quadratic_model():
    x = np.arange(64, dtype=float)
    profile = 1.0 + 1e-3 * x + 2e-5 * x**2
    flat = np.broadcast_to(profile[None, :], (64, 64)).copy()

    output, spectral, mask = flat_module.make_flat_chip(
        flat,
        np.zeros_like(flat, dtype=np.uint32),
        bad_pixel_bits=0,
    )

    np.testing.assert_allclose(output, flat)
    np.testing.assert_allclose(spectral[0, :], profile, rtol=1e-10)
    assert not np.any(mask)


def test_nrep_validation_happens_before_pipeline_io():
    with pytest.raises(ValueError, match="nrep"):
        flat_module.build_flat([12345678], nrep=0)


def test_build_flat_validates_empty_input_before_io():
    with pytest.raises(ValueError, match="at least one"):
        flat_module.build_flat([])


def test_normalize_flat_chips_uses_python_y_x_regions(monkeypatch):
    monkeypatch.setattr(flat_module, "DETECTOR_SHAPE", (6, 8))
    monkeypatch.setattr(flat_module, "NORM_SLICE", (slice(2, 4), slice(2, 6)))
    monkeypatch.setattr(flat_module, "A_OUTER_SLICE", (slice(1, 5), slice(6, 8)))
    monkeypatch.setattr(flat_module, "B_LEFT_SLICE", (slice(1, 5), slice(0, 2)))
    monkeypatch.setattr(flat_module, "C_INNER_SLICE", (slice(1, 5), slice(0, 2)))
    monkeypatch.setattr(flat_module, "B_RIGHT_SLICE", (slice(1, 5), slice(6, 8)))
    summed = np.empty((6, 8, 3), float)
    summed[:, :, 0] = 10
    summed[:, :, 1] = 2
    summed[:, :, 2] = 21
    summed[1:5, 6:8, 0] = 5
    summed[1:5, 0:2, 1] = 4
    summed[1:5, 6:8, 1] = 6
    summed[1:5, 0:2, 2] = 7
    result = flat_module.normalize_flat_chips(summed)
    assert result[0, 3, 1] == pytest.approx(1)
    assert result[0, 3, 0] == pytest.approx(4)
    assert result[0, 3, 2] == pytest.approx(9)


def test_normalize_flat_chips_does_not_modify_input(monkeypatch):
    monkeypatch.setattr(flat_module, "DETECTOR_SHAPE", (4, 4))
    for name in ["NORM_SLICE", "A_OUTER_SLICE", "B_LEFT_SLICE",
                 "C_INNER_SLICE", "B_RIGHT_SLICE"]:
        monkeypatch.setattr(flat_module, name, (slice(None), slice(None)))
    values = np.ones((4, 4, 3))
    original = values.copy()
    flat_module.normalize_flat_chips(values)
    np.testing.assert_array_equal(values, original)


def test_normalize_flat_chips_rejects_shape(monkeypatch):
    monkeypatch.setattr(flat_module, "DETECTOR_SHAPE", (4, 4))
    with pytest.raises(ValueError, match="shape"):
        flat_module.normalize_flat_chips(np.ones((4, 5, 3)))


def test_normalize_flat_chips_rejects_invalid_region(monkeypatch):
    monkeypatch.setattr(flat_module, "DETECTOR_SHAPE", (4, 4))
    monkeypatch.setattr(flat_module, "NORM_SLICE", (slice(None), slice(None)))
    values = np.ones((4, 4, 3))
    values[:, :, 1] = 0
    with pytest.raises(ValueError, match="middle-chip"):
        flat_module.normalize_flat_chips(values)


def test_make_flat_chip_requires_two_dimensions():
    with pytest.raises(ValueError, match="two-dimensional"):
        flat_module.make_flat_chip(
            np.ones((2, 3, 4)), np.zeros((2, 3, 4)), bad_pixel_bits=0)


def test_dithered_spectral_smoothing_runs_along_x(monkeypatch):
    calls = []
    original = flat_module.nan_uniform_filter

    def record(array, size):
        calls.append(size)
        return original(array, size)

    monkeypatch.setattr(flat_module, "nan_uniform_filter", fake_filter)
    flat_module.make_flat_chip(
        np.ones((40, 60)), np.zeros((40, 60), np.uint32),
        dithered=True, bad_pixel_bits=0)
    assert calls == [(100, 10), 100, (1, 50)]


def test_dithered_kludge_repairs_top_and_bottom_rows_not_columns():
    image = np.ones((40, 60))
    reduction_mask = np.zeros_like(image, np.uint32)
    reduction_mask[0, 30] = 1
    reduction_mask[-1, 30] = 1
    reduction_mask[20, 0] = 1
    output, _, mask = flat_module.make_flat_chip(
        image, reduction_mask, dithered=True, kludge=True, bad_pixel_bits=1)
    assert output[0, 30] == 1 and mask[0, 30] == 0
    assert output[-1, 30] == 1 and mask[-1, 30] == 0
    assert output[20, 0] == 0 and mask[20, 0] & 1


def test_non_dithered_spectral_profile_varies_with_x():
    x = np.arange(60, dtype=float)
    image = np.broadcast_to((1 + x / 100)[None, :], (40, 60)).copy()
    _, spectral, _ = flat_module.make_flat_chip(
        image, np.zeros_like(image, np.uint32), bad_pixel_bits=0)
    np.testing.assert_allclose(spectral[0], 1 + x / 100, rtol=1e-10)
    np.testing.assert_allclose(spectral[:, 10], spectral[0, 10])


def test_non_dithered_spectral_profile_falls_back_with_too_few_columns():
    image = np.ones((20, 2))
    _, spectral, _ = flat_module.make_flat_chip(
        image, np.zeros_like(image, np.uint32), bad_pixel_bits=0)
    np.testing.assert_array_equal(spectral, 1)


class FakeLoad:
    prefix = "ap"

    def __init__(self, root):
        self.root = Path(root)

    def filename(self, kind, num=None, chip=None, directory=False, **kwargs):
        if directory:
            return str(self.root)
        return str(self.root / f"ap{kind}-{chip}-{int(num):08d}.fits")

    def product_files(self, product, name):
        assert product == "flat"
        files = [self.filename("Flat", num=name, chip=chip)
                 for chip in flat_module.CHIPS]
        return files + [str(self.root / f"apFlat-{int(name):08d}.tab")]

    def product_exists(self, product, name):
        return all(Path(filename).is_file() and Path(filename).stat().st_size > 0
                   for filename in self.product_files(product, name))

    def product_delete(self, product, name, **kwargs):
        for filename in self.product_files(product, name):
            path = Path(filename)
            if path.exists() or path.is_symlink():
                path.unlink()


def test_add_provenance_records_dark_and_versions(monkeypatch):
    monkeypatch.setattr(flat_module.utils, "software_version", lambda: "git-test")
    monkeypatch.setattr(flat_module.utils, "reduction_version", lambda: "drp-test")
    monkeypatch.setattr(flat_module.getpass, "getuser", lambda: "observer")
    monkeypatch.setattr(flat_module.socket, "gethostname", lambda: "flat-host")
    header = fits.Header()
    flat_module._add_provenance(header, "/cal/apDark-a-12.fits", 34)
    assert header["DARKFILE"] == "apDark-a-12.fits"
    assert header["FLATID"] == 34
    assert header["V_APRED"] == "git-test"
    assert header["APRED"] == "drp-test"
    history = " ".join(header["HISTORY"])
    assert "observer on flat-host" in history
    assert "APOGEE Reduction Pipeline Version: drp-test" in history


def test_add_provenance_handles_no_dark(monkeypatch):
    monkeypatch.setattr(flat_module.utils, "software_version", lambda: "test")
    monkeypatch.setattr(flat_module.utils, "reduction_version", lambda: "test")
    header = fits.Header()
    flat_module._add_provenance(header, None, 34)
    assert header["DARKFILE"] == "NONE"


def test_obsolete_calibration_filename_helper_is_removed():
    assert not hasattr(flat_module, "_calibration_filename")


def test_process_flat_frames_forwards_calibrations(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    calls = []
    monkeypatch.setattr(
        flat_module.ap3d, "process_file",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    flat_module._process_flat_frames(
        load, [10, 11], detid=20, darkid=30, clobber=True, verbose=True)
    assert len(calls) == 6
    args, options = calls[0]
    assert Path(args[0]).name == "apR-a-00000010.fits"
    assert Path(args[1]).name == "ap2D-a-00000010.fits"
    assert Path(options["detector"]).name == "apDetector-a-00000020.fits"
    assert Path(options["dark"]).name == "apDark-a-00000030.fits"
    assert options["detect_cosmic_rays"] is False
    assert options["nfowler"] == 1


@pytest.mark.parametrize(
    "detid,darkid",
    [
        (None, None),
        (0, 0),
        (None, 30),
        (20, None),
    ],
)
def test_process_flat_frames_handles_optional_calibrations(
        monkeypatch, tmp_path, detid, darkid):
    """Optional calibration IDs are converted to filenames only when valid."""
    load = FakeLoad(tmp_path)
    calls = []
    monkeypatch.setattr(
        flat_module.ap3d, "process_file",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    flat_module._process_flat_frames(
        load, [10], detid=detid, darkid=darkid)

    assert len(calls) == 3
    for chip, (_, options) in zip(flat_module.CHIPS, calls):
        if detid is None or int(detid) <= 0:
            assert options["detector"] is None
        else:
            assert Path(options["detector"]).name == (
                f"apDetector-{chip}-{int(detid):08d}.fits"
            )

        if darkid is None or int(darkid) <= 0:
            assert options["dark"] is None
        else:
            assert Path(options["dark"]).name == (
                f"apDark-{chip}-{int(darkid):08d}.fits"
            )


def test_process_flat_frames_reuses_existing_2d(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    for chip in flat_module.CHIPS:
        Path(load.filename("2D", num=10, chip=chip)).touch()
    monkeypatch.setattr(
        flat_module.ap3d, "process_file",
        lambda *args, **kwargs: pytest.fail("existing ap2D should be reused"),
    )
    flat_module._process_flat_frames(load, [10])


def write_2d(load, number, chip, value, mask=0):
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.full((4, 5), value, np.float32)),
        fits.ImageHDU(np.ones((4, 5), np.float32)),
        fits.ImageHDU(np.full((4, 5), mask, np.uint32)),
    ]).writeto(load.filename("2D", num=number, chip=chip), overwrite=True)


def test_combine_flat_frames_medians_groups(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(flat_module, "DETECTOR_SHAPE", (4, 5))
    for number, value in zip([10, 11, 12], [1, 100, 3]):
        for chip in flat_module.CHIPS:
            write_2d(load, number, chip, value, mask=number)
    summed, masks, header = flat_module.combine_flat_frames(
        load, [10, 11, 12], nrep=2)
    np.testing.assert_allclose(summed, 53.5)
    np.testing.assert_array_equal(masks, 12)
    assert isinstance(header, fits.Header)


def test_combine_flat_frames_rejects_no_frames(monkeypatch, tmp_path):
    monkeypatch.setattr(flat_module, "DETECTOR_SHAPE", (4, 5))
    with pytest.raises(RuntimeError, match="No ap2D"):
        flat_module.combine_flat_frames(FakeLoad(tmp_path), [], 1)


@pytest.fixture
def flat_environment(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    lock_calls, plot_calls, html_calls = [], [], []
    monkeypatch.setattr(flat_module.apload, "ApLoad", lambda **kwargs: load)
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda filename, **kwargs: lock_calls.append((Path(filename), kwargs)),
    )
    monkeypatch.setattr(flat_module, "_process_flat_frames", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        flat_module, "combine_flat_frames",
        lambda *args, **kwargs: (
            np.ones((40, 60, 3)), np.zeros((40, 60, 3), np.uint32),
            fits.Header(),
        ),
    )
    monkeypatch.setattr(flat_module, "normalize_flat_chips", lambda values: values)
    monkeypatch.setattr(flat_module, "_add_provenance", lambda *args: None)
    monkeypatch.setattr(
        flat_module, "flatplot",
        lambda image, filename: plot_calls.append((image.shape, Path(filename))),
    )
    monkeypatch.setattr(
        flat_module, "flathtml",
        lambda directory, rows: html_calls.append((Path(directory), rows)),
    )
    return load, lock_calls, plot_calls, html_calls


def test_build_flat_writes_complete_product_set(flat_environment):
    load, lock_calls, plot_calls, html_calls = flat_environment
    outputs = load.product_files("flat", 12345678)[:3]
    assert flat_module.build_flat(
        [12345678, 12345679], detid=20, darkid=30, verbose=True) is None
    assert [Path(name).name for name in outputs] == [
        "apFlat-a-12345678.fits", "apFlat-b-12345678.fits",
        "apFlat-c-12345678.fits",
    ]
    for output in outputs:
        with fits.open(output) as hdul:
            assert [hdu.name for hdu in hdul] == [
                "PRIMARY", "FLAT", "SPECTRAL FLAT", "MASK"]
            assert hdul[1].data.shape == (40, 60)
            assert hdul[2].data.shape == (40, 60)
            assert hdul[3].data.shape == (40, 60)
    summary = load.root / "apFlat-12345678.tab"
    with fits.open(summary) as hdul:
        assert len(hdul[1].data) == 3
        np.testing.assert_array_equal(hdul[1].data["nframes"], 2)
    assert len(plot_calls) == 3
    assert len(html_calls) == 1
    assert lock_calls[-1][1] == {"clear": True}


def test_build_flat_reuses_complete_products(flat_environment, capsys,
                                             monkeypatch):
    load, lock_calls, _, _ = flat_environment
    outputs = [load.root / f"apFlat-{chip}-00000200.fits"
               for chip in flat_module.CHIPS]
    for output in outputs:
        output.write_bytes(b"existing")
    (load.root / "apFlat-00000200.tab").write_bytes(b"existing")
    monkeypatch.setattr(
        flat_module, "_process_flat_frames",
        lambda *args, **kwargs: pytest.fail("complete products must be reused"),
    )
    assert flat_module.build_flat([200], verbose=True) is None
    assert "flat product 200 already exists" in capsys.readouterr().out
    assert not any(options.get("lock") for _, options in lock_calls)


def test_build_flat_partial_products_are_rebuilt(flat_environment):
    load, _, _, _ = flat_environment
    partial = load.root / "apFlat-a-00000201.fits"
    partial.write_bytes(b"old")
    flat_module.build_flat([201])
    with fits.open(partial) as hdul:
        assert hdul[1].name == "FLAT"


def test_build_flat_clobber_rewrites_products(flat_environment):
    load, _, _, _ = flat_environment
    flat_module.build_flat([202])
    outputs = load.product_files("flat", 202)[:3]
    for output in outputs:
        Path(output).write_bytes(b"old")
    assert flat_module.build_flat([202], clobber=True) is None
    assert all(Path(output).stat().st_size > 3 for output in outputs)


def test_build_flat_clears_lock_after_failure(flat_environment, monkeypatch):
    _, lock_calls, _, _ = flat_environment
    monkeypatch.setattr(
        flat_module, "combine_flat_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad flat")),
    )
    with pytest.raises(RuntimeError, match="bad flat"):
        flat_module.build_flat([203])
    assert lock_calls[-1][1] == {"clear": True}

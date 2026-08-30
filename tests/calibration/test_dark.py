"""Tests for the numbered dark-calibration implementation."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import dark
from apogee_drp.apred.cal import utils as cal_utils


def make_ramps(nframe=3, nread=8, ny=16, nx=20, rate=2.0):
    reads = np.arange(nread, dtype=float)
    ramps = rate * reads[None, :, None, None]
    return np.broadcast_to(ramps, (nframe, nread, ny, nx)).copy()


def test_dark_variance_matches_idl_formula():
    data = np.array([-5.0, 0.0, 19.0])
    expected = np.maximum(data / 1.9, 0) + (18.0 / 1.9) ** 2
    np.testing.assert_allclose(dark.dark_variance(data), expected)


def test_dark_variance_scales_readnoise_with_nread():
    first = dark.dark_variance(0, nread=1)
    fourth = dark.dark_variance(0, nread=4)
    assert fourth == 4 * first


def test_make_dark_recovers_exact_ramp():
    ramps = make_ramps()
    model, chi2, mask, rate, stats = dark.combine_dark_ramps(
        ramps, row_block=5)
    np.testing.assert_allclose(model, ramps[0])
    np.testing.assert_allclose(chi2, 0)
    np.testing.assert_allclose(rate, 2.0)
    assert not np.any(mask)
    assert stats["nframes"] == 3
    assert stats["nreads"] == 8


def test_make_dark_uses_frame_median():
    ramps = make_ramps()
    ramps[0] += 1000
    model, _, _, _, _ = dark.combine_dark_ramps(ramps)
    np.testing.assert_allclose(model, ramps[1])


def test_make_dark_flags_hot_pixel_and_neighbor():
    ramps = make_ramps(rate=1.0)
    reads = np.arange(ramps.shape[1], dtype=float)
    ramps[:, :, 8, 8] = 20.0 * reads[None, :]
    ramps[:, :, 8, 9] = 3.0 * reads[None, :]
    _, _, mask, rate, stats = dark.combine_dark_ramps(ramps)
    assert rate[8, 8] > 10
    assert mask[8, 8] & 2
    assert mask[8, 9] & 4
    assert stats["nhot"] == 1
    assert stats["nhotneigh"] == 8


def test_make_dark_flags_nonfinite_rate():
    ramps = make_ramps()
    ramps[:, -1, 4, 5] = np.nan
    _, _, mask, _, stats = dark.combine_dark_ramps(ramps)
    assert mask[4, 5] & 1
    assert stats["nsat"] == 1


def test_make_dark_does_not_modify_input():
    ramps = make_ramps()
    original = ramps.copy()
    dark.combine_dark_ramps(ramps)
    np.testing.assert_array_equal(ramps, original)


@pytest.mark.parametrize("shape", [(3, 8, 10), (0, 8, 10, 10),
                                    (2, 2, 10, 10)])
def test_make_dark_rejects_invalid_ramps(shape):
    with pytest.raises(ValueError):
        dark.combine_dark_ramps(np.zeros(shape))


def test_build_dark_validates_empty_input_before_io():
    with pytest.raises(ValueError, match="at least one"):
        dark.build_dark([])


def test_build_dark_rejects_unported_psf_branch_before_io():
    with pytest.raises(NotImplementedError, match="psfid"):
        dark.build_dark([12345678], psfid=12345679)


def test_combine_dark_ramps_chi2_detects_frame_scatter():
    ramps = make_ramps(nframe=3)
    ramps[0, 4, 3, 7] += 30
    _, chi2, _, _, _ = dark.combine_dark_ramps(ramps)
    assert chi2[4, 3, 7] > 0
    assert np.count_nonzero(chi2) == 1


def test_combine_dark_ramps_flags_only_cross_neighbors():
    ramps = make_ramps(rate=1.0)
    reads = np.arange(ramps.shape[1], dtype=float)
    ramps[:, :, 8, 8] = 20 * reads
    for y, x in [(7, 8), (9, 8), (8, 7), (8, 9), (7, 7)]:
        ramps[:, :, y, x] = 3 * reads
    _, _, mask, _, stats = dark.combine_dark_ramps(ramps)
    assert np.all((mask[[7, 9, 8, 8], [8, 8, 7, 9]] & 4) != 0)
    assert mask[7, 7] == 0
    # IDL records 8 * NHOT, not the number of flagged neighbors.
    assert stats["nhot"] == 1
    assert stats["nhotneigh"] == 8


def test_nonfinite_rate_does_not_bias_median_rate():
    ramps = make_ramps(rate=2.0)
    ramps[:, -1, 0, 0] = np.nan
    _, _, _, rate, stats = dark.combine_dark_ramps(ramps)
    assert rate[0, 0] == 0
    assert stats["medrate"] == pytest.approx(2.0)


def test_combine_dark_ramps_zeros_strongly_negative_values():
    ramps = make_ramps(rate=0.0)
    ramps[:, :, 2, 3] = -20
    model, _, _, _, stats = dark.combine_dark_ramps(ramps)
    assert np.all(model[:, 2, 3] == 0)
    assert stats["nneg"] == ramps.shape[1]


def test_combine_dark_ramps_matches_idl_rejected_read_behavior():
    ramps = make_ramps(nframe=3, nread=8, ny=4, nx=5)
    original = ramps.copy()
    read_masks = np.zeros((3, 8), dtype=bool)
    read_masks[:, 0] = True

    model, chi2, _, _, stats = dark.combine_dark_ramps(
        ramps, read_masks=read_masks)

    # IDL's invalid LONG plane contributes to NNEG and is then zeroed.
    assert stats["nneg"] == 4 * 5
    np.testing.assert_array_equal(model[0], 0)
    np.testing.assert_array_equal(chi2[0], 0)
    np.testing.assert_allclose(model[1:], ramps[0, 1:])
    np.testing.assert_array_equal(ramps, original)


def test_combine_dark_ramps_rejects_bad_read_mask_shape():
    with pytest.raises(ValueError, match="read_masks"):
        dark.combine_dark_ramps(
            make_ramps(), read_masks=np.zeros((3, 7), dtype=bool))


def test_combine_dark_ramps_row_block_does_not_change_result():
    rng = np.random.default_rng(12)
    ramps = make_ramps(nframe=5, ny=11, nx=9)
    ramps += rng.normal(0, 0.2, ramps.shape)
    small = dark.combine_dark_ramps(ramps, row_block=2)
    large = dark.combine_dark_ramps(ramps, row_block=50)
    for left, right in zip(small[:4], large[:4]):
        np.testing.assert_allclose(left, right)
    assert small[4] == large[4]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"row_block": 0}, "row_block"),
        ({"gain": 0}, "gain"),
        ({"readnoise": -1}, "readnoise"),
        ({"maxrate": 0}, "maxrate"),
    ],
)
def test_combine_dark_ramps_rejects_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        dark.combine_dark_ramps(make_ramps(), **kwargs)


def test_load_ramps_creates_memmap(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    load_calls = []
    correction_calls = []

    def load_raw(filename, **kwargs):
        load_calls.append((filename, kwargs))
        offset = 10 * len(load_calls)
        cube = (np.arange(4, dtype=float)[:, None, None]
                + np.zeros((4, 3, 2)) + offset)
        return cube, fits.Header({"TEST": len(load_calls)})

    def correct(cube, header, **kwargs):
        correction_calls.append(kwargs)
        read_mask = np.array([True, False, False, False])
        return cube, None, read_mask, None

    monkeypatch.setattr(dark.ap3d, "load_raw_ramp", load_raw)
    monkeypatch.setattr(dark.ap3d, "reference_correct", correct)
    ramps, header, read_masks = dark._load_ramps(
        load, [11, 12], "b", tmp_path, max_read=7,
        unlock=True, verbose=True)
    try:
        assert isinstance(ramps, np.memmap)
        assert ramps.shape == (2, 4, 3, 2)
        np.testing.assert_array_equal(
            ramps[:, :, 0, 0],
            [[10, 0, 1, 2], [20, 0, 1, 2]])
        assert header["TEST"] == 1
        np.testing.assert_array_equal(
            read_masks,
            [[True, False, False, False],
             [True, False, False, False]],
        )
    finally:
        dark._close_memmap(ramps)
    assert Path(load_calls[0][0]).name == "apR-b-00000011.fits"
    assert load_calls[0][1] == {
        "max_read": 7, "temporary_directory": tmp_path,
        "unlock": True, "verbose": True,
    }
    assert correction_calls == [
        {"indiv": 3},
        {"indiv": 3},
    ]

    
def test_load_ramps_rejects_inconsistent_shapes(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    shapes = iter([(4, 3, 2), (5, 3, 2)])
    monkeypatch.setattr(dark.ap3d, "load_raw_ramp",
                        lambda *args, **kwargs:
                        (np.zeros(next(shapes)), fits.Header()))
    monkeypatch.setattr(dark.ap3d, "reference_correct",
                        lambda cube, header, **kwargs:
                        (cube, None, None, None))
    with pytest.raises(ValueError, match="same shape"):
        dark._load_ramps(load, [11, 12], "a", tmp_path)


def test_load_ramps_rejects_empty_image_list(tmp_path):
    with pytest.raises(RuntimeError, match="No dark ramps"):
        dark._load_ramps(FakeLoad(tmp_path), [], "a", tmp_path)


def test_add_provenance_records_versions(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(dark.utils, "software_version", lambda: "git-test")
    monkeypatch.setattr(dark.utils, "reduction_version", lambda: "drp-test")
    monkeypatch.setattr(dark.getpass, "getuser", lambda: "observer")
    monkeypatch.setattr(dark.socket, "gethostname", lambda: "reduction-host")
    header = fits.Header()
    dark._add_provenance(header, 123, load)
    assert header["DARKID"] == 123
    assert header["V_APRED"] == "git-test"
    assert header["APRED"] == "drp-test"
    history = " ".join(header["HISTORY"])
    assert "observer on reduction-host" in history
    assert "APOGEE Reduction Pipeline Version: daily" in history


class FakeLoad:
    prefix = "ap"
    apred = "daily"

    def __init__(self, root):
        self.root = Path(root)

    def filename(self, kind, num=None, chip=None, directory=False, **kwargs):
        if directory:
            return str(self.root)
        infix = f"-{chip}" if chip is not None else ""
        return str(self.root / f"ap{kind}{infix}-{int(num):08d}.fits")

    def product_files(self, product, name):
        assert product == "dark"
        files = [self.filename("Dark", num=name, chip=chip)
                 for chip in dark.CHIPS]
        return files + [str(self.root / f"apDark-{int(name):08d}.tab")]

    def product_exists(self, product, name):
        return all(Path(filename).is_file() and Path(filename).stat().st_size > 0
                   for filename in self.product_files(product, name))

    def product_delete(self, product, name, **kwargs):
        for filename in self.product_files(product, name):
            path = Path(filename)
            if path.exists() or path.is_symlink():
                path.unlink()


@pytest.fixture
def dark_environment(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    lock_calls = []
    plot_calls = []
    html_calls = []
    monkeypatch.setattr(dark.apload, "ApLoad", lambda **kwargs: load)
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda filename, **kwargs: lock_calls.append((Path(filename), kwargs)),
    )
    monkeypatch.setattr(
        dark, "_load_ramps",
        lambda load, images, chip, directory, **kwargs:
        (make_ramps(nframe=len(images), ny=6, nx=7), fits.Header(),
         np.zeros((len(images), 8), dtype=bool)),
    )
    monkeypatch.setattr(
        dark, "darkplot",
        lambda cube, mask, filename: plot_calls.append(
            (cube.shape, mask.shape, Path(filename))),
    )
    monkeypatch.setattr(
        dark, "darkhtml",
        lambda directory, rows: html_calls.append((Path(directory), rows)),
    )
    monkeypatch.setattr(dark, "_add_provenance", lambda *args: None)
    return load, lock_calls, plot_calls, html_calls


def test_build_dark_writes_complete_product_set(dark_environment):
    load, lock_calls, plot_calls, html_calls = dark_environment
    outputs = load.product_files("dark", 12345678)[:3]
    assert dark.build_dark([12345678, 12345679], verbose=True) is None
    assert [Path(name).name for name in outputs] == [
        "apDark-a-12345678.fits", "apDark-b-12345678.fits",
        "apDark-c-12345678.fits",
    ]
    for output in outputs:
        with fits.open(output) as hdul:
            assert [hdu.name for hdu in hdul] == [
                "PRIMARY", "DARK", "CHI-SQUARED", "MASK"]
            assert hdul[1].data.shape == (8, 6, 7)
            assert hdul[2].data.shape == (8, 6, 7)
            assert hdul[3].data.shape == (6, 7)
    for chip in dark.CHIPS:
        ratefile = load.root / f"apDarkRate-{chip}-12345678.fits"
        assert fits.getdata(ratefile).shape == (6, 7)
    summary = load.root / "apDark-12345678.tab"
    with fits.open(summary) as hdul:
        table = hdul[1].data
        assert len(table) == 3
        np.testing.assert_array_equal(table["nframes"], 2)
        np.testing.assert_array_equal(table["nreads"], 8)
    assert len(plot_calls) == 3
    assert len(html_calls) == 1
    assert lock_calls[-1][1] == {"clear": True}


def test_build_dark_reuses_complete_products(
        dark_environment, capsys, monkeypatch):
    load, lock_calls, _, _ = dark_environment
    outputs = [load.root / f"apDark-{chip}-00000200.fits" for chip in dark.CHIPS]
    for output in outputs:
        output.write_bytes(b"existing")
    (load.root / "apDark-00000200.tab").write_bytes(b"existing")
    monkeypatch.setattr(
        dark, "_load_ramps",
        lambda *args, **kwargs: pytest.fail("existing products must be reused"),
    )
    assert dark.build_dark([200], verbose=True) is None
    assert "dark product 200 already exists" in capsys.readouterr().out
    assert not any(options.get("lock") for _, options in lock_calls)


def test_build_dark_partial_product_set_is_rebuilt(dark_environment):
    load, _, _, _ = dark_environment
    partial = load.root / "apDark-a-00000201.fits"
    partial.write_bytes(b"old")
    dark.build_dark([201])
    with fits.open(partial) as hdul:
        assert hdul[1].name == "DARK"


def test_build_dark_clobber_rewrites_existing_products(dark_environment):
    load, _, _, _ = dark_environment
    dark.build_dark([202])
    first = load.product_files("dark", 202)[:3]
    for filename in first:
        Path(filename).write_bytes(b"old")
    assert dark.build_dark([202], clobber=True) is None
    assert all(Path(filename).stat().st_size > 3 for filename in first)


def test_build_dark_clears_lock_after_failure(dark_environment, monkeypatch):
    _, lock_calls, _, _ = dark_environment
    monkeypatch.setattr(
        dark, "_load_ramps",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad ramp")),
    )
    with pytest.raises(RuntimeError, match="bad ramp"):
        dark.build_dark([203])
    assert lock_calls[-1][1] == {"clear": True}


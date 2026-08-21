"""Focused unit tests for :mod:`apogee_drp.utils.apload`.

These tests deliberately mock ``sdss_access``.  They test ApLoad's routing
logic without requiring an SDSS tree installation, calibration files, or
network access.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.utils import apload


@pytest.fixture
def load():
    """Return an ApLoad instance without running its environment-heavy init."""
    obj = object.__new__(apload.ApLoad)
    obj.apred = "daily"
    obj.apstar = "stars"
    obj.aspcap = "aspcap"
    obj.results = "results"
    obj.telescope = "apo25m"
    obj.instrument = "apogee-n"
    obj.observatory = "apo"
    obj.verbose = False
    obj.sdss_path = MagicMock()
    obj.sdss_path.full.side_effect = _fake_full
    obj.sdss_path.url.side_effect = (
        lambda root, **kwargs: "https://example.test/" + _fake_full(root, **kwargs)
    )
    # http_access is a read-only property backed by _http_access.
    obj._http_access = MagicMock()
    obj.cmjd = MagicMock(return_value="60000")
    return obj


@pytest.fixture
def frame_load(load, tmp_path, monkeypatch):
    """Create a lightweight loader backed by synthetic ap2D files."""
    def filename(kind, num=None, chip=None, **kwargs):
        assert kind == "2D"
        return str(tmp_path / f"ap2D-{chip}-{int(num):08d}.fits")

    monkeypatch.setattr(load, "filename", filename)
    for index, chip in enumerate("abc"):
        path = Path(filename("2D", num=123, chip=chip))
        fits.HDUList([
            fits.PrimaryHDU(header=fits.Header({"CHIP": chip})),
            fits.ImageHDU(np.full((2, 3), index + 1.0)),
            fits.ImageHDU(np.full((2, 3), index + 0.1)),
            fits.ImageHDU(np.full((2, 3), index, dtype=np.uint16)),
        ]).writeto(path)
    return load


def _fake_full(root, **kwargs):
    """Make a recognizable fake path from the arguments passed to Path.full."""
    chip = kwargs.get("chip")
    number = kwargs.get("num")
    chip_part = f"-{chip}" if chip is not None else ""
    number_part = f"-{number}" if number is not None else ""
    return f"/redux/{root}{chip_part}{number_part}.fits"


def test_filename_chipless(load):
    result = load.filename("Dark", num=123)

    assert result == "/redux/apDark-123.fits"
    assert load.sdss_path.full.call_args.kwargs["chip"] is None


@pytest.mark.parametrize("chip", ["a", "b", "c"])
def test_filename_single_chip(load, chip):
    result = load.filename("Dark", num=123, chip=chip)

    assert result == f"/redux/apDark-{chip}-123.fits"
    assert load.sdss_path.full.call_args.kwargs["chip"] == chip


@pytest.mark.parametrize(
    "chips",
    [
        ["a", "b", "c"],
        ("a", "b", "c"),
        np.array(["a", "b", "c"]),
    ],
)
def test_filename_chip_sequence(load, chips):
    result = load.filename("Dark", num=123, chip=chips)

    assert result == {
        "a": "/redux/apDark-a-123.fits",
        "b": "/redux/apDark-b-123.fits",
        "c": "/redux/apDark-c-123.fits",
    }


def test_filename_chip_subset_preserves_order(load):
    result = load.filename("Dark", num=123, chip=["c", "a"])

    assert list(result) == ["c", "a"]
    assert list(result.values()) == [
        "/redux/apDark-c-123.fits",
        "/redux/apDark-a-123.fits",
    ]


def test_filename_empty_chip_sequence(load):
    with pytest.raises(ValueError, match="chip sequence cannot be empty"):
        load.filename("Dark", num=123, chip=[])
    

@pytest.mark.parametrize("chip", ["d", "A", "", 1])
def test_filename_rejects_invalid_scalar_chip(load, chip):
    with pytest.raises(ValueError, match="Invalid chip"):
        load.filename("Dark", num=123, chip=chip)

    load.sdss_path.full.assert_not_called()


def test_filename_forwards_public_arguments(load):
    load.filename(
        "Visit",
        location=42,
        obj="2M00000000+0000000",
        plate=1234,
        mjd=60000,
        num=5678,
        fiber=9,
        chip="b",
        field="test-field",
        configid=321,
        fps=True,
    )

    kwargs = load.sdss_path.full.call_args.kwargs
    assert kwargs["location"] == 42
    assert kwargs["obj"] == "2M00000000+0000000"
    assert kwargs["plate"] == 1234
    assert kwargs["mjd"] == 60000
    assert kwargs["num"] == 5678
    assert kwargs["fiber"] == 9
    assert kwargs["chip"] == "b"
    assert kwargs["field"] == "test-field"
    assert kwargs["configid"] == 321


@pytest.mark.parametrize(
    ("root", "expected_sdssroot"),
    [
        ("Dark", "apDark"),
        ("allStar", "allStar"),
        ("aspcapStar", "aspcapStar"),
        ("cannonStar", "cannonStar"),
        ("confSummary", "confSummary"),
    ],
)
def test_allfile_selects_sdss_root(load, root, expected_sdssroot):
    load.allfile(root, num=123, chip="a")

    assert load.sdss_path.full.call_args.args == (expected_sdssroot,)


def test_allfile_selects_southern_raw_root(load):
    load.instrument = "apogee-s"
    load.telescope = "lco25m"
    load.observatory = "lco"

    load.allfile("R", num=123, chip="a")

    assert load.sdss_path.full.call_args.args == ("asR",)
    assert load.sdss_path.full.call_args.kwargs["prefix"] == "as"
    load.cmjd.assert_called_once_with(123)


@pytest.mark.parametrize("root", ["R", "2D", "1D"])
def test_allfile_derives_mjd_for_exposure_products(load, root):
    load.allfile(root, num=123, chip="a")

    load.cmjd.assert_called_once_with(123)
    assert load.sdss_path.full.call_args.kwargs["mjd"] == "60000"


def test_allfile_keeps_explicit_mjd(load):
    load.allfile("R", num=123, mjd=59999, chip="a")

    load.cmjd.assert_not_called()
    assert load.sdss_path.full.call_args.kwargs["mjd"] == 59999


@pytest.mark.parametrize(
    "root",
    ["Plan", "PlateSum", "Visit", "VisitSum", "Tellstar", "Cframe", "Plate"],
)
def test_allfile_selects_apo1m_root(load, root):
    load.telescope = "apo1m"

    load.allfile(root, num=123)

    assert load.sdss_path.full.call_args.args == (f"ap{root}-1m",)


def test_allfile_infers_fps_and_field(load, monkeypatch):
    mock_apfield = MagicMock(return_value=("field-1234",))
    monkeypatch.setattr(apload, "apfield", mock_apfield)

    load.allfile("Visit", plate=1234, mjd=60000)

    mock_apfield.assert_called_once_with(1234, telescope="apo25m", fps=True)
    assert load.sdss_path.full.call_args.kwargs["field"] == "field-1234"


@pytest.mark.parametrize(
    ("mjd", "expected_fps"),
    [(59555, False), (59556, True), (60000, True)],
)
def test_allfile_fps_boundary(load, monkeypatch, mjd, expected_fps):
    mock_apfield = MagicMock(return_value=("field",))
    monkeypatch.setattr(apload, "apfield", mock_apfield)

    load.allfile("Visit", plate=1234, mjd=mjd)

    assert mock_apfield.call_args.kwargs["fps"] is expected_fps


def test_allfile_does_not_replace_explicit_field(load, monkeypatch):
    mock_apfield = MagicMock()
    monkeypatch.setattr(apload, "apfield", mock_apfield)

    load.allfile("Visit", plate=1234, mjd=60000, field="given-field")

    mock_apfield.assert_not_called()
    assert load.sdss_path.full.call_args.kwargs["field"] == "given-field"


def test_allfile_calculates_star_healpix(load, monkeypatch):
    mock_obj2healpix = MagicMock(return_value=12345)
    monkeypatch.setattr(apload, "obj2healpix", mock_obj2healpix)

    load.allfile("Star", obj="2M00000000+0000000")

    mock_obj2healpix.assert_called_once_with("2M00000000+0000000")
    assert load.sdss_path.full.call_args.kwargs["healpix"] == 12345


def test_allfile_keeps_explicit_star_healpix(load, monkeypatch):
    mock_obj2healpix = MagicMock()
    monkeypatch.setattr(apload, "obj2healpix", mock_obj2healpix)

    load.allfile("Star", obj="2M00000000+0000000", healpix=54321)

    mock_obj2healpix.assert_not_called()
    assert load.sdss_path.full.call_args.kwargs["healpix"] == 54321


def test_allfile_adds_fz_suffix(load):
    result = load.allfile("2Dmodel", num=123, chip="a", fz=True)

    assert result.endswith(".fits.fz")


def test_allfile_does_not_download_by_default(load, monkeypatch):
    monkeypatch.setattr(apload.os.path, "exists", MagicMock(return_value=False))

    load.allfile("Dark", num=123, chip="a")

    load.http_access.get.assert_not_called()
    load.sdss_path.url.assert_not_called()


def test_allfile_does_not_download_existing_file(load, monkeypatch):
    monkeypatch.setattr(apload.os.path, "exists", MagicMock(return_value=True))

    load.allfile("Dark", num=123, chip="a", download=True)

    load.http_access.get.assert_not_called()


def test_allfile_downloads_missing_file(load, monkeypatch):
    monkeypatch.setattr(apload.os.path, "exists", MagicMock(return_value=False))

    result = load.allfile("Dark", num=123, chip="b", download=True)

    assert result == "/redux/apDark-b-123.fits"
    load.http_access.get.assert_called_once()
    assert load.http_access.get.call_args.args == ("apDark",)
    assert load.http_access.get.call_args.kwargs["chip"] == "b"


def test_allfile_verbose_reports_local_and_download_paths(
    load, monkeypatch, capsys
):
    load.verbose = True
    monkeypatch.setattr(apload.os.path, "exists", MagicMock(return_value=False))

    load.allfile("Dark", num=123, chip="a", download=True)

    output = capsys.readouterr().out
    assert "allfile: root= Dark chip= a" in output
    assert "filePath:" in output
    assert "downloadPath:" in output
    load.sdss_path.url.assert_called_once()


def test_allfile_passes_same_arguments_to_path_and_download(load, monkeypatch):
    monkeypatch.setattr(apload.os.path, "exists", MagicMock(return_value=False))

    load.allfile("Dark", num=123, chip="c", download=True)

    full_kwargs = load.sdss_path.full.call_args.kwargs
    get_kwargs = load.http_access.get.call_args.kwargs
    assert get_kwargs == full_kwargs
class TestFrame:
    def test_loads_all_chips_by_default(self, frame_load):
        frames = frame_load.frame(123)

        assert set(frames) == {"a", "b", "c"}
        assert frames["b"]["header"]["CHIP"] == "b"
        np.testing.assert_array_equal(frames["c"]["flux"], 3)
        np.testing.assert_array_equal(frames["c"]["err"], 2.1)
        np.testing.assert_array_equal(frames["c"]["mask"], 2)

    def test_can_load_one_chip(self, frame_load):
        frame = frame_load.frame(123, chip="b")

        assert set(frame) == {"header", "flux", "err", "mask"}
        np.testing.assert_array_equal(frame["flux"], 2)

    def test_rejects_invalid_chip(self, frame_load):
        with pytest.raises(ValueError, match="chip"):
            frame_load.frame(123, chip="d")

    def test_arrays_remain_available_after_file_closes(self, frame_load):
        frame = frame_load.frame(123, chip="a")

        assert frame["flux"].flags.owndata
        np.testing.assert_array_equal(frame["flux"], 1)

    def test_requires_all_extensions(self, frame_load):
        filename = Path(
            frame_load.filename("2D", num=124, chip="a")
        )
        fits.PrimaryHDU().writeto(filename)

        with pytest.raises(ValueError, match="required extensions"):
            frame_load.frame(124, chip="a")

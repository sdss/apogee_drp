"""Focused unit tests for :mod:`apogee_drp.utils.apload`.

These tests deliberately mock ``sdss_access``.  They test ApLoad's routing
logic without requiring an SDSS tree installation, calibration files, or
network access.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

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
    obj.http_access = MagicMock()
    obj.cmjd = MagicMock(return_value="60000")
    return obj


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
    assert load.filename("Dark", num=123, chip=[]) == {}
    load.sdss_path.full.assert_not_called()


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

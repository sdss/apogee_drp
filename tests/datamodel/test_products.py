"""Tests for the APOGEE logical-product registry."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from apogee_drp.datamodel import products


@pytest.fixture
def load():
    obj = MagicMock()
    obj.cmjd.side_effect = lambda number: str(number)[:5]

    def filename(root, *, num=None, mjd=None, chip=None, **kwargs):
        stem = f"ap{root}"
        if isinstance(chip, (list, tuple)):
            return {
                value: f"/cal/{stem}-{value}-{num}.fits"
                for value in chip
            }
        middle = f"-{chip}" if chip is not None else ""
        return f"/cal/{stem}{middle}-{num}.fits"

    obj.filename.side_effect = filename
    return obj


def test_registry_covers_every_makecal_product():
    assert set(products.PRODUCTS) == {
        "detector", "dark", "flat", "bpm", "fiber", "sparse",
        "littrow", "psf", "modelpsf", "fpi", "persist",
        "persistmodel", "flux", "response", "wave", "multiwave",
        "dailywave", "telluric", "lsf",
    }


def test_standard_three_chip_product(load):
    assert products.product_files(load, "detector", 12345678) == [
        "/cal/apDetector-a-12345678.fits",
        "/cal/apDetector-b-12345678.fits",
        "/cal/apDetector-c-12345678.fits",
    ]


@pytest.mark.parametrize(
    ("product", "root"),
    [
        ("bpm", "BPM"),
        ("fiber", "Fiber"),
        ("fpi", "WaveFPI"),
        ("persist", "Persist"),
        ("persistmodel", "PersistModel"),
        ("flux", "Flux"),
        ("response", "Response"),
        ("wave", "Wave"),
        ("multiwave", "Wave"),
    ],
)
def test_simple_calibration_products(load, product, root):
    files = products.product_files(load, product, 12345678)
    assert len(files) == 3
    assert files[0] == f"/cal/ap{root}-a-12345678.fits"
    assert files[-1] == f"/cal/ap{root}-c-12345678.fits"


def test_dark_includes_tab_summary(load):
    files = products.product_files(load, "dark", 12345678)
    assert files[-1] == "/cal/apDark-12345678.tab"
    assert len(files) == 4


def test_flat_includes_tab_summary(load):
    files = products.product_files(load, "flat", 12345678)
    assert files[-1] == "/cal/apFlat-12345678.tab"
    assert len(files) == 4


def test_sparse_has_chipless_sparse_and_three_epsf_files(load):
    assert products.product_files(load, "sparse", 12345678) == [
        "/cal/apSparse-12345678.fits",
        "/cal/apEPSF-a-12345678.fits",
        "/cal/apEPSF-b-12345678.fits",
        "/cal/apEPSF-c-12345678.fits",
    ]


def test_littrow_requires_only_chip_b(load):
    assert products.product_files(load, "littrow", 12345678) == [
        "/cal/apLittrow-b-12345678.fits"
    ]


def test_psf_has_all_current_component_files(load):
    files = products.product_files(load, "psf", 12345678)
    assert len(files) == 9
    assert files[:3] == [
        "/cal/apPSF-a-12345678.fits",
        "/cal/apPSF-b-12345678.fits",
        "/cal/apPSF-c-12345678.fits",
    ]
    assert files[3:6] == [
        "/cal/apEPSF-a-12345678.fits",
        "/cal/apEPSF-b-12345678.fits",
        "/cal/apEPSF-c-12345678.fits",
    ]
    assert files[6:] == [
        "/cal/apETrace-a-12345678.fits",
        "/cal/apETrace-b-12345678.fits",
        "/cal/apETrace-c-12345678.fits",
    ]


def test_modelpsf_preserves_compound_identifier(load):
    files = products.product_files(load, "modelpsf", "12345678-87654321")
    assert files == [
        "/cal/apPSFModel-a-12345678-87654321.fits",
        "/cal/apPSFModel-b-12345678-87654321.fits",
        "/cal/apPSFModel-c-12345678-87654321.fits",
    ]
    load.cmjd.assert_called_once_with(12345678)


def test_dailywave_uses_name_as_mjd(load):
    products.product_files(load, "dailywave", 60000)
    load.cmjd.assert_not_called()
    assert load.filename.call_args.kwargs["mjd"] == 60000


def test_lsf_includes_diagnostics(load):
    files = products.product_files(load, "lsf", 12345678)
    assert files[-1] == "/cal/apLSF-12345678-diagnostics.fits"
    assert len(files) == 4


def test_explicit_mjd_bypasses_cmjd(load):
    products.product_files(load, "dark", 12345678, mjd=60001)
    load.cmjd.assert_not_called()
    assert all(
        item.kwargs["mjd"] == 60001
        for item in load.filename.call_args_list
    )


def test_product_names_are_case_insensitive(load):
    assert products.product_files(load, "DETECTOR", 12345678) == (
        products.product_files(load, "detector", 12345678)
    )


def test_unknown_product_has_clear_error():
    with pytest.raises(ValueError, match="Unknown APOGEE product"):
        products.product_spec("not-a-product")


def test_empty_compound_identifier_is_rejected(load):
    with pytest.raises(ValueError, match="empty identifier"):
        products.product_files(load, "modelpsf", "")


@pytest.mark.parametrize("require_nonempty", [False, True])
def test_file_is_complete_for_nonempty_file(tmp_path, require_nonempty):
    filename = tmp_path / "product.fits"
    filename.write_bytes(b"FITS")
    assert products.file_is_complete(
        filename, require_nonempty=require_nonempty)


def test_file_is_complete_rejects_missing_file(tmp_path):
    assert not products.file_is_complete(tmp_path / "missing.fits")


def test_file_is_complete_rejects_empty_file_by_default(tmp_path):
    filename = tmp_path / "empty.fits"
    filename.touch()
    assert not products.file_is_complete(filename)
    assert products.file_is_complete(filename, require_nonempty=False)


def test_product_status_reports_each_file(load, monkeypatch):
    monkeypatch.setattr(
        products,
        "file_is_complete",
        lambda filename, require_nonempty=True: "-b-" not in str(filename),
    )
    status = products.product_status(load, "detector", 12345678)
    assert status == {
        "/cal/apDetector-a-12345678.fits": True,
        "/cal/apDetector-b-12345678.fits": False,
        "/cal/apDetector-c-12345678.fits": True,
    }


def test_product_exists_requires_every_file(load, monkeypatch):
    monkeypatch.setattr(
        products,
        "file_is_complete",
        lambda filename, require_nonempty=True: "-b-" not in str(filename),
    )
    assert not products.product_exists(load, "detector", 12345678)


def test_product_exists_when_every_file_is_complete(load, monkeypatch):
    monkeypatch.setattr(products, "file_is_complete", lambda *args, **kwargs: True)
    assert products.product_exists(load, "detector", 12345678)


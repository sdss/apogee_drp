"""Tests for calibration plots and HTML diagnostics."""

import numpy as np

from apogee_drp.apred.cal.diagnostics import (
    darkhtml,
    darkplot,
    flathtml,
    flatplot,
)
from apogee_drp.apred.cal.getrn import rnhtml


def test_flatplot_writes_nonempty_jpeg(tmp_path):
    image = np.linspace(0, 2, 64 * 64).reshape(64, 64)
    output = flatplot(image, tmp_path / "flat")
    assert output.suffix == ".jpg"
    assert output.stat().st_size > 0


def test_darkplot_writes_plot_and_image(tmp_path):
    rng = np.random.default_rng(2)
    cube = np.cumsum(rng.normal(5, 1, (40, 50, 10)), axis=-1)
    mask = np.zeros((40, 50), dtype=np.int32)
    plot, image = darkplot(cube, mask, tmp_path / "dark")
    assert plot.exists() and plot.stat().st_size > 0
    assert image.exists() and image.stat().st_size > 0


def test_darkplot_rejects_incompatible_shapes(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        darkplot(np.zeros((5, 6, 7)), np.zeros((6, 5)), tmp_path / "dark")


def test_darkhtml_contains_rows_and_links(tmp_path):
    rows = [{
        "num": 123, "chip": "a", "nreads": 10, "nframes": 3,
        "medrate": 0.2, "nsat": 1, "nhot": 2, "nhotneigh": 3,
        "nbad": 4, "nneg": 5,
    }]
    output = darkhtml(tmp_path, rows)
    text = output.read_text()
    assert "apDark-a-00000123" in text
    assert "<td>10</td>" in text


def test_darkhtml_escapes_text(tmp_path):
    rows = [{
        "num": 123, "chip": "<a>", "nreads": 10, "nframes": 3,
        "medrate": 0.2, "nsat": 1, "nhot": 2, "nhotneigh": 3,
        "nbad": 4, "nneg": 5,
    }]
    text = darkhtml(tmp_path, rows).read_text()
    assert "&lt;a&gt;" in text


def test_flathtml_has_three_chip_thumbnails(tmp_path):
    output = flathtml(tmp_path, [{"num": 42, "nframes": 5}])
    text = output.read_text()
    for chip in "abc":
        assert f"apFlat-{chip}-00000042.jpg" in text


def test_rnhtml_contains_all_sampling_columns(tmp_path):
    dtype = [("rn2", "f4", (4, 6))]
    table = np.zeros(3, dtype=dtype)
    output = rnhtml([("apRN-test", table)], tmp_path / "html" / "rn.html")
    text = output.read_text()
    assert "FS 0, Q0" in text
    assert "FS 5, Q3" in text
    assert text.count("apRN-test") == 3


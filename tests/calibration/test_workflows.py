"""Tests for calibration selection and command wrappers."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apogee_drp.apred.cal.getpsfcal import _parse_psflibrary, getpsfcal
from apogee_drp.apred.cal.makehist import makehist
from apogee_drp.apred.cal.run_multi_apwavecal import (
    build_multiwave_lines,
    run_multi_apwavecal,
)


def test_parse_psflibrary_result():
    output = """header
PSF FLAT RESULTS:
40620001 40620003 2
40620002 40620003 1

footer
"""
    assert _parse_psflibrary(output) == 40620003


@pytest.mark.parametrize("output", ["", "PSF FLAT RESULTS:\n\n", "other"])
def test_parse_psflibrary_missing_result(output):
    assert _parse_psflibrary(output) is None


def test_getpsfcal_prefers_successful_library(monkeypatch):
    completed = SimpleNamespace(returncode=0, stdout=(
        "PSF FLAT RESULTS:\n40620001 40620009\n\n"
    ))
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: completed)
    assert getpsfcal(
        40620001, mjd=60000, telescope="apo25m", psflibrary=True
    ) == 40620009


def test_getpsfcal_falls_back_to_closest_disk_file(monkeypatch, tmp_path):
    completed = SimpleNamespace(returncode=1, stdout="")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: completed)
    files = [
        tmp_path / "apPSF-a-40610001.fits",
        tmp_path / "apPSF-a-40620005.fits",
        tmp_path / "apPSF-a-40630001.fits",
    ]
    assert getpsfcal(
        40620001, mjd=60000, telescope="apo25m", psflibrary=True,
        psf_files=files,
    ) == 40620005


def test_getpsfcal_pre_fps_does_not_use_library(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("psflibrary should not run")
    monkeypatch.setattr("subprocess.run", fail)
    assert getpsfcal(
        30010000, mjd=59000, telescope="apo25m",
        psf_files=["apPSF-a-30010003.fits"],
    ) == 30010003


def test_getpsfcal_builds_closest_quartzflat():
    calls = []
    rows = [
        {"num": 100, "exptype": "DOMEFLAT"},
        {"num": 120, "exptype": "QUARTZFLAT"},
        {"num": 180, "exptype": "QUARTZFLAT"},
    ]
    result = getpsfcal(
        150, mjd=59000, telescope="apo25m", psflibrary=False,
        exposure_rows=rows,
        makecal_func=lambda **kwargs: calls.append(kwargs),
        verify_files_func=lambda value: value == 120,
        unlock=True,
    )
    assert result == 120
    assert calls == [{"psf": 120, "unlock": True}]


def test_getpsfcal_returns_minus_one_without_quartzflat():
    assert getpsfcal(
        150, mjd=59000, telescope="apo25m", psflibrary=False,
        exposure_rows=[{"num": 100, "exptype": "DOMEFLAT"}],
    ) == -1


def test_getpsfcal_returns_minus_one_if_build_not_verified():
    assert getpsfcal(
        150, mjd=59000, telescope="apo25m", psflibrary=False,
        exposure_rows=[{"num": 140, "exptype": "QUARTZFLAT"}],
        makecal_func=lambda **kwargs: None,
        verify_files_func=lambda value: False,
    ) == -1


def test_makehist_constructs_standard_command():
    calls = []
    makehist(60123, apred="daily", runner=lambda *a, **k: calls.append((a, k)))
    assert calls == [(
        (["makehist", "60123", "--apred", "daily"],),
        {"check": True},
    )]


def test_makehist_adds_dark_and_clobber():
    calls = []
    makehist(
        60123, apred="1.4", dark=12345678, clobber=True,
        runner=lambda *a, **k: calls.append((a, k)),
    )
    command = calls[0][0][0]
    assert command[-3:] == ["--darkid", "12345678", "--clobber"]
    assert ["--darkid", "12345678"] == command[4:6]


def test_build_multiwave_lines_groups_by_days_and_count():
    wave_ids = [10000001, 10000002, 10000003, 10000004]
    mjds = [56000, 56003, 56020, 56021]
    psfs = {10000001: 90000001, 10000003: 90000003}
    lines = build_multiwave_lines(
        wave_ids, mjds, psfs, shutdown_mjds=(55900, 56100),
        max_days=10, max_solutions=10,
    )
    assert len(lines) == 2
    assert "10000001,10000002" in lines[0]
    assert "10000003,10000004" in lines[1]


def test_build_multiwave_lines_obeys_max_solutions():
    ids = np.arange(10000001, 10000007)
    dates = np.arange(56000, 56006)
    psfs = {int(value): 9 for value in ids}
    lines = build_multiwave_lines(
        ids, dates, psfs, shutdown_mjds=(55900, 56100),
        max_solutions=2, max_days=30,
    )
    assert len(lines) == 3


def test_build_multiwave_lines_skips_group_without_psf():
    assert build_multiwave_lines(
        [10000001], [56000], {}, shutdown_mjds=(55900, 56100)
    ) == []


def test_run_multi_apwavecal_writes_lines(tmp_path):
    outfile = tmp_path / "mwave.par"
    lines = run_multi_apwavecal(
        [10000001, 10000002], [56000, 56001], {10000001: 90000001},
        shutdown_mjds=(55900, 56100), outfile=outfile,
    )
    assert outfile.read_text().splitlines() == lines

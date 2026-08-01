"""Unit tests for mkwave."""

from types import SimpleNamespace

import numpy as np
import pytest

from apogee_drp.apred.cal import mkwave


def test_build_wave_command_minimal():
    command = mkwave.build_wave_command(
        [35820003, 35820004], 35820003, "daily", "apogee-n", verbose=False
    )
    assert command == [
        "apmultiwavecal", "--name", "35820003", "--vers", "daily",
        "--inst", "apogee-n", "35820003", "35820004",
    ]


def test_build_wave_command_options():
    command = mkwave.build_wave_command(
        [1], 2, "test", "apogee-s", nofit=True, plot=True, clobber=True
    )
    assert "--nofit" in command
    assert command[command.index("--plot"):command.index("--plot") + 2] == [
        "--plot", "--hard"
    ]
    assert "--clobber" in command
    assert "--verbose" in command


@pytest.mark.parametrize(
    "lamp,center,threshold",
    [("UNE", 1452, 40.0), ("THAR", 1566, 1000.0),
     ("UNKNOWN", 1000, 10.0)],
)
def test_arc_flux_diagnostic_lamps(lamp, center, threshold):
    image = np.zeros((32, 2048), dtype=float)
    image[:, center - 3:center + 4] = threshold * 10
    header = {"NREAD": 2}
    if lamp == "UNE":
        header["LAMPUNE"] = True
    elif lamp == "THAR":
        header["LAMPTHAR"] = True
    result = mkwave.arc_flux_diagnostic(image, header)
    assert result["lamp"] == lamp
    assert result["threshold"] == threshold
    assert result["okay"]


def test_arc_flux_diagnostic_rebins_spatial_axis_by_summing():
    image = np.zeros((16, 2048), dtype=float)
    image[:, 1449:1456] = 20
    result = mkwave.arc_flux_diagnostic(
        image, {"NREAD": 2, "LAMPUNE": True}, bin_size=8
    )
    assert result["average_peakflux"] == 160
    assert result["flux_per_read"] == 80


def test_arc_flux_diagnostic_rejects_weak_arc():
    image = np.zeros((16, 2048), dtype=float)
    result = mkwave.arc_flux_diagnostic(
        image, {"NREAD": 10, "LAMPUNE": True}
    )
    assert not result["okay"]


def test_arc_flux_diagnostic_requires_nread():
    with pytest.raises(ValueError, match="NREAD"):
        mkwave.arc_flux_diagnostic(np.zeros((16, 2048)), {})


def test_run_wavecal_reports_subprocess_output():
    completed = SimpleNamespace(returncode=2, stdout="output", stderr="error")
    with pytest.raises(RuntimeError, match="output"):
        mkwave._run_wavecal(["false"], runner=lambda *args, **kwargs: completed)


def test_call_processor_passes_calibrations():
    calls = []
    mkwave._call_processor(
        lambda *args, **kwargs: calls.append((args, kwargs)), [1, 2],
        darkid=3, flatid=4, clobber=True
    )
    assert calls[0][0] == ([1, 2],)
    assert calls[0][1]["darkid"] == 3
    assert calls[0][1]["flatid"] == 4
    assert calls[0][1]["clobber"] is True

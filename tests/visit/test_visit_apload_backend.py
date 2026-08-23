from unittest.mock import MagicMock

import numpy as np

from apogee_drp.apred.visit.apload_backend import ApLoadVisitBackend
from apogee_drp.apred.visit.models import VisitFrame


def frame_mapping():
    return {chip: {"header": {}, "flux": np.ones((2, 4)), "err": np.ones((2, 4)),
                   "mask": np.zeros((2, 4), dtype=np.int16)} for chip in "abc"}


def backend():
    load = MagicMock()
    load.instrument = "apogee-n"
    load.observatory = "apo"
    load.prefix = "ap"
    return ApLoadVisitBackend("daily", "apo25m", load=load), load


def test_load_1d_uses_apload_spectrum():
    visit, load = backend()
    load.spectrum.return_value = frame_mapping()
    files = [f"/tmp/ap1D-{chip}-12345678.fits" for chip in "abc"]

    frame = visit.load_frame(files, kind="1D")

    assert isinstance(frame, VisitFrame)
    load.spectrum.assert_called_once_with(12345678)
    assert frame.chipb.filename == files[1]


def test_load_cframe_uses_apload_cframe_and_plan_context():
    visit, load = backend()
    visit._current_plan = {"plateid": 1234, "mjd": 60000, "field": "field"}
    loaded = frame_mapping()
    loaded.update(shift={"shiftfit": [0, 0]}, tellstar=None, plugmap={})
    load.cframe.return_value = loaded
    files = [f"/tmp/apCframe-{chip}-12345678.fits" for chip in "abc"]

    frame = visit.load_frame(files, kind="Cframe")

    assert isinstance(frame, VisitFrame)
    load.cframe.assert_called_once_with(12345678, 1234, 60000, field="field")


def test_files_uses_apload_multiple_chip_interface():
    visit, load = backend()
    load.filename.return_value = {chip: f"/tmp/apWave-{chip}-12.fits" for chip in "abc"}

    files = visit._files("Wave", num=12)

    load.filename.assert_called_once_with("Wave", chip=["a", "b", "c"], num=12)
    assert files == [f"/tmp/apWave-{chip}-12.fits" for chip in "abc"]

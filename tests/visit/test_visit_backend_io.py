import numpy as np
from astropy.io import fits
from astropy.table import Table

from apogee_drp.apred.visit.io import BADERR, read_cframes, write_cframes
from apogee_drp.apred.visit.models import VisitFrame


def _frame():
    frame = {
        "tellstar": Table({"fiber": [1], "scale": [1.1]}),
        "shift": Table({"shiftfit": [[0.0, 0.0]], "shifterr": [0.0]}),
    }
    for chip in "abc":
        shape = (2, 5)
        frame["chip" + chip] = {
            "header": fits.Header({"DITHPIX": 0.5}),
            "flux": np.arange(10).reshape(shape),
            "err": np.ones(shape),
            "mask": np.zeros(shape, dtype=np.int32),
            "wavelength": np.full(shape, 16000.0),
            "sky": np.zeros(shape),
            "skyerr": np.ones(shape),
            "telluric": np.ones(shape),
            "telluricerr": np.zeros(shape),
            "wcoef": np.arange(14.0).reshape(2, 7),
            "lsfcoef": np.arange(12.0).reshape(2, 6),
        }
    frame["chipa"]["err"][0, 0] = np.nan
    return frame


def test_cframe_hdu_contract_and_roundtrip(tmp_path):
    files = [tmp_path / f"apCframe-{chip}-1.fits" for chip in "abc"]
    plugmap = {
        "fiberdata": Table({"fiberid": [1, 2], "objtype": ["STAR", "SKY"]}),
        "plateid": 1234,
        "field": "test",
    }
    write_cframes(_frame(), plugmap, files)
    with fits.open(files[0]) as hdul:
        assert len(hdul) == 15
        assert [h.name for h in hdul[1:11]] == [
            "FLUX",
            "ERROR",
            "MASK",
            "WAVELENGTH",
            "SKY FLUX",
            "SKY ERROR",
            "TELLURIC",
            "TELLURIC ERROR",
            "WAVE COEFFICIENTS",
            "LSF COEFFICIENTS",
        ]
        assert hdul[1].data.shape == (2, 5)
        assert hdul[1].data.dtype.kind == "f"
        assert hdul[2].data[0, 0] == BADERR
        assert hdul[3].data.dtype.itemsize == 2
        assert hdul[4].data.dtype.itemsize == 8
    restored = read_cframes(files)
    assert isinstance(restored, VisitFrame)
    np.testing.assert_array_equal(
        restored["chipb"]["flux"], _frame()["chipb"]["flux"]
    )

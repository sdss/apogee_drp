import numpy as np
from astropy.io import fits

from apogee_drp.apred.visit.io import BADERR
from apogee_drp.apred.visit.output import PIXEL_BITS, STAR_BITS
from apogee_drp.apred.visit.plate import write_plate_products
from apogee_drp.apred.visit.products import build_visit_hdul, write_visit_products
from apogee_drp.apred.visit.models import ChipFrame, VisitFrame


def _frame(nfiber=4, npix=40):
    chips = []
    for chip_index, letter in enumerate("abc"):
        flux = np.full((nfiber, npix), (10 + chip_index) * 1e-17, np.float32)
        error = np.full((nfiber, npix), 1e-17, np.float32)
        chips.append(ChipFrame(flux, error, np.zeros((nfiber, npix), np.int16),
            header={
                "DATE-OBS": "2026-01-01T01:02:03",
                "EXPTIME": 1000.0,
                "JD-MID": 2460000.5,
                "UT-MID": "01:10:00",
                "NCOMBINE": 2,
                "FRAME1": 1001,
                "FRAME2": 1002,
                "NPAIRS": 1,
                "HISTORY": ["AP3D: remove me", "keep me"],
            }, wavelength=np.broadcast_to(
                np.linspace(15100 + chip_index * 600, 15600 + chip_index * 600, npix),
                (nfiber, npix),
            ).copy(), sky=np.full((nfiber, npix), 2e-17, np.float32),
            skyerr=np.full((nfiber, npix), 0.5e-17, np.float32),
            telluric=np.full((nfiber, npix), 0.95, np.float32),
            telluricerr=np.full((nfiber, npix), 0.01, np.float32),
            wcoef=np.zeros((nfiber, 14), np.float64),
            lsfcoef=np.zeros((nfiber, 8), np.float64)))
    frame = VisitFrame(*chips)
    frame.metadata["fluxcorr"] = np.full(nfiber, 2.5e-17, dtype=np.float32)
    return frame


def _plugmap():
    fiberid = np.array([300, 299, 298, 297])
    mag = np.array(
        [[12, 10, 9], [6, 4, 3], [11, 9, 8], [13, 11, 10]], dtype=float
    )
    return {
        "plateid": 1234,
        "mjd": 60000,
        "locationid": 42,
        "field": "field",
        "fiberdata": {
            "fiberid": fiberid,
            "spectrographid": np.full(4, 2),
            "holetype": np.full(4, "OBJECT"),
            "objtype": np.array(["STAR", "SKY", "HOT_STD", "STAR"]),
            "tmass_style": np.array(["2M000", "-", "2M002", "2MNone"]),
            "ra": np.arange(4) + 10.0,
            "dec": np.arange(4) - 20.0,
            "mag": mag,
            "target1": np.arange(4),
            "target2": np.zeros(4, int),
            "target3": np.zeros(4, int),
            "target4": np.zeros(4, int),
            "sdssv_apogee_target0": np.arange(4) + 100,
            "catalogid": np.arange(4) + 1000,
            "sdss_id": np.arange(4) + 2000,
            "gaia_release": np.full(4, "dr3"),
            "gaia_sourceid": np.arange(4) + 3000,
            "gaia_plx": np.ones(4),
            "gaia_pmra": np.ones(4) * 2,
            "gaia_pmdec": np.ones(4) * 3,
            "gaia_gmag": np.ones(4) * 12,
            "gaia_bpmag": np.ones(4) * 13,
            "gaia_rpmag": np.ones(4) * 11,
            "assigned": np.ones(4, bool),
            "on_target": np.ones(4, bool),
            "valid": np.ones(4, bool),
            "firstcarton": np.full(4, "carton"),
            "cadence": np.full(4, "cadence"),
            "program": np.full(4, "program"),
            "category": np.full(4, "science"),
        },
    }


def test_plate_hdu_contract_and_sentinels(tmp_path):
    frame = _frame()
    frame["chipa"]["flux"][0, 2] = np.nan
    frame["chipa"]["err"][0, 3] = BADERR
    files = [tmp_path / f"apPlate-{letter}.fits" for letter in "abc"]
    written = write_plate_products(
        frame,
        _plugmap(),
        {"shift": np.array([0.5])},
        {"pair": np.array([1])},
        files,
    )
    assert written == [str(path) for path in files]
    with fits.open(files[0]) as hdul:
        assert len(hdul) == 16
        assert [hdu.name for hdu in hdul[1:]] == [
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
            "PLUGMAP",
            "PLUGMAP HEADER",
            "SHIFT",
            "PAIR",
            "FLUX CONVERSION",
        ]
        assert hdul["FLUX"].data.dtype.kind == "f"
        assert hdul["FLUX"].data[0, 2] == 0
        assert hdul["MASK"].data[0, 2] & int(PIXEL_BITS.getval("BADPIX"))
        assert hdul["ERROR"].data[0, 3] == BADERR
        assert "AP3D: remove me" not in str(hdul[0].header["HISTORY"])


def test_visit_hdu_contract_metadata_and_quality_flags():
    frame = _frame()
    persistence = int(PIXEL_BITS.getval("PERSIST_HIGH"))
    frame["chipa"]["mask"][0, :30] = persistence
    frame["chipc"]["flux"][0] = 40e-17
    hdul = build_visit_hdul(
        frame,
        _plugmap(),
        0,
        survey="mwm",
        relflux=np.array([1.0, 0.8, 0.6, 0.4]),
    )
    assert len(hdul) == 11
    assert [hdu.name for hdu in hdul[1:]] == [
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
    header = hdul[0].header
    assert header["FIBERID"] == 300
    assert header["OBJID"] == "2M000"
    assert header["NCOMBINE"] == 2
    assert header["HMINUS"] == -6
    assert header["SURVEY"] == "mwm"
    assert header["FLUXFLAM"] == frame["fluxcorr"][0]
    flag = header["VISITFLG"]
    assert flag & int(STAR_BITS.getval("VERY_BRIGHT_NEIGHBOR"))
    assert flag & int(STAR_BITS.getval("PERSIST_HIGH"))
    assert flag & int(STAR_BITS.getval("PERSIST_JUMP_POS"))
    assert np.all(
        hdul["MASK"].data[0, :30] & int(PIXEL_BITS.getval("BADPIX"))
    )
    np.testing.assert_allclose(hdul["FLUX"].data[0], 10)


def test_write_visit_products_selects_only_valid_objects(tmp_path):
    frame = _frame()
    result = write_visit_products(
        frame,
        _plugmap(),
        {"shift": np.array([0.5])},
        {"pair": np.array([1])},
        single=True,
        visit_directory=tmp_path,
    )
    assert result.plate_files == []
    assert len(result.visit_files) == 2
    assert {fits.getheader(path)["FIBERID"] for path in result.visit_files} == {
        300,
        298,
    }
    assert all(fits.getheader(path)["TELESCOP"] == "apo1m" for path in result.visit_files)


def test_missing_plugmap_fiber_is_reported_as_skipped(tmp_path):
    plugmap = _plugmap()
    for key, value in plugmap["fiberdata"].items():
        plugmap["fiberdata"][key] = np.asarray(value)[:3]
    result = write_visit_products(
        _frame(),
        plugmap,
        {},
        {},
        single=True,
        visit_directory=tmp_path,
    )
    assert result.skipped_fibers == [297]

"""Extensive tests for :mod:`apogee_drp.apred.ap2d`.

The tests deliberately separate small numerical and validation contracts from
the still-partially-translated extraction branches.  Pipeline dependencies are
mocked so the suite can run without an SDSS data tree.
"""

import inspect
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred import ap2d


class FakeLoad:
    """Minimal ApLoad replacement used by the AP2DPROC tests."""

    def __init__(self, root, apred="test", telescope="apo25m"):
        self.root = Path(root)
        self.apred = apred
        self.telescope = telescope
        self.prefix = "ap"
        self.exists_values = {}
        self.frame = None

    def filename(self, product, num=None, chips=True, **kwargs):
        del chips, kwargs
        number = int(num) if num is not None else 0
        return str(self.root / f"ap{product}-{number:08d}.fits")

    def exists(self, product, num=None, **kwargs):
        del num, kwargs
        return self.exists_values.get(product, True)

    def ap2D(self, number):
        del number
        return self.frame

    def cmjd(self, number):
        del number
        return "60000"


def make_frame(shape=(8, 8), nread=4):
    """Create an in-memory three-chip ap2D-like HDU mapping."""

    frame = {}
    for chip in "abc":
        header = fits.Header({"NREAD": nread, "IMAGETYP": "object"})
        flux = np.ones(shape, dtype=np.float32)
        error = np.ones(shape, dtype=np.float32) * 2
        mask = np.zeros(shape, dtype=np.int16)
        frame[chip] = fits.HDUList([
            fits.PrimaryHDU(header=header), fits.ImageHDU(flux),
            fits.ImageHDU(error), fits.ImageHDU(mask),
        ])
    return frame


def prepare_inputs(tmp_path, monkeypatch, frameid=123, psfid=456):
    """Prepare the files and mocks needed to reach the extraction loop."""

    input_dir = tmp_path / "input"
    psf_dir = tmp_path / "psf"
    output_dir = tmp_path / "output"
    local_dir = tmp_path / "local"
    for directory in (input_dir, psf_dir, output_dir, local_dir):
        directory.mkdir()

    load = FakeLoad(psf_dir)
    load.frame = make_frame()
    monkeypatch.setenv("APOGEE_LOCALDIR", str(local_dir))

    for chip in "abc":
        fits.PrimaryHDU(header=fits.Header({"IMAGETYP": "object"})).writeto(
            input_dir / f"ap2D-{chip}-{frameid:08d}.fits"
        )
        Path(load.filename("PSF", num=psfid).replace("PSF-", f"PSF-{chip}-")).touch()
        Path(load.filename("EPSF", num=psfid).replace("EPSF-", f"EPSF-{chip}-")).touch()

    monkeypatch.setattr(ap2d.fits, "getdata", lambda *args, **kwargs: np.zeros(300, dtype=[("fiber", int)]))
    calls = []
    monkeypatch.setattr(ap2d.lock, "lock", lambda *args, **kwargs: calls.append((args, kwargs)))
    return load, input_dir / str(frameid), psf_dir / str(psfid), output_dir, calls


def prepare_empirical_extraction(tmp_path, monkeypatch):
    """Prepare a small successful extract_type=4 execution."""

    load, inpfile, psffile, outdir, calls = prepare_inputs(tmp_path, monkeypatch)
    epsf = [{"fiber": 0}, {"fiber": 1}]
    loaded = []

    def fake_loadepsf(filename):
        loaded.append(filename)
        return epsf

    def fake_extract(chstr, input_epsf, outstr, scat=True):
        del outstr
        assert input_epsf is epsf
        assert scat is True
        npix = chstr["flux"].shape[1]
        output = {
            "flux": np.arange(2 * npix, dtype=float).reshape(2, npix),
            "err": np.ones((2, npix), dtype=float) * 2,
            "mask": np.zeros((2, npix), dtype=np.int16),
        }
        model = np.ones_like(chstr["flux"], dtype=float)
        return output, np.zeros(npix), model

    monkeypatch.setattr(ap2d.psf, "loadepsf", fake_loadepsf)
    monkeypatch.setattr(ap2d.psf, "extract", fake_extract)
    monkeypatch.setattr(ap2d.plan, "getgitvers", lambda: "test-version")
    ap2d.savedepsffiles[:] = [None, None, None]
    ap2d.epsfchip[:] = [None, None, None]
    return load, inpfile, psffile, outdir, calls, loaded


class TestErrout:
    @pytest.mark.parametrize("value", [ap2d.BADERR, 0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_invalid_values_become_baderr(self, value):
        data = np.array([value], dtype=float)
        result = ap2d.errout(data)
        assert result[0] == ap2d.BADERR

    @pytest.mark.parametrize("value", [1e-12, 1.0, 25.5, ap2d.BADERR - 1])
    def test_positive_finite_values_are_preserved(self, value):
        data = np.array([value], dtype=float)
        np.testing.assert_array_equal(ap2d.errout(data), [value])

    def test_mixed_array(self):
        data = np.array([1.0, 0.0, -2.0, np.nan, 5.0])
        expected = np.array([1.0, ap2d.BADERR, ap2d.BADERR, ap2d.BADERR, 5.0])
        np.testing.assert_array_equal(ap2d.errout(data), expected)

    def test_operates_in_place(self):
        data = np.array([0.0, 2.0])
        result = ap2d.errout(data)
        assert result is data
        np.testing.assert_array_equal(data, [ap2d.BADERR, 2.0])

    def test_multidimensional_array(self):
        data = np.array([[1.0, 0.0], [np.nan, 3.0]])
        result = ap2d.errout(data)
        assert result.shape == (2, 2)
        assert np.count_nonzero(result == ap2d.BADERR) == 2


class TestAP2DProcInterface:
    def test_defaults(self):
        signature = inspect.signature(ap2d.ap2dproc)
        assert signature.parameters["extract_type"].default == 1
        assert signature.parameters["clobber"].default is False
        assert signature.parameters["fixbadpix"].default is False
        assert signature.parameters["nowrite"].default is False
        assert signature.parameters["chips"].default == [0, 1, 2]

    def test_requires_load_or_apred_and_telescope(self, tmp_path):
        with pytest.raises(ValueError, match="load or apred"):
            ap2d.ap2dproc(str(tmp_path / "123"), str(tmp_path / "456"), silent=True)

    def test_constructs_apload(self, tmp_path, monkeypatch):
        created = []

        class ConstructorLoad(FakeLoad):
            def __init__(self, apred, telescope):
                created.append((apred, telescope))
                super().__init__(tmp_path, apred, telescope)

        monkeypatch.setattr(ap2d.apload, "ApLoad", ConstructorLoad)
        monkeypatch.setenv("APOGEE_LOCALDIR", str(tmp_path / "local"))
        result = ap2d.ap2dproc(str(tmp_path / "missing" / "123"), str(tmp_path / "456"), apred="daily", telescope="apo25m", silent=True)
        assert created == [("daily", "apo25m")]
        assert result == ([], [])

    @pytest.mark.parametrize("chips", [[-1], [3], [0, 1, 2, 0]])
    def test_invalid_chips(self, tmp_path, monkeypatch, chips):
        monkeypatch.setenv("APOGEE_LOCALDIR", str(tmp_path / "local"))
        with pytest.raises(ValueError, match="chips must"):
            ap2d.ap2dproc(str(tmp_path / "123"), str(tmp_path / "456"), load=FakeLoad(tmp_path), chips=chips, silent=True)

    @pytest.mark.parametrize("fibers", [[-1], [300], list(range(301))])
    def test_invalid_fibers(self, tmp_path, monkeypatch, fibers):
        monkeypatch.setenv("APOGEE_LOCALDIR", str(tmp_path / "local"))
        with pytest.raises(ValueError, match="fibers must"):
            ap2d.ap2dproc(str(tmp_path / "123"), str(tmp_path / "456"), load=FakeLoad(tmp_path), fibers=fibers, silent=True)

    def test_creates_output_and_local_directories(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "missing_input"
        output_dir = tmp_path / "new_output"
        local_dir = tmp_path / "local"
        monkeypatch.setenv("APOGEE_LOCALDIR", str(local_dir))
        result = ap2d.ap2dproc(str(input_dir / "123"), str(tmp_path / "456"), load=FakeLoad(tmp_path), outdir=str(output_dir), silent=True)
        assert result == ([], [])
        assert output_dir.is_dir()
        assert (local_dir / "test").is_dir()

    def test_missing_input_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APOGEE_LOCALDIR", str(tmp_path / "local"))
        result = ap2d.ap2dproc(str(tmp_path / "missing" / "123"), str(tmp_path / "456"), load=FakeLoad(tmp_path), silent=True)
        assert result == ([], [])

    def test_missing_three_chip_inputs(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("APOGEE_LOCALDIR", str(tmp_path / "local"))
        result = ap2d.ap2dproc(str(input_dir / "123"), str(tmp_path / "456"), load=FakeLoad(tmp_path), silent=True)
        assert result == ([], [])

    def test_missing_psf_directory(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for chip in "abc":
            (input_dir / f"ap2D-{chip}-00000123.fits").touch()
        monkeypatch.setenv("APOGEE_LOCALDIR", str(tmp_path / "local"))
        result = ap2d.ap2dproc(str(input_dir / "123"), str(tmp_path / "missing_psf" / "456"), load=FakeLoad(tmp_path), silent=True)
        assert result == ([], [])

    def test_missing_psf_calibration(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "input"
        psf_dir = tmp_path / "psf"
        input_dir.mkdir()
        psf_dir.mkdir()
        for chip in "abc":
            (input_dir / f"ap2D-{chip}-00000123.fits").touch()
        load = FakeLoad(psf_dir)
        load.exists_values["PSF"] = False
        monkeypatch.setenv("APOGEE_LOCALDIR", str(tmp_path / "local"))
        result = ap2d.ap2dproc(str(input_dir / "123"), str(psf_dir / "456"), load=load, silent=True)
        assert result == ([], [])

    def test_existing_outputs_skip_and_clear_lock(self, tmp_path, monkeypatch):
        load, inpfile, psffile, outdir, calls = prepare_inputs(tmp_path, monkeypatch)
        for chip in "abc":
            (outdir / f"ap1D-{chip}-00000123.fits").touch()
        result = ap2d.ap2dproc(str(inpfile), str(psffile), load=load, outdir=str(outdir), silent=True)
        assert result == ([], [])
        assert any(kwargs.get("clear") is True for args, kwargs in calls)

    @pytest.mark.parametrize("extract_type", [1, 2, 3])
    def test_untranslated_extraction_modes_raise(self, tmp_path, monkeypatch, extract_type):
        load, inpfile, psffile, outdir, calls = prepare_inputs(tmp_path, monkeypatch)
        del calls
        with pytest.raises(ValueError, match="Not Translated yet"):
            ap2d.ap2dproc(str(inpfile), str(psffile), extract_type=extract_type, load=load, outdir=str(outdir), silent=True)

    def test_model_extraction_requires_model_file(self, tmp_path, monkeypatch):
        load, inpfile, psffile, outdir, calls = prepare_inputs(tmp_path, monkeypatch)
        del calls
        with pytest.raises(ValueError, match="Need Model PSF"):
            ap2d.ap2dproc(str(inpfile), str(psffile), extract_type=5, load=load, outdir=str(outdir), silent=True)

    def test_fiber_larger_than_trace_table_is_rejected(self, tmp_path, monkeypatch):
        load, inpfile, psffile, outdir, calls = prepare_inputs(tmp_path, monkeypatch)
        del calls
        monkeypatch.setattr(ap2d.fits, "getdata", lambda *args, **kwargs: np.zeros(2, dtype=[("fiber", int)]))
        result = ap2d.ap2dproc(str(inpfile), str(psffile), extract_type=1, load=load, outdir=str(outdir), fibers=[2], silent=True)
        assert result is None


class TestEmpiricalExtraction:
    def test_returns_extracted_spectrum_and_model(self, tmp_path, monkeypatch):
        load, inpfile, psffile, outdir, calls, loaded = prepare_empirical_extraction(tmp_path, monkeypatch)
        del calls
        output, models = ap2d.ap2dproc(str(inpfile), str(psffile), extract_type=4, load=load, outdir=str(outdir), chips=[0], nowrite=True, silent=True)
        assert list(output) == [0]
        assert list(models) == [0]
        assert output[0]["flux"].shape == (2, 8)
        assert output[0]["header"]["EXTRTYPE"] == 4
        assert models[0].shape == (8, 8)
        assert len(loaded) == 1

    def test_nowrite_does_not_create_1d_product(self, tmp_path, monkeypatch):
        load, inpfile, psffile, outdir, calls, loaded = prepare_empirical_extraction(tmp_path, monkeypatch)
        del calls, loaded
        ap2d.ap2dproc(str(inpfile), str(psffile), extract_type=4, load=load, outdir=str(outdir), chips=[0], nowrite=True, silent=True)
        assert not (outdir / "ap1D-a-00000123.fits").exists()
        assert (outdir / "ap2Dmodel-a-00000123.fits").exists()

    def test_written_product_has_expected_hdus(self, tmp_path, monkeypatch):
        load, inpfile, psffile, outdir, calls, loaded = prepare_empirical_extraction(tmp_path, monkeypatch)
        del calls, loaded
        ap2d.ap2dproc(str(inpfile), str(psffile), extract_type=4, load=load, outdir=str(outdir), chips=[0], silent=True)
        filename = outdir / "ap1D-a-00000123.fits"
        with fits.open(filename) as hdul:
            assert [hdu.name for hdu in hdul] == ["PRIMARY", "FLUX", "ERROR", "MASK"]
            assert hdul[1].data.shape == (8, 2)
            assert hdul[2].data.shape == (8, 2)
            assert hdul[3].data.shape == (8, 2)
            assert hdul[0].header["EXTRTYPE"] == 4

    def test_outlong_writes_integer_flux_and_error(self, tmp_path, monkeypatch):
        load, inpfile, psffile, outdir, calls, loaded = prepare_empirical_extraction(tmp_path, monkeypatch)
        del calls, loaded
        ap2d.ap2dproc(str(inpfile), str(psffile), extract_type=4, load=load, outdir=str(outdir), chips=[0], outlong=True, silent=True)
        with fits.open(outdir / "ap1D-a-00000123.fits") as hdul:
            assert hdul[1].data.dtype.kind in "iu"
            assert hdul[2].data.dtype.kind in "iu"

    def test_epsf_is_cached_between_calls(self, tmp_path, monkeypatch):
        load, inpfile, psffile, outdir, calls, loaded = prepare_empirical_extraction(tmp_path, monkeypatch)
        del calls
        for _ in range(2):
            ap2d.ap2dproc(str(inpfile), str(psffile), extract_type=4, load=load, outdir=str(outdir), chips=[0], nowrite=True, silent=True)
        assert len(loaded) == 1

    def test_lock_is_cleared_after_success(self, tmp_path, monkeypatch):
        load, inpfile, psffile, outdir, calls, loaded = prepare_empirical_extraction(tmp_path, monkeypatch)
        del loaded
        ap2d.ap2dproc(str(inpfile), str(psffile), extract_type=4, load=load, outdir=str(outdir), chips=[0], nowrite=True, silent=True)
        assert any(kwargs.get("lock") is True for args, kwargs in calls)
        assert any(kwargs.get("clear") is True for args, kwargs in calls)


def make_plan(platetype="dark", psfid=1):
    apexp = np.zeros(1, dtype=[("name", int), ("flavor", "U10"), ("psfid", int)])
    apexp["name"] = 123
    apexp["flavor"] = "object"
    apexp["psfid"] = psfid
    return {
        "apred_vers": "test", "telescope": "apo25m", "plateid": "1",
        "mjd": 60000, "platetype": platetype, "psfid": psfid,
        "APEXP": apexp,
    }


class TestAP2DWrapper:
    def test_interface_defaults(self):
        signature = inspect.signature(ap2d.ap2d)
        assert signature.parameters["verbose"].default is False
        assert signature.parameters["clobber"].default is False
        assert signature.parameters["exttype"].default == 4
        assert signature.parameters["calclobber"].default is False
        assert signature.parameters["psflibrary"].default is False

    def test_single_plan_filename_is_accepted(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(ap2d.plan, "load", lambda *args, **kwargs: None)
        ap2d.ap2d("plan.yaml")
        output = capsys.readouterr().out
        assert "1 plan files" in output
        assert "processing plan file plan.yaml" in output

    def test_plan_list_is_processed_in_order(self, monkeypatch, tmp_path):
        seen = []

        def fake_load(filename, np=True):
            del np
            seen.append(filename)
            return None

        monkeypatch.setattr(ap2d.plan, "load", fake_load)
        ap2d.ap2d(["one.yaml", "two.yaml"])
        assert seen == ["one.yaml", "two.yaml"]

    @pytest.mark.parametrize("platetype", ["dark", "intflat"])
    def test_dark_and_internal_flat_plans_are_skipped(self, monkeypatch, tmp_path, platetype):
        plan_data = make_plan(platetype=platetype)
        created = []

        class WrapperLoad(FakeLoad):
            def __init__(self, apred, telescope):
                created.append((apred, telescope))
                super().__init__(tmp_path, apred, telescope)

            def filename(self, product, **kwargs):
                del kwargs
                return str(tmp_path / f"ap{product}.fits")

        monkeypatch.setattr(ap2d.plan, "load", lambda *args, **kwargs: plan_data)
        monkeypatch.setattr(ap2d.apload, "ApLoad", WrapperLoad)
        monkeypatch.setattr(ap2d, "ap2dproc", lambda *args, **kwargs: pytest.fail("ap2dproc should not be called"))
        ap2d.ap2d("plan.yaml")
        assert created == [("test", "apo25m")]

    def test_none_plan_is_skipped_without_constructing_apload(self, monkeypatch):
        monkeypatch.setattr(ap2d.plan, "load", lambda *args, **kwargs: None)
        monkeypatch.setattr(ap2d.apload, "ApLoad", lambda *args, **kwargs: pytest.fail("ApLoad should not be constructed"))
        ap2d.ap2d("plan.yaml")

    def test_empty_plan_list(self, capsys):
        ap2d.ap2d([])
        assert "0 plan files" in capsys.readouterr().out


class TestConstants:
    def test_baderr_is_large_and_finite(self):
        assert np.isfinite(ap2d.BADERR)
        assert ap2d.BADERR == 1e10

    def test_pixel_bad_value_contains_bad_bits(self):
        assert isinstance(ap2d.PIXBADVAL, (int, np.integer))
        assert ap2d.PIXBADVAL > 0

"""Extensive numerical, helper, FITS-I/O, and workflow tests for bpm."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apogee_drp.apred.cal import bpm
from apogee_drp.apred.cal import utils as cal_utils


BPMID = 12345678
DARKID = 12340001
FLATID = 12340002
BITS = {"BADPIX": 1, "BADDARK": 2, "BADFLAT": 4}


class FakePixelBitMask:
    def getval(self, name):
        return BITS[name]


@pytest.fixture
def environment(monkeypatch, tmp_path):
    class FakeLoad:
        prefix = "ap"

        def filename(self, kind, num=None, chip=None, **kwargs):
            directory = tmp_path / kind.lower()
            directory.mkdir(parents=True, exist_ok=True)
            if chip is not None:
                return str(directory / f"ap{kind}-{chip}-{int(num):08d}.fits")
            return str(directory / f"ap{kind}-{int(num):08d}.fits")

        def product_files(self, product, name):
            assert product == "bpm"
            return [self.filename("BPM", num=name, chip=chip)
                    for chip in bpm.CHIPS]

        def product_exists(self, product, name):
            return all(
                Path(filename).is_file() and Path(filename).stat().st_size > 0
                for filename in self.product_files(product, name)
            )

        def product_delete(self, product, name, **kwargs):
            for filename in self.product_files(product, name):
                path = Path(filename)
                if path.exists() or path.is_symlink():
                    path.unlink()

    load = FakeLoad()
    apload_calls, lock_calls, read_calls = [], [], []
    darkmasks = {chip: np.zeros((8, 10), dtype=np.uint8) for chip in "abc"}
    flatmasks = {chip: np.zeros((8, 10), dtype=np.uint8) for chip in "abc"}

    def getdata(filename, ext=None):
        path = Path(filename)
        read_calls.append((path, ext))
        chip = path.name.split("-")[1]
        return (darkmasks if "Dark" in path.name else flatmasks)[chip].copy()

    monkeypatch.setattr(
        bpm.apload, "ApLoad",
        lambda **kwargs: apload_calls.append(kwargs) or load,
    )
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda filename, **kwargs:
        lock_calls.append((Path(filename), kwargs)),
    )
    monkeypatch.setattr(bpm, "PixelBitMask", FakePixelBitMask)
    monkeypatch.setattr(bpm.fits, "getdata", getdata)
    monkeypatch.setattr(bpm.utils, "software_version", lambda: "git-test")
    monkeypatch.setattr(bpm.utils, "reduction_version", lambda: "drp-test")
    return SimpleNamespace(root=tmp_path, load=load, apload_calls=apload_calls,
                           lock_calls=lock_calls, read_calls=read_calls,
                           darkmasks=darkmasks, flatmasks=flatmasks)


def output_files(env, bpmid=BPMID):
    return [Path(filename)
            for filename in env.load.product_files("bpm", bpmid)]


def lockfile(env, bpmid=BPMID):
    return output_files(env, bpmid)[0]


def assert_lock_cleared(env):
    assert any(filename == lockfile(env) and kwargs.get("clear") is True
               for filename, kwargs in env.lock_calls)


class TestMakeBpm:
    def test_empty_masks_produce_zero_mask(self):
        result = bpm.combine_bpm_masks(np.zeros((5, 7), dtype=np.uint8), pixmask=FakePixelBitMask())
        np.testing.assert_array_equal(result, 0)

    def test_dark_pixels_receive_dark_and_general_bits(self):
        dark = np.zeros((5, 7), dtype=int)
        dark[2, 3] = 8
        result = bpm.combine_bpm_masks(dark, pixmask=FakePixelBitMask())
        assert result[2, 3] == BITS["BADDARK"] | BITS["BADPIX"]
        assert np.count_nonzero(result) == 1

    def test_flat_pixels_receive_flat_and_general_bits(self):
        dark = np.zeros((5, 7), dtype=int)
        flat = np.zeros_like(dark)
        flat[1, 4] = 16
        result = bpm.combine_bpm_masks(dark, flat, pixmask=FakePixelBitMask())
        assert result[1, 4] == BITS["BADFLAT"] | BITS["BADPIX"]

    def test_overlapping_dark_and_flat_pixels_combine_all_bits(self):
        dark = np.zeros((5, 7), dtype=int)
        flat = np.zeros_like(dark)
        dark[3, 2] = flat[3, 2] = 1
        result = bpm.combine_bpm_masks(dark, flat, pixmask=FakePixelBitMask())
        assert result[3, 2] == BITS["BADDARK"] | BITS["BADFLAT"] | BITS["BADPIX"]

    @pytest.mark.parametrize("value", [-3, 0.5, np.nan, np.inf])
    def test_any_nonzero_or_nonfinite_dark_value_is_bad(self, value):
        dark = np.zeros((3, 4), dtype=float)
        dark[1, 2] = value
        result = bpm.combine_bpm_masks(dark, pixmask=FakePixelBitMask())
        assert result[1, 2] & BITS["BADDARK"]

    def test_badrows_flag_detector_rows_in_python_orientation(self):
        result = bpm.combine_bpm_masks(
            np.zeros((5, 7)), badrows=[1, 4], pixmask=FakePixelBitMask())
        np.testing.assert_array_equal(result[1, :], BITS["BADPIX"])
        np.testing.assert_array_equal(result[4, :], BITS["BADPIX"])
        np.testing.assert_array_equal(result[[0, 2, 3], :], 0)

    def test_badrow_preserves_existing_specific_bits(self):
        dark = np.zeros((5, 7))
        dark[2, 1] = 1
        result = bpm.combine_bpm_masks(dark, badrows=[2], pixmask=FakePixelBitMask())
        assert result[2, 1] == BITS["BADDARK"] | BITS["BADPIX"]

    def test_empty_badrow_list_changes_nothing(self):
        dark = np.zeros((5, 7))
        dark[2, 1] = 1
        expected = bpm.combine_bpm_masks(dark, pixmask=FakePixelBitMask())
        actual = bpm.combine_bpm_masks(dark, badrows=[], pixmask=FakePixelBitMask())
        np.testing.assert_array_equal(actual, expected)

    def test_duplicate_badrows_are_harmless(self):
        result = bpm.combine_bpm_masks(np.zeros((5, 7)), badrows=[2, 2], pixmask=FakePixelBitMask())
        np.testing.assert_array_equal(result[2, :], BITS["BADPIX"])

    @pytest.mark.parametrize("badrows", [[-1], [5], [0, 8]])
    def test_rejects_badrows_outside_detector(self, badrows):
        with pytest.raises(ValueError, match="outside the detector"):
            bpm.combine_bpm_masks(np.zeros((5, 7)), badrows=badrows, pixmask=FakePixelBitMask())

    @pytest.mark.parametrize("shape", [(10,), (2, 3, 4)])
    def test_darkmask_must_be_two_dimensional(self, shape):
        with pytest.raises(ValueError, match="two-dimensional"):
            bpm.combine_bpm_masks(np.zeros(shape), pixmask=FakePixelBitMask())

    def test_flatmask_shape_must_match(self):
        with pytest.raises(ValueError, match="identical shapes"):
            bpm.combine_bpm_masks(np.zeros((3, 4)), np.zeros((4, 3)), pixmask=FakePixelBitMask())

    def test_uses_default_pixel_bitmask(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            bpm, "PixelBitMask",
            lambda: calls.append(True) or FakePixelBitMask(),
        )
        dark = np.zeros((3, 4))
        dark[1, 1] = 1
        result = bpm.combine_bpm_masks(dark)
        assert calls == [True] and result[1, 1] & BITS["BADDARK"]

    def test_output_is_int64_and_inputs_are_unchanged(self):
        dark = np.zeros((3, 4), dtype=np.uint8)
        flat = np.zeros_like(dark)
        dark[1, 1], flat[2, 2] = 2, 4
        dark_copy, flat_copy = dark.copy(), flat.copy()
        result = bpm.combine_bpm_masks(dark, flat, pixmask=FakePixelBitMask())
        assert result.dtype == np.int64
        np.testing.assert_array_equal(dark, dark_copy)
        np.testing.assert_array_equal(flat, flat_copy)


class TestChipBadrows:
    def test_none_is_preserved(self):
        assert bpm._chip_badrows(None, 0) is None

    def test_simple_objects_are_filtered_by_chip(self):
        entries = [SimpleNamespace(chip=0, row=2), SimpleNamespace(chip=1, row=4), SimpleNamespace(chip=0, row=6)]
        np.testing.assert_array_equal(bpm._chip_badrows(entries, 0), [2, 6])

    def test_structured_array_entries_are_supported(self):
        entries = np.array([(0, 3), (2, 8), (2, 9)], dtype=[("chip", int), ("row", int)])
        np.testing.assert_array_equal(bpm._chip_badrows(entries, 2), [8, 9])

    def test_mapping_entries_are_supported(self):
        entries = [{"chip": 1, "row": 5}, {"chip": 0, "row": 7}]
        np.testing.assert_array_equal(bpm._chip_badrows(entries, 1), [5])

    def test_no_matching_rows_returns_empty_integer_array(self):
        result = bpm._chip_badrows([SimpleNamespace(chip=1, row=4)], 0)
        assert result.dtype.kind == "i" and result.size == 0


class TestMkbpmWorkflow:
    def test_requires_darkid_before_constructing_apload(self, monkeypatch):
        monkeypatch.setattr(
            bpm.apload, "ApLoad",
            lambda **kwargs: pytest.fail("ApLoad should not be called"),
        )
        with pytest.raises(ValueError, match="darkid"):
            bpm.build_bpm(BPMID)

    def test_apload_configuration_and_unlock_are_forwarded(self, environment):
        bpm.build_bpm(
            BPMID, apred="test", telescope="lco25m", darkid=DARKID,
            flatid=FLATID, unlock=True, verbose=True)
        assert environment.apload_calls == [{"apred": "test", "telescope": "lco25m"}]
        assert environment.lock_calls[0] == (
            lockfile(environment), {"waittime": 10, "unlock": True})

    def test_existing_complete_product_returns_without_reading_inputs(
            self, environment, capsys):
        for filename in output_files(environment):
            filename.write_bytes(b"existing")
        assert bpm.build_bpm(
            BPMID, darkid=DARKID, flatid=FLATID, verbose=True) is None
        assert environment.read_calls == []
        assert not any(kwargs.get("lock") for _, kwargs in environment.lock_calls)
        assert f"bpm product {BPMID} already exists" in capsys.readouterr().out

    def test_clobber_rebuilds_existing_products(self, environment):
        for filename in output_files(environment):
            filename.write_text("old")
        bpm.build_bpm(BPMID, darkid=DARKID, flatid=FLATID, clobber=True)
        assert all(filename.stat().st_size > 3 for filename in output_files(environment))

    def test_partial_products_are_removed_and_all_chips_built(self, environment):
        output_files(environment)[0].write_text("stale")
        assert bpm.build_bpm(BPMID, darkid=DARKID, flatid=FLATID) is None
        assert all(filename.exists() for filename in output_files(environment))
        assert_lock_cleared(environment)

    def test_reads_dark_and_flat_mask_extension_three_for_each_chip(self, environment):
        bpm.build_bpm(BPMID, darkid=DARKID, flatid=FLATID)
        assert len(environment.read_calls) == 6
        assert all(ext == 3 for _, ext in environment.read_calls)
        assert [path.name for path, _ in environment.read_calls] == [
            f"apDark-a-{DARKID:08d}.fits", f"apFlat-a-{FLATID:08d}.fits",
            f"apDark-b-{DARKID:08d}.fits", f"apFlat-b-{FLATID:08d}.fits",
            f"apDark-c-{DARKID:08d}.fits", f"apFlat-c-{FLATID:08d}.fits",
        ]

    def test_numerical_masks_are_written_as_int16(self, environment):
        environment.darkmasks["a"][1, 2] = 1
        environment.flatmasks["a"][3, 4] = 1
        bpm.build_bpm(BPMID, darkid=DARKID, flatid=FLATID)
        with bpm.fits.open(output_files(environment)[0]) as hdus:
            data = hdus[0].data
            assert data.dtype.kind == "i" and data.dtype.itemsize == 2
            assert data[1, 2] == BITS["BADDARK"] | BITS["BADPIX"]
            assert data[3, 4] == BITS["BADFLAT"] | BITS["BADPIX"]

    def test_badrows_are_applied_only_to_selected_chip(self, environment):
        rows = [SimpleNamespace(chip=0, row=2), SimpleNamespace(chip=2, row=6)]
        bpm.build_bpm(BPMID, darkid=DARKID, flatid=FLATID, badrow=rows)
        data = []
        for filename in output_files(environment):
            with bpm.fits.open(filename) as hdus:
                data.append(hdus[0].data.copy())
        assert np.all(data[0][2, :] & BITS["BADPIX"])
        assert not np.any(data[1])
        assert np.all(data[2][6, :] & BITS["BADPIX"])

    def test_primary_header_contains_inputs_versions_and_history(self, environment, monkeypatch):
        monkeypatch.setattr(bpm.getpass, "getuser", lambda: "tester")
        monkeypatch.setattr(bpm.socket, "gethostname", lambda: "host")
        bpm.build_bpm(BPMID, darkid=DARKID, flatid=FLATID)
        with bpm.fits.open(output_files(environment)[0]) as hdus:
            header = hdus[0].header
            assert header["EXTNAME"] == "BPM"
            assert Path(header["DARKFILE"]).name == f"apDark-a-{DARKID:08d}.fits"
            assert Path(header["FLATFILE"]).name == f"apFlat-a-{FLATID:08d}.fits"
            assert header["V_APRED"] == "git-test"
            assert header["APRED"] == "drp-test"
            assert any("tester on host" in line for line in header["HISTORY"])

    def test_flat_is_optional_and_header_records_no_flat(self, environment):
        assert bpm.build_bpm(BPMID, darkid=DARKID, flatid=None) is None
        assert len(environment.read_calls) == 3
        with bpm.fits.open(output_files(environment)[0]) as hdus:
            assert hdus[0].header["FLATFILE"] in ("", "None", "NONE")

    def test_zero_flat_id_is_treated_as_no_flat(self, environment):
        bpm.build_bpm(BPMID, darkid=DARKID, flatid=0)
        assert len(environment.read_calls) == 3
        assert all("Dark" in path.name for path, _ in environment.read_calls)

    def test_output_directories_are_created(self, environment):
        directory = environment.root / "bpm"
        assert not list(directory.glob("apBPM-?-*.fits"))
        bpm.build_bpm(BPMID, darkid=DARKID, flatid=FLATID)
        assert all(filename.parent == directory and filename.exists() for filename in output_files(environment))

    def test_lock_is_cleared_after_read_failure(self, environment, monkeypatch):
        monkeypatch.setattr(bpm.fits, "getdata", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("boom")))
        with pytest.raises(OSError, match="boom"):
            bpm.build_bpm(BPMID, darkid=DARKID, flatid=FLATID)
        assert_lock_cleared(environment)

    def test_lock_is_cleared_after_write_failure(self, environment, monkeypatch):
        monkeypatch.setattr(bpm.fits.HDUList, "writeto", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")))
        with pytest.raises(OSError, match="write failed"):
            bpm.build_bpm(BPMID, darkid=DARKID, flatid=FLATID)
        assert_lock_cleared(environment)

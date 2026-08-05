from pathlib import Path

import pytest

from apogee_drp.apred.process import process


class FakeLoad:
    apred = "test"

    def __init__(self, root):
        self.root = Path(root)

    def cmjd(self, number):
        return "60000"

    def filename(self, root, num=None, mjd=None, chips=False, chip=None, dir=False):
        directory = self.root / ("exp" if root in ("R", "2D") else "cal")
        if dir:
            return str(directory)
        if chips:
            return str(directory / f"ap{root}-{int(num):08d}.fits")
        return str(directory / f"ap{root}-{chip}-{int(num):08d}.fits")


def test_process_dispatches_three_chips_with_idl_options(tmp_path):
    load = FakeLoad(tmp_path)
    calls = []

    def process_3d(input_file, output_file, **kwargs):
        calls.append((input_file, output_file, kwargs))
        Path(output_file).touch()

    records = process(
        123, load=load, dark=10, flatid=20, littrowid=30,
        persist=40, nfs=1, nocr=True, doap3dproc=True,
        maxread=[5, 6, 7], process_3d=process_3d,
    )

    assert [record.chip for record in records] == ["a", "b", "c"]
    assert len(calls) == 3
    assert calls[0][2]["up_the_ramp"] is False
    assert calls[0][2]["nfowler"] == 1
    assert calls[0][2]["detect_cosmic_rays"] is False
    assert [call[2]["max_read"] for call in calls] == [5, 6, 7]
    assert calls[0][2]["dark"].endswith("apDark-a-00000010.fits")
    assert calls[0][2]["bpm"].endswith("apBPM-a-00000010.fits")
    assert calls[1][2]["littrow"].endswith("apLittrow-b-00000030.fits")
    assert calls[0][2]["littrow"] is None


def test_process_extracts_after_all_ap2d_files_exist(tmp_path):
    load = FakeLoad(tmp_path)
    for chip in "abc":
        filename = Path(load.filename("2D", num=123, chips=True))
        filename = Path(str(filename).replace("2D-", f"2D-{chip}-"))
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.touch()
    calls = []

    def process_2d(*args, **kwargs):
        calls.append((args, kwargs))

    records = process(123, load=load, psfid=50, fluxid=60, waveid=70,
                        doap2dproc=True, process_2d=process_2d)

    assert len(records) == 1 and records[0].stage == "2d"
    assert calls[0][0][0].endswith("exp/00000123")
    assert calls[0][0][1].endswith("cal/00000050")
    assert calls[0][1]["extract_type"] == 4


def test_process_rejects_conflicting_legacy_id(tmp_path):
    with pytest.raises(ValueError, match="Conflicting dark"):
        process(123, load=FakeLoad(tmp_path), darkid=1, dark=2,
                  doap3dproc=True, process_3d=lambda *args, **kwargs: None)


def test_process_validates_three_element_maxread(tmp_path):
    with pytest.raises(ValueError, match="three-element"):
        process(123, load=FakeLoad(tmp_path), doap3dproc=True,
                  maxread=[1, 2], process_3d=lambda *args, **kwargs: None)

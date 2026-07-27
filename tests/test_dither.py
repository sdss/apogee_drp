import numpy as np

from apogee_drp.apred.visit.dither import dither_pairs


def _shifts(values, sn=None):
    n = len(values)
    dtype = [
        ("index", "i8"),
        ("framenum", "U8"),
        ("shift", "f4"),
        ("sn", "f4"),
    ]
    out = np.zeros(n, dtype=dtype)
    out["index"] = np.arange(n)
    out["framenum"] = [f"{101+i:08d}" for i in range(n)]
    out["shift"] = values
    out["sn"] = 10 if sn is None else sn
    return out


def test_dither_pairs_idl_example():
    pairs = dither_pairs(_shifts([0.0, 0.01, 0.49, -0.02, 0.51]))
    assert len(pairs) == 2
    # The pair with the largest original shift is promoted to reference.
    np.testing.assert_array_equal(pairs[0].index, [4, 1])
    np.testing.assert_allclose(pairs[0].oldshift, [0.51, 0.01])
    assert pairs[0].relshift == np.float32(0.50)
    assert pairs[0].refshift == np.float32(0.0)
    np.testing.assert_array_equal(pairs[1].index, [2, 0])
    assert pairs[1].refshift == np.float32(0.51) - np.float32(0.49)


def test_negative_fraction_uses_idl_truncation():
    pairs = dither_pairs(_shifts([0.0, -0.51]))
    assert len(pairs) == 1
    assert pairs[0].relshift == np.float32(0.51)


def test_snsort_matches_idl_primary_frame_behavior():
    shifts = _shifts([0.0, 0.5, 0.01, 0.51], sn=[10, 2, 9, 8])
    pairs = dither_pairs(shifts, snsort=True)
    assert len(pairs) == 2
    # IDL applies MINSN only to the candidate partner.  A low-S/N frame can
    # therefore still become the primary frame later in the loop.
    assert any(1 in pair.index for pair in pairs)

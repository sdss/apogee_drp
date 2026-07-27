"""Small synthetic tests for the APOGEE 3-D to 2-D translation."""

import numpy as np
from astropy.io import fits

from ap3d import PIXMASK, detect_and_fix_cosmic_rays, process_cube


def test_cosmic_ray_detection_and_fix():
    dcounts = np.full((3, 15), 100.0, dtype=np.float32)
    dcounts[1, 7] = 1000.0
    sat = np.zeros((3, 3), dtype=np.int32)
    fixed, median, variability, events = detect_and_fix_cosmic_rays(
        dcounts, sat, noise=5.0
    )
    assert len(events) == 1
    assert events[0].x == 1
    assert events[0].read == 8
    assert fixed[1, 7] == median[1] == 100.0
    assert variability.shape == (3,)


def test_fowler_collapse_without_reference_output():
    # A small ny is intentionally rejected: the detector geometry is fixed,
    # so use a full science image but only three reads.
    y, x = np.indices((2048, 2048))
    signal = (10 + (x % 3)).astype(np.float32)
    cube = np.stack([1000 + i * signal for i in range(4)])
    header = fits.Header({"CHIP": "a", "NFRAMES": 4, "EXPTIME": 42.588})
    result = process_cube(
        cube,
        header,
        detect_cosmic_rays=False,
        nfowler=1,
        use_reference=False,
    )
    np.testing.assert_allclose(result.flux, 3 * signal)
    assert result.flux.shape == (2048, 2048)
    assert result.mask.dtype == np.uint16


def test_saturation_is_flagged():
    cube = np.zeros((4, 2048, 2048), dtype=np.float32)
    cube[:] = np.arange(4)[:, None, None] * 10
    cube[2:, 100, 200] = 65535
    result = process_cube(
        cube,
        fits.Header({"CHIP": "a", "NFRAMES": 4}),
        detect_cosmic_rays=False,
        nfowler=1,
        use_reference=False,
    )
    assert result.mask[100, 200] & PIXMASK.getval("SATPIX")

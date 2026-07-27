"""APOGEE visit-level reduction.

The public API is deliberately small.  Numerical stages live in focused
modules so they can be compared with their IDL counterparts independently.
"""

from .dither import DitherPair, dither_pairs
from .combine import (
    CombinedSpectrum,
    combine_pair_frames,
    combine_spectra,
    convert_lsf_to_half_pixel,
    convert_wcoef_to_half_pixel,
    dither_combine,
    estimate_continuum,
    interlace_frame_pair,
    interlace_pair,
    sinc_interlaced,
)
from .driver import (
    FrameFailure,
    PlanFailure,
    ShiftRecord,
    VisitBackend,
    VisitResult,
    ap1dvisit,
)
from .io import BADERR, read_cframes, write_cframes
from .flux import BADMASK, H_ZEROPOINT_FLAMBDA, flux_calibrate
from .qa import check
from .shift import DitherShiftResult, LinePeak, dither_shift
from .output import (
    VisitProductResult,
    build_visit_hdul,
    write_plate_products,
    write_visit_products,
)

__all__ = [
    "BADERR",
    "BADMASK",
    "CombinedSpectrum",
    "DitherPair",
    "DitherShiftResult",
    "FrameFailure",
    "H_ZEROPOINT_FLAMBDA",
    "PlanFailure",
    "LinePeak",
    "ShiftRecord",
    "VisitBackend",
    "VisitProductResult",
    "VisitResult",
    "ap1dvisit",
    "build_visit_hdul",
    "check",
    "combine_pair_frames",
    "combine_spectra",
    "convert_lsf_to_half_pixel",
    "convert_wcoef_to_half_pixel",
    "dither_combine",
    "dither_pairs",
    "dither_shift",
    "estimate_continuum",
    "flux_calibrate",
    "interlace_frame_pair",
    "interlace_pair",
    "read_cframes",
    "sinc_interlaced",
    "write_cframes",
    "write_plate_products",
    "write_visit_products",
]

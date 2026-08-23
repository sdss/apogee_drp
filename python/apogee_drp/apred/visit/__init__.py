"""APOGEE visit-level reduction.

The public API is deliberately small.  Numerical stages live in focused
modules so they can be compared with their IDL counterparts independently.
"""

from .dither import DitherPair, dither_pairs, find_dither_pairs
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
from .models import ChipFrame, VisitFrame
from .backend import NativeVisitBackendMixin
from .apload_backend import ApLoadVisitBackend
from .flux import BADMASK, H_ZEROPOINT_FLAMBDA, flux_calibrate
try:
    from .qa import check
except ImportError:  # The standalone translation bundle omits full Yanny I/O.
    check = None
from .shift import DitherShiftResult, LinePeak, dither_shift
from .products import (
    VisitProductResult,
    build_visit_hdul,
    write_visit_products,
)
from .plate import write_plate_products

__all__ = [
    "BADERR",
    "BADMASK",
    "CombinedSpectrum",
    "ChipFrame",
    "DitherPair",
    "DitherShiftResult",
    "FrameFailure",
    "H_ZEROPOINT_FLAMBDA",
    "PlanFailure",
    "LinePeak",
    "NativeVisitBackendMixin",
    "ApLoadVisitBackend",
    "ShiftRecord",
    "VisitBackend",
    "VisitFrame",
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
    "find_dither_pairs",
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

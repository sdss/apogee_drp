"""Compatibility entry point for the global native LSF calibration builder."""

from .lsf import build_lsf


def mklsf(
    lsfid,
    waveid,
    darkid=None,
    flatid=None,
    psfid=None,
    fiberid=None,
    clobber=False,
    full=False,
    newwave=False,
    pl=False,
    fibers=None,
    nowait=False,
    unlock=False,
    **options
):
    """Build full native apLSF products with a shared solution per fiber.

    Old argument names remain accepted. Additional options, including apred,
    telescope, native_degree, native_horder and line_catalog, go to build_lsf.
    Set global_fit=False only to explicitly select the former per-chip models.
    """
    return build_lsf(
        lsfid,
        waveid,
        darkid=darkid,
        flatid=flatid,
        psfid=psfid,
        fiberid=fiberid,
        clobber=clobber,
        full=full,
        newwave=newwave,
        plot=pl,
        fibers=fibers,
        nowait=nowait,
        unlock=unlock,
        **options
    )

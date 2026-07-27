"""Compatibility interface for the legacy ``apmkfluxcal.pro`` command."""

from .mkflux import mkflux


def apmkfluxcal(flatid, *, bbtemp=None, waveid=None, reproc=False,
                clobber=False, absolute=False, collapse=False, unlock=False,
                **kwargs):
    """Build a flux calibration using the maintained :func:`mkflux` path.

    ``absolute`` and ``collapse`` are accepted for IDL call compatibility;
    raw collapse belongs to the normal Python reduction workflow.
    """
    if absolute:
        kwargs.setdefault("holtz", True)
    return mkflux(
        [int(flatid)],
        waveid=waveid,
        bbtemp=bbtemp,
        clobber=clobber,
        onedclobber=reproc,
        unlock=unlock,
        **kwargs,
    )


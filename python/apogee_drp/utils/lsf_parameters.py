"""Lightweight APOGEE/Doppler packed-LSF format adapter.

The layout follows Doppler's unpack_ghlsf_params/repack_ghlsf_params, originally
based on the APOGEE IDL routines. This adapter requires only NumPy; importing a
spectral fitting package and loading its model libraries is unnecessary for I/O.
"""

import numpy as np


def unpack_parameters(values):
    """Decode one packed parameter vector, validating counts and polynomial orders."""
    a = np.asarray(values, float)
    if a.ndim != 1 or len(a) < 6 or not np.isfinite(a).all():
        raise ValueError("Expected a finite one-dimensional packed LSF array")
    position = 0

    def take(n):
        nonlocal position
        if position + n > len(a):
            raise ValueError("Truncated packed LSF array")
        value = a[position : position + n]
        position += n
        return value

    def integers(n):
        x = take(n)
        if np.any(x != np.floor(x)) or np.any(x < 0) or np.any(x > 100):
            raise ValueError("Invalid packed LSF count or polynomial order")
        return x.astype(int)

    binsize, offset = take(2)
    order = int(integers(1)[0])
    porder = integers(order + 1)

    def coefficients(orders):
        result = np.zeros((len(orders), max(orders) + 1))
        for i, p in enumerate(orders):
            result[i, : p + 1] = take(p + 1)
        return result

    gh = coefficients(porder)
    wtype, nwing = integers(2)
    if nwing < 1:
        raise ValueError("Wing parameters are required by this format adapter")
    wporder = integers(nwing)
    wings = coefficients(wporder)
    # Some files pad the packed vector with zeros.
    if np.any(a[position:] != 0):
        raise ValueError("Unexpected trailing packed LSF parameters")
    return dict(
        binsize=binsize,
        Xoffset=offset,
        Horder=order,
        Porder=porder,
        GHcoefs=gh,
        Wproftype=int(wtype),
        nWpar=int(nwing),
        WPorder=wporder,
        Wcoefs=wings,
    )


def pack_parameters(p):
    """Encode ascending-power GH and wing polynomial coefficient arrays."""
    values = [p["binsize"], p["Xoffset"], p["Horder"], *p["Porder"]]
    for row, order in zip(p["GHcoefs"], p["Porder"]):
        values.extend(row[: order + 1])
    values.extend([p["Wproftype"], p["nWpar"], *p["WPorder"]])
    for row, order in zip(p["Wcoefs"], p["WPorder"]):
        values.extend(row[: order + 1])
    result = np.asarray(values, float)
    unpack_parameters(result)
    return result

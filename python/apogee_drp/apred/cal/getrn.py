"""Read-noise measurements translated from ``getrn.pro``."""

from __future__ import annotations

import numpy as np


def _robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    center = np.median(values)
    return 1.4826 * np.median(np.abs(values - center))


def fowler_sample(cube, nfowler):
    """Collapse a ``(ny, nx, nread)`` cube using UTR or Fowler sampling."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("cube must have shape (ny, nx, nread)")
    nread = cube.shape[-1]
    if nfowler == 0:
        # Equally spaced up-the-ramp least-squares slope, scaled to total counts.
        x = np.arange(nread, dtype=float)
        weights = x - x.mean()
        return np.tensordot(cube, weights, axes=(-1, 0)) / np.sum(
            weights**2
        ) * (nread - 1)
    m = int(nfowler)
    if m < 1 or 2 * m > nread:
        raise ValueError("invalid Fowler sample count")
    return np.mean(cube[..., -m:], axis=-1) - np.mean(cube[..., :m], axis=-1)


def getrn(cubes1, cubes2, *, max_reads=47):
    """Measure read noise for three chip cube pairs.

    Parameters
    ----------
    cubes1, cubes2 : sequence of array-like
        Two exposures for chips a, b, and c.  Each cube uses NumPy order
        ``(ny, nx, nread)``.
    max_reads : int
        Match the IDL routine's standard 47-read truncation.

    Returns
    -------
    numpy.ndarray
        Three-row structured array containing the IDL ``rnlog`` fields.
    """
    if len(cubes1) != 3 or len(cubes2) != 3:
        raise ValueError("cubes1 and cubes2 must each contain three chips")
    dtype = [
        ("n", "i2", (6,)),
        ("m", "i2", (6,)),
        ("rn1", "f4", (4, 6)),
        ("rn1corr", "f4", (4, 6)),
        ("rn2", "f4", (4, 6)),
        ("rn2corr", "f4", (4, 6)),
        ("rn3", "f4", (4, 6)),
        ("rn4", "f4", (4, 6)),
    ]
    result = np.zeros(3, dtype=dtype)

    for ichip, (first, second) in enumerate(zip(cubes1, cubes2)):
        first = np.asarray(first, dtype=float)[..., :max_reads]
        second = np.asarray(second, dtype=float)[..., :max_reads]
        if first.shape != second.shape or first.ndim != 3:
            raise ValueError("paired cubes must have identical 3-D shapes")
        nreads = first.shape[-1]
        for sampling in range(6):
            m = 1 if sampling == 0 else sampling
            n = nreads if sampling == 0 else 2
            result["m"][ichip, sampling] = m
            result["n"][ichip, sampling] = n
            image1 = fowler_sample(first, sampling)
            image2 = fowler_sample(second, sampling)
            correction = np.sqrt(m * n * (n + 1.0) / (12.0 * (n - 1.0)))

            for quadrant in range(4):
                x0 = quadrant * 512 + 5
                x1 = min((quadrant + 1) * 512 - 4, first.shape[1])
                y0, y1 = 10, min(2041, first.shape[0])
                if x0 >= x1 or y0 >= y1:
                    continue
                section2 = image2[y0:y1, x0:x1]
                difference = image1[y0:y1, x0:x1] - section2
                rn1 = _robust_sigma(section2)
                rn2 = _robust_sigma(difference) / np.sqrt(2.0)
                result["rn1"][ichip, quadrant, sampling] = rn1
                result["rn1corr"][ichip, quadrant, sampling] = rn1 * correction
                result["rn2"][ichip, quadrant, sampling] = rn2
                result["rn2corr"][ichip, quadrant, sampling] = rn2 * correction
                if sampling == 1:
                    reads = first[y0:y1, x0:x1]
                    result["rn3"][ichip, quadrant, sampling] = (
                        _robust_sigma(np.diff(reads, axis=-1)) / np.sqrt(2.0)
                    )
    return result


def rnhtml(rn_tables, outfile):
    """Write the compact HTML summary formerly produced by ``RNHTML``."""
    from pathlib import Path

    rows = [
        "<html><body>",
        "Readout noise is given in DN (as measured)",
        "<table border='2'><tr><th>File</th><th>Chip</th>",
    ]
    rows.extend(
        f"<th>FS {fs}, Q{quad}</th>"
        for fs in range(6)
        for quad in range(4)
    )
    rows.append("</tr>")
    for name, table in rn_tables:
        for chip in range(3):
            rows.append(f"<tr><td>{name}</td><td>{chip}</td>")
            rows.extend(
                f"<td>{table['rn2'][chip, quad, fs]:.2f}</td>"
                for fs in range(6)
                for quad in range(4)
            )
            rows.append("</tr>")
    rows.append("</table></body></html>")
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("".join(rows), encoding="utf-8")
    return outfile

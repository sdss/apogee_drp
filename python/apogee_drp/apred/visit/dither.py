"""Dither pairing utilities for APOGEE visit reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class DitherPair:
    """Python representation of the IDL ``pairstr`` record."""

    framename: np.ndarray
    framenum: np.ndarray
    oldshift: np.ndarray
    shift: np.ndarray
    sn: np.ndarray
    refshift: np.float32
    relshift: np.float32
    nused: np.ndarray
    index: np.ndarray


def _field(records: Sequence[Any], name: str) -> np.ndarray:
    try:
        return np.asarray(records[name])
    except (IndexError, KeyError, TypeError, ValueError):
        values = []
        for row in records:
            if isinstance(row, dict):
                values.append(row[name])
            else:
                values.append(getattr(row, name))
        return np.asarray(values)


def dither_pairs(
    shifts: Sequence[Any],
    *,
    snsort: bool = False,
    minshift: float = 0.2,
    maxshift: float = 0.8,
    minsn: float = 3.0,
    verbose: bool = False,
) -> list[DitherPair]:
    """Pair dithered frames as ``apditherpairs.pro`` does.

    In particular, fractional shifts use truncation toward zero (IDL ``FIX``),
    not modulo or floor.  This matters for negative shifts.
    """

    nframes = len(shifts)
    if nframes < 2:
        raise ValueError("Only ONE frame input. Need at least TWO")

    indices = _field(shifts, "index").astype(np.int64)
    names = _field(shifts, "framenum").astype(str)
    numbers = names.astype(np.int64)
    shift_values = _field(shifts, "shift").astype(np.float32)
    sn = _field(shifts, "sn").astype(np.float32)
    nused = np.zeros(nframes, dtype=np.int64)

    if snsort:
        # IDL REVERSE(SORT()) puts the largest value first.
        order = np.argsort(sn, kind="stable")[::-1]
    else:
        order = np.arange(nframes)

    pairs: list[DitherPair] = []
    for position in range(nframes):
        index = int(order[position])
        if nused[index] != 0:
            continue
        relative = shift_values[order] - shift_values[index]
        fractional = relative - np.trunc(relative)
        shifted = np.abs(fractional) >= minshift
        if not np.any(shifted):
            if verbose:
                print(f"No frame shifted enough for {names[index]}")
            continue

        if snsort:
            passing = np.flatnonzero(
                shifted
                & (np.abs(fractional) <= maxshift)
                & (sn[order] > minsn)
                & (numbers[order] != numbers[index])
                & (nused[order] == 0)
            )
        else:
            passing = np.flatnonzero(
                shifted
                & (numbers[order] > numbers[index])
                & (nused[order] == 0)
            )
            if passing.size == 0:
                passing = np.flatnonzero(
                    shifted
                    & (numbers[order] < numbers[index])
                    & (nused[order] == 0)
                )
        if passing.size == 0:
            if verbose:
                print(f"No frame to pair with {names[index]}")
            continue

        partner = int(order[passing[0]])
        nused[index] += 1
        nused[partner] += 1
        oldshift = np.asarray(
            [shift_values[index], shift_values[partner]], dtype=np.float32
        )
        pair = DitherPair(
            framename=np.asarray([names[index], names[partner]]),
            framenum=np.asarray(
                [numbers[index], numbers[partner]], dtype=np.int64
            ),
            oldshift=oldshift.copy(),
            shift=oldshift.copy(),
            sn=np.asarray([sn[index], sn[partner]], dtype=np.float32),
            refshift=np.float32(0),
            relshift=np.float32(oldshift[1] - oldshift[0]),
            nused=np.zeros(2, dtype=np.int64),
            # IDL stores positions in shiftstr/allframes here.
            index=np.asarray([index, partner], dtype=np.int64),
        )
        pairs.append(pair)
        if verbose:
            print(f"Pairing {names[index]} with {names[partner]}")

    if not pairs:
        raise ValueError("NO PAIRS")

    pair_maxima = np.asarray([np.max(pair.shift) for pair in pairs])
    reference_pair = int(np.argmax(pair_maxima))
    if reference_pair:
        pairs.insert(0, pairs.pop(reference_pair))
    reference_shift = np.max(pairs[0].shift)
    for pair in pairs:
        pair.shift = np.asarray(reference_shift - pair.shift, dtype=np.float32)

    all_names = np.asarray([pair.framename for pair in pairs])
    for pair in pairs:
        pair.nused = np.asarray(
            [
                np.count_nonzero(all_names == pair.framename[0]),
                np.count_nonzero(all_names == pair.framename[1]),
            ],
            dtype=np.int64,
        )
        if pair.relshift > 0:
            pair.framename = pair.framename[::-1].copy()
            pair.framenum = pair.framenum[::-1].copy()
            pair.oldshift = pair.oldshift[::-1].copy()
            pair.shift = pair.shift[::-1].copy()
            pair.sn = pair.sn[::-1].copy()
            pair.nused = pair.nused[::-1].copy()
            pair.index = pair.index[::-1].copy()
        else:
            pair.relshift = np.float32(-pair.relshift)
        pair.refshift = np.float32(pair.shift[0])
    return pairs

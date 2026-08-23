"""Canonical in-memory data models for visit reduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import numpy as np
from astropy.io import fits

CHIPS = ("a", "b", "c")


@dataclass
class ChipFrame:
    """Arrays and metadata for one APOGEE detector chip."""

    flux: np.ndarray
    err: np.ndarray
    mask: np.ndarray
    header: fits.Header = field(default_factory=fits.Header)
    filename: str = ""
    wavelength: np.ndarray | None = None
    sky: np.ndarray | None = None
    skyerr: np.ndarray | None = None
    telluric: np.ndarray | None = None
    telluricerr: np.ndarray | None = None
    wcoef: np.ndarray | None = None
    lsfcoef: np.ndarray | None = None
    wavefile: str | None = None
    lsffile: str | None = None
    wave_dir: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, "err" if key == "error" else key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, "err" if key == "error" else key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, "err" if key == "error" else key, default)

    @property
    def error(self) -> np.ndarray:
        """Alias retained for routines translated with the IDL spelling."""

        return self.err

    @error.setter
    def error(self, value: np.ndarray) -> None:
        self.err = value

    def validate(self) -> None:
        """Validate the required pixel arrays."""

        shape = np.asarray(self.flux).shape
        if np.asarray(self.err).shape != shape or np.asarray(self.mask).shape != shape:
            raise ValueError("flux, err, and mask must have identical shapes")
        if len(shape) != 2:
            raise ValueError("chip arrays must have shape (nfiber, npix)")


@dataclass
class VisitFrame:
    """A three-chip exposure used throughout visit reduction."""

    chipa: ChipFrame
    chipb: ChipFrame
    chipc: ChipFrame
    shift: Any = field(default_factory=dict)
    tellstar: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[ChipFrame]:
        return iter((self.chipa, self.chipb, self.chipc))

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self.chip(CHIPS[key])
        if key in CHIPS:
            return self.chip(key)
        if key.startswith("chip") and key[-1] in CHIPS:
            return self.chip(key[-1])
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key.startswith("chip") and key[-1] in CHIPS:
            setattr(self, key, value)
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            self.metadata[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def chip(self, chip: str | int) -> ChipFrame:
        """Return a chip by letter or zero-based index."""

        name = CHIPS[chip] if isinstance(chip, int) else str(chip).lower()
        if name not in CHIPS:
            raise ValueError("chip must be 'a', 'b', 'c', or an index from 0 to 2")
        return getattr(self, f"chip{name}")

    def validate(self) -> None:
        """Validate every chip and require a common fiber count."""

        for chip in self:
            chip.validate()
        nfibers = {chip.flux.shape[0] for chip in self}
        if len(nfibers) != 1:
            raise ValueError("all chips must have the same number of fibers")

    @classmethod
    def from_mapping(cls, frame: Mapping[str, Any]) -> "VisitFrame":
        """Convert a legacy visit-frame dictionary."""

        def convert(name: str) -> ChipFrame:
            value = frame.get(f"chip{name}", frame.get(name))
            if isinstance(value, ChipFrame):
                return value
            if value is None:
                raise KeyError(f"chip{name}")
            kwargs = dict(value)
            if "error" in kwargs and "err" not in kwargs:
                kwargs["err"] = kwargs.pop("error")
            allowed = ChipFrame.__dataclass_fields__
            return ChipFrame(**{key: val for key, val in kwargs.items() if key in allowed})

        result = cls(*(convert(chip) for chip in CHIPS), shift=frame.get("shift", {}),
                     tellstar=frame.get("tellstar"))
        result.metadata.update({key: value for key, value in frame.items()
                                if key not in {"a", "b", "c", "chipa", "chipb", "chipc",
                                               "shift", "tellstar"}})
        return result

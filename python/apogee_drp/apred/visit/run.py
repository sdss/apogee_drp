"""Command-line entry point for the Python ap1dvisit translation."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .apload_backend import ApLoadVisitBackend
from .driver import ap1dvisit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m apogee_drp.apred.visit.run")
    result.add_argument("planfiles", nargs="+")
    result.add_argument("--apred", required=True)
    result.add_argument("--telescope", required=True)
    result.add_argument("--clobber", action="store_true")
    result.add_argument("--test", action="store_true")
    result.add_argument("--halt", action="store_true")
    result.add_argument("--dithonly", action="store_true")
    result.add_argument("--verbose", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    backend = ApLoadVisitBackend(
        apred=args.apred,
        telescope=args.telescope,
        verbose=args.verbose,
    )
    results = ap1dvisit(
        args.planfiles,
        backend,
        clobber=args.clobber,
        test=args.test,
        halt=args.halt,
        dithonly=args.dithonly,
        verbose=args.verbose,
    )
    failed = [result for result in results if result.errors]
    for result in results:
        state = "FAILED" if result.errors else "OK"
        logging.info(
            "%s %s: %d frame(s), %d failed",
            state,
            result.planfile,
            result.processed_frames,
            result.failed_frames,
        )
        for error in result.errors:
            logging.error("%s", error)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

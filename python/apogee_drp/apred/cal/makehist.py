"""Create nightly histogram products through the Python ``makehist`` CLI."""

from __future__ import annotations

import subprocess


def makehist(mjd, *, apred, dark=None, clobber=False, runner=subprocess.run):
    """Run the Python histogram builder and raise on failure."""
    command = ["makehist", f"{int(mjd):05d}", "--apred", str(apred)]
    if dark is not None:
        command.extend(["--darkid", str(int(dark))])
    if clobber:
        command.append("--clobber")
    return runner(command, check=True)


"""HTML summaries for dark calibration products."""

from html import escape
from pathlib import Path


def darkhtml(darkdir, rows):
    """Write ``html/darks.html`` from iterable dark-summary rows."""
    columns = ("num", "chip", "nreads", "nframes", "medrate", "nsat",
               "nhot", "nhotneigh", "nbad", "nneg")
    parts = ["<html><body><table border='1'><tr>"]
    parts.extend(f"<th>{name.upper()}</th>" for name in columns)
    parts.append("<th>Image</th><th>Plots</th></tr>")
    for row in rows:
        get = (lambda key: row[key]) if hasattr(row, "keys") else (
            lambda key: getattr(row, key)
        )
        parts.append("<tr>")
        parts.extend(f"<td>{escape(str(get(name)))}</td>" for name in columns)
        base = f"apDark-{get('chip')}-{int(get('num')):08d}"
        parts.append(f"<td><a href='../plots/{base}2.jpg'>Image</a></td>")
        parts.append(f"<td><a href='../plots/{base}.png'>Plots</a></td></tr>")
    parts.append("</table></body></html>")
    outfile = Path(darkdir) / "html" / "darks.html"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("".join(parts), encoding="utf-8")
    return outfile


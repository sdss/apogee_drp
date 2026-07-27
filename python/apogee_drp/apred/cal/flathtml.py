"""HTML summaries for flat calibration products."""

from pathlib import Path


def flathtml(flatdir, rows):
    """Write ``html/flats.html`` from iterable flat-summary rows."""
    parts = [
        "<html><body><table border='1'>",
        "<tr><th>ID</th><th>NFRAMES</th><th>A</th><th>B</th><th>C</th></tr>",
    ]
    for row in rows:
        get = (lambda key: row[key]) if hasattr(row, "keys") else (
            lambda key: getattr(row, key)
        )
        num = int(get("num"))
        parts.append(f"<tr><td>{num}</td><td>{get('nframes')}</td>")
        for chip in "abc":
            base = f"apFlat-{chip}-{num:08d}.jpg"
            parts.append(
                f"<td><a href='../plots/{base}'>"
                f"<img src='../plots/{base}' width='100'></a></td>"
            )
        parts.append("</tr>")
    parts.append("</table></body></html>")
    outfile = Path(flatdir) / "html" / "flats.html"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("".join(parts), encoding="utf-8")
    return outfile


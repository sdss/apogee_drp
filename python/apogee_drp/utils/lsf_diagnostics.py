"""Plots of filled-fiber widths and their agreement with measured peaks."""

from copy import deepcopy
from pathlib import Path
import numpy as np


def plot_sigma_check(model, lines, output):
    """Write a multipage PDF showing sigma curves and every usable peak.

    Dashed curves show the original neighbor estimate; solid curves show the
    adopted solution. Rejected candidates are shown separately. Peak overlays
    fit only centers, amplitudes and backgrounds, independently for each model.
    Training/validation labels retain the split used for the sigma correction.
    ``lines=None`` writes an explicit no-data status page. Returns the PDF path.
    Matplotlib is imported only when plotting is requested; models are unchanged.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    from .calibration_lsf import evaluate_held_out

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = model.diagnostics.get("sigma_refinement", {})
    fiber = model.diagnostics.get("fiber", "?")
    status = info.get("status", "NOT_CHECKED")
    original = deepcopy(model)
    if "initial_sigma_coefficients" in info:
        original.coefficients[0] = info["initial_sigma_coefficients"]
    candidate = None
    if not info.get("accepted") and "candidate_sigma_coefficients" in info:
        candidate = deepcopy(model)
        candidate.coefficients[0] = info["candidate_sigma_coefficients"]
    with PdfPages(output) as pdf:
        fig = Figure(figsize=(11, 8))
        axes = fig.subplots(3, 1, sharex=True)
        for chip, ax in enumerate(axes):
            pixel = np.linspace(0, model.npix - 1, 256)
            x = pixel + model.chip_offsets[chip]
            ax.plot(pixel, original.fields(x)[0], "--", label="Neighbor estimate")
            ax.plot(pixel, model.fields(x)[0], label="Adopted")
            if candidate is not None:
                ax.plot(
                    pixel,
                    candidate.fields(x)[0],
                    ":",
                    color="tab:red",
                    label="Rejected candidate",
                )
            if lines is not None:
                selected = lines.chip == chip
                p = lines.centers[selected] - model.chip_offsets[chip]
                for training, marker, label in [
                    (True, "|", "Training peak positions"),
                    (False, "x", "Held-out peak positions"),
                ]:
                    keep = lines.train[selected] == training
                    ax.scatter(
                        p[keep],
                        model.fields(lines.centers[selected][keep])[0],
                        marker=marker,
                        label=label,
                    )
            ax.set_ylabel(f'Chip {"abc"[chip]} sigma\n(native pixels)')
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8, ncol=2)
        axes[-1].set_xlabel("Native detector pixel")
        fig.suptitle(
            f"Fiber {fiber}: {status}\nOnly core sigma is refined; Hermite and wing parameters remain estimated."
        )
        if lines is None:
            axes[1].text(
                0.5,
                0.5,
                "No usable peaks available for this diagnostic",
                transform=axes[1].transAxes,
                ha="center",
            )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        pdf.savefig(fig)
        if lines is None:
            return output
        # Evaluate every peak without changing the stored training split.
        all_lines = deepcopy(lines)
        all_lines.train[:] = False
        _, before, _ = evaluate_held_out(original, all_lines)
        stats, after, _ = evaluate_held_out(model, all_lines)
        for start in range(0, len(lines.centers), 12):
            fig = Figure(figsize=(11, 10))
            axes = fig.subplots(4, 3).ravel()
            for ax, i in zip(axes, range(start, min(start + 12, len(lines.centers)))):
                valid = lines.valid[i]
                x = lines.x[i]
                ax.errorbar(
                    x[valid],
                    lines.flux[i][valid],
                    yerr=lines.error[i][valid],
                    fmt=".",
                    color="black",
                    ms=3,
                    lw=0.5,
                    label="Data",
                )
                ax.plot(x, before[i], "--", label="Neighbor estimate")
                ax.plot(x, after[i], label="Adopted")
                p = lines.centers[i] - model.chip_offsets[lines.chip[i]]
                ax.set_title(
                    f'{"abc"[lines.chip[i]]}: pixel {p:.1f}, {"train" if lines.train[i] else "held out"}\nRMS/peak {stats[i]["rms_peak_relative"]:.1%}',
                    fontsize=9,
                )
                ax.set_xlabel("Pixel offset")
                ax.set_ylabel("Flux")
            for ax in axes[min(12, len(lines.centers) - start) :]:
                ax.set_visible(False)
            axes[0].legend(fontsize=7)
            fig.suptitle(f"Fiber {fiber}: measured peaks — {status}")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
    return output


def plot_calibration(calibration, output):
    """Write a run-wide PDF of fit status, held-out residuals and LSF fields.

    Covers every attempted fiber, including unavailable models. Page one shows
    status, observed-peak residuals and sigma maps; page two maps all model
    fields over fiber and native detector pixel. Blank cells indicate missing
    models or unavailable validation, never a zero residual. Filled fibers use
    their own peak-check statistics, not donor-fiber diagnostics. Metrics are
    descriptive, not an automatic science-quality certification. No raw spectra
    are needed; this also works on a saved native-pixel calibration.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    from .pixel_lsf import NativePixelLSF

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fibers = np.array(sorted(set(calibration.status) | set(calibration.models)))
    if not len(fibers):
        raise ValueError("No attempted fibers to plot")
    if any(not isinstance(m, NativePixelLSF) for m in calibration.models.values()):
        raise ValueError("Run overview currently requires native-pixel models")
    # Physical fiber coordinates keep gaps and failed fibers visible.
    size = max(300, int(fibers.max()))
    horder = max((m.horder for m in calibration.models.values()), default=5)
    names = (
        ["Core sigma"]
        + [f"H{h}" for h in range(3, horder + 1)]
        + ["Wing fraction", "Wing sigma"]
    )
    maps = np.full((len(names), 3, size, 64), np.nan)
    quality = np.full((3, size), np.nan)
    counts = np.zeros((3, size), int)
    status = []
    labels = [
        "Direct fit",
        "Filled: sigma checked",
        "Filled: estimate retained",
        "Unavailable",
    ]
    for fiber in fibers:
        model = calibration.models.get(int(fiber))
        if model is None:
            status.append(3)
            continue
        filled = model.diagnostics.get("filled", False)
        status.append(
            1
            if filled and model.diagnostics.get("sigma_data_constrained")
            else 2 if filled else 0
        )
        if filled:
            info = model.diagnostics.get("sigma_refinement", {})
            stats = info.get("after" if info.get("accepted") else "before", {}).get(
                "per_line", []
            )
        else:
            stats = model.diagnostics.get("validation", [])
        for chip in range(3):
            rows = [r for r in stats if r["chip"] == chip]
            counts[chip, fiber - 1] = len(rows)
            if rows:
                quality[chip, fiber - 1] = 100 * np.median(
                    [r["rms_peak_relative"] for r in rows]
                )
            pixels = np.linspace(0, model.npix - 1, 64)
            fields = model.fields(pixels + model.chip_offsets[chip])
            # Support varying GH order without relabeling wing fields.
            indices = (
                [0]
                + list(range(1, model.horder - 1))
                + [len(names) - 2, len(names) - 1]
            )
            for src, dest in enumerate(indices):
                maps[dest, chip, fiber - 1] = fields[src]
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    with PdfPages(output) as pdf:
        fig = Figure(figsize=(12, 11))
        axes = fig.subplots(3, 2)
        for category, label in enumerate(labels):
            sel = np.array(status) == category
            axes[0, 0].scatter(
                fibers[sel],
                np.full(sel.sum(), category),
                s=14,
                color=colors[category],
                label=f"{label}: {sel.sum()}",
            )
        axes[0, 0].set(
            yticks=range(4),
            yticklabels=labels,
            xlabel="Fiber",
            title="Fit status — all attempted fibers",
        )
        axes[0, 0].legend(fontsize=7, loc="upper center")
        for chip, color in enumerate(["tab:red", "tab:green", "tab:blue"]):
            axes[0, 1].scatter(
                np.arange(1, size + 1),
                quality[chip],
                s=10,
                color=color,
                label=f'Chip {"abc"[chip]}',
            )
        axes[0, 1].set(
            xlabel="Fiber",
            ylabel="Median held-out RMS / peak (%)",
            title="Residuals against measured peaks",
        )
        axes[0, 1].legend(fontsize=8)
        for chip, ax in enumerate([axes[1, 0], axes[1, 1], axes[2, 0]]):
            im = ax.imshow(
                np.ma.masked_invalid(maps[0, chip]),
                origin="lower",
                aspect="auto",
                extent=(0, 1, 0.5, size + 0.5),
            )
            ax.set(
                xlabel="Fraction of native detector pixel range",
                ylabel="Fiber",
                title=f'Chip {"abc"[chip]} core sigma (native pixels)',
            )
            fig.colorbar(im, ax=ax)
        ax = axes[2, 1]
        for chip, color in enumerate(["tab:red", "tab:green", "tab:blue"]):
            ax.plot(
                np.arange(1, size + 1),
                counts[chip],
                ".",
                color=color,
                label=f'Chip {"abc"[chip]}',
            )
        ax.set(
            xlabel="Fiber",
            ylabel="Held-out peaks",
            title="Validation coverage (zero means unavailable)",
        )
        ax.legend(fontsize=8)
        fig.suptitle(
            "Native LSF calibration overview\nBlank maps/residuals mean unavailable data. Filled LSF shapes remain estimated.",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig)
        fig = Figure(figsize=(12, max(8, len(names) * 1.7)))
        axes = fig.subplots(len(names), 3, squeeze=False)
        for field, name in enumerate(names):
            finite = maps[field][np.isfinite(maps[field])]
            limits = (
                {}
                if not len(finite)
                else dict(vmin=float(finite.min()), vmax=float(finite.max()))
            )
            for chip, ax in enumerate(axes[field]):
                im = ax.imshow(
                    np.ma.masked_invalid(maps[field, chip]),
                    origin="lower",
                    aspect="auto",
                    extent=(0, 1, 0.5, size + 0.5),
                    **limits,
                )
                ax.set(title=f'{name} — chip {"abc"[chip]}', ylabel="Fiber")
                fig.colorbar(im, ax=ax)
        for ax in axes[-1]:
            ax.set_xlabel("Fraction of native detector pixel range")
        fig.suptitle(
            "LSF shape across all fibers and detectors\nShared color limits within each field; inspect discontinuities and outliers."
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
    return output

from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from .analysis import PHASE_MODE, model

SIGNIFICANCE_FOR_ANGLE = 2.0
"""The angle of c is given only when |c| exceeds this many standard deviations."""
FONTS = {"axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.titlesize": 14}
"""Font sizes of the axis labels, tick labels and figure title of both figures."""


@plt.rc_context(FONTS)
def plot_pair_fits(ds_fit: xr.Dataset):
    """
    Plots, for every measured pair, the victim's response over the complex tone z (Re z, Im z), as in
    Nuerbolati et al., arXiv:2204.02946, Fig. 2(b):

    - Ramsey (phase mode): S = (1 + cos(Ramsey phase - phase at z = -c)) / 2, from 0 to 1. Population mode: P_e.
    - Background and white lines: the fit, over the measured region only; circles: the measured points, on the same
      colour scale. The lines of equal response are circles around z = -c (red x, +/- 1 sigma), the tone that cancels
      the leak. Without crosstalk they would be centred on z = 0 (black +), and all the points of a ring would have
      the same colour.
    - Title: |c|, and when |c| > 2 sigma, arg c and the angle of the correction -c (arg c + 180 deg), both measured
      from the Re z axis.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    pairs = [str(p) for p in ds_fit.qubit_pair.values]
    ncols = min(len(pairs), 4)
    nrows = int(np.ceil(len(pairs) / ncols))
    legend_columns = 2 if ncols >= 3 else 1
    legend_height = 0.3 * int(np.ceil(4 / legend_columns)) + 0.3  # inches below the panels for the legend
    height = 4.6 * nrows + legend_height + 0.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, height), squeeze=False)
    for ax in axes.flat[len(pairs) :]:
        ax.axis("off")
    for ax, pair in zip(axes.flat, pairs):
        fit = ds_fit.sel(qubit_pair=pair)
        measured = np.isfinite(fit.obs.values)
        z, obs = (fit.z_re.values + 1j * fit.z_im.values)[measured], fit.obs.values[measured]
        c, c_err = complex(fit.c_re, fit.c_im), float(fit.c_err)
        a, e0 = float(fit.curvature), float(fit.offset)
        center = complex(fit.center_re, fit.center_im) if "center_re" in fit else 0j
        phase_mode = str(fit["mode"].values) == PHASE_MODE

        # Map of the fitted response over the measured region (grid centre +/- 1.12 times the largest tone)
        half = 1.12 * np.max(np.abs(z - center))
        x = np.linspace(center.real - half, center.real + half, 301)
        y = np.linspace(center.imag - half, center.imag + half, 301)
        grid = x[None, :] + 1j * y[:, None]
        if phase_mode:
            # the phase relative to its value at z = -c, where the leak is cancelled (S = 1)
            fitted, points = 0.5 * (1 + np.cos(model(grid, a, c, e0) - e0)), 0.5 * (1 + np.cos(obs - e0))
            vmin, vmax, label = 0.0, 1.0, "S"
        else:
            fitted, points = model(grid, a, c, e0), obs
            vmin, vmax = min(fitted.min(), points.min()), max(fitted.max(), points.max())
            label = "P_e"
        extent = (x[0], x[-1], y[0], y[-1])
        im = ax.imshow(fitted, extent=extent, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.contour(x, y, fitted, levels=np.linspace(vmin, vmax, 11)[1:-1], colors="w", linewidths=0.6, alpha=0.7)
        ax.scatter(
            z.real, z.imag, c=points, cmap="viridis", vmin=vmin, vmax=vmax, s=60, edgecolors="k", linewidths=1, zorder=3
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(label)
        ax.plot(0, 0, "k+", ms=14, mew=2, zorder=4)
        ax.plot(-c.real, -c.imag, "rx", ms=11, mew=2.5, zorder=5)
        ax.add_patch(Circle((-c.real, -c.imag), c_err, fill=False, ec="r", lw=1.2, zorder=5))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xlabel("Re z")
        ax.set_ylabel("Im z")
        failed = "" if bool(fit.success) else " (failed)"
        if abs(c) > SIGNIFICANCE_FOR_ANGLE * c_err:
            arg_c = np.rad2deg(np.angle(c))
            angle = f"arg c = {arg_c:.0f} deg; cancel with -c (x) at {np.rad2deg(np.angle(-c)):.0f} deg"
        else:
            angle = "arg c: not significant"
        ax.set_title(f"{pair}{failed}: |c| = {100 * abs(c):.2f} +/- {100 * c_err:.2f} %\n{angle}", fontsize=11)
    handles = [
        Line2D([], [], marker="o", ls="", mfc="0.6", mec="k", label="measured (same colour scale)"),
        Line2D([], [], color="0.4", lw=0.8, label="equal response, from the fit (circles around -c)"),
        Line2D([], [], marker="+", ls="", color="k", mew=2, ms=11, label="z = 0: no tone"),
        Line2D(
            [], [], marker="x", ls="", color="r", mew=2, ms=10, label="-c: the tone that cancels the leak (+/-1 sigma)"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=legend_columns, fontsize=11, frameon=False)
    fig.suptitle(
        "Drive-line crosstalk over the tone z (without crosstalk, circles centred on +)\n"
        "S = (1 + cos(Ramsey phase - phase at -c)) / 2; population mode: P_e"
    )
    fig.tight_layout(rect=(0, legend_height / height, 1, 1))
    return fig


@plt.rc_context(FONTS)
def plot_crosstalk_matrix(matrix: dict, names: List[str], unmeasurable: Iterable[Tuple[str, str]] = ()):
    """Plots |c[victim, source]| (in % of amplitude) and the phase of c as the state holds them after this run.

    Outlines: green, written by this run; red, measured by this run without a passing fit (the stored value, if any,
    is kept). Hatched: stored value whose victim LO or source frequency changed since. "n/a": pair this node cannot
    measure. Empty: never measured.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    n = len(names)
    c = matrix["c"]
    unmeasurable = set(unmeasurable)
    fontsize = 10 if n <= 4 else 7 if n <= 8 else 5
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 4 + 0.75 * n), max(5, 2.5 + 0.38 * n)))
    panels = (
        (axes[0], 100 * np.abs(c), "|c| [% of amplitude]", "magma", 0, None, "{:.1f}"),
        (axes[1], np.rad2deg(np.angle(c)), "phase of c [deg]", "twilight", -180, 180, "{:.0f}"),
    )
    for ax, data, title, cmap, vmin, vmax, fmt in panels:
        ax.set_facecolor("whitesmoke")
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for j in range(n):  # victim
            for i in range(n):  # source
                if i == j:
                    ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, color="lightgray"))
                    continue
                if (names[i], names[j]) in unmeasurable and not np.isfinite(data[j, i]):
                    ax.text(i, j, "n/a", ha="center", va="center", fontsize=fontsize, color="gray")
                if matrix["stale"][j, i]:
                    ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, hatch="///", ec="gray", lw=0))
                if matrix["updated"][j, i] or matrix["failed"][j, i]:
                    color = "limegreen" if matrix["updated"][j, i] else "red"
                    ax.add_patch(Rectangle((i - 0.45, j - 0.45), 0.9, 0.9, fill=False, ec=color, lw=2))
                if np.isfinite(data[j, i]):
                    # Dark text on bright cells, white text on dark ones
                    r, g, b, _ = im.cmap(im.norm(data[j, i]))
                    color = "k" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "w"
                    ax.text(i, j, fmt.format(data[j, i]), ha="center", va="center", fontsize=fontsize, color=color)
        ax.set_xticks(range(n), names, rotation=90)
        ax.set_yticks(range(n), names)
        ax.set_xlabel("source qubit")
        ax.set_ylabel("victim qubit")
        ax.set_title(title)
    handles = [
        Patch(fill=False, ec="limegreen", lw=2, label="updated by this run"),
        Patch(fill=False, ec="red", lw=2, label="failed in this run (stored value kept)"),
        Patch(fill=False, hatch="///", ec="gray", label="stale (victim LO or source frequency changed)"),
        Patch(fc="whitesmoke", ec="gray", label="empty: never measured, n/a: not measurable"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11, frameon=False)
    fig.suptitle(
        "Drive-line crosstalk matrix in the state, c[victim <- source]\n(compensation: play -c on the victim line)"
    )
    fig.tight_layout(rect=(0, 0.12 if n > 8 else 0.19, 1, 1))
    return fig

"""Plotting module for the JAZZ2-N CZ amplitude calibration."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qualibration_libs.core import BatchableList


def plot_raw_data_with_fit(ds_fit: xr.Dataset, qubit_pairs: BatchableList) -> plt.Figure:
    """Plot the JAZZ2-N data and sinc fit per qubit pair.

    For each qubit pair we show two stacked panels:

    * Top: ``p00`` as a 2D map versus the repetition count N = 2k and the
      amplitude scale, with the fitted optimal amplitude drawn as a vertical
      line.
    * Bottom: the averaged-over-N curve ``p_avg(amp)`` together with the
      fitted sinc model and the optimum, so the fit can be eyeballed.
    """
    n_pairs = len(qubit_pairs)
    cols = min(4, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(2 * rows, cols, figsize=(4.5 * cols, 5.5 * rows), squeeze=False)

    for i, qp in enumerate(qubit_pairs):
        row, col = divmod(i, cols)
        ax_map = axes[2 * row, col]
        ax_avg = axes[2 * row + 1, col]
        qp_name = qp.name
        fr = ds_fit.sel(qubit_pair=qp_name)

        amps_scale = fr.amp.values
        amps_abs = fr["amp_full"].values if "amp_full" in fr.coords else amps_scale
        n_values = fr.N.values

        # --- Top: 2D heatmap of P_|00> ---
        p_map = fr["p"].transpose("N", "amp")
        xg, yg = np.meshgrid(amps_scale, n_values)
        pcm = ax_map.pcolormesh(xg, yg, p_map.values, cmap="magma", shading="auto")

        opt_scale = float(fr.optimal_amplitude_scale.values)
        opt_method = str(fr.fit_method.values)
        if np.isfinite(opt_scale):
            ax_map.axvline(opt_scale, color="lime", lw=2, label=f"opt = {opt_scale:.4f} ({opt_method})")

        def amp_scale_to_abs(s, abs_values=amps_abs, scale_values=amps_scale):
            return np.interp(s, scale_values, abs_values)

        def amp_abs_to_scale(a, abs_values=amps_abs, scale_values=amps_scale):
            return np.interp(a, abs_values, scale_values)

        secax = ax_map.secondary_xaxis("top", functions=(amp_scale_to_abs, amp_abs_to_scale))
        secax.set_xlabel("Amplitude (V)")
        ax_map.set_title(qp_name)
        ax_map.set_xlabel("Amplitude scale (a.u.)")
        ax_map.set_ylabel("Repetition N = 2k")
        ax_map.legend(loc="upper right", fontsize=8)
        cbar = fig.colorbar(pcm, ax=ax_map, shrink=0.85)
        cbar.set_label("$P_{|00\\rangle}$")

        # --- Bottom: averaged P_|00> with sinc fit ---
        if "p_avg" in fr.data_vars:
            ax_avg.plot(amps_scale, fr["p_avg"].values, "o", ms=3, color="C0", label=r"$\langle P_{|00\rangle}\rangle_N$")
        if "sinc_fit" in fr.data_vars:
            fit_vals = fr["sinc_fit"].values
            if np.any(np.isfinite(fit_vals)):
                ax_avg.plot(amps_scale, fit_vals, "-", lw=1.5, color="C3", label="sinc fit")
        if np.isfinite(opt_scale):
            ax_avg.axvline(opt_scale, color="lime", lw=1.5, label=f"opt = {opt_scale:.4f}")
        ax_avg.set_xlabel("Amplitude scale (a.u.)")
        ax_avg.set_ylabel(r"$\langle P_{|00\rangle}\rangle_N$")
        ax_avg.legend(loc="upper right", fontsize=8)

    used = set()
    for i in range(n_pairs):
        row, col = divmod(i, cols)
        used.add((2 * row, col))
        used.add((2 * row + 1, col))
    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            if (r, c) not in used:
                axes[r, c].axis("off")

    fig.suptitle("JAZZ2-N CZ amplitude calibration")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig

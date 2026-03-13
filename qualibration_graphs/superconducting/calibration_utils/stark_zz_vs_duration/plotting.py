from typing import List, Dict, Optional
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.analysis import oscillation_decay_exp
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: List[AnyTransmonPair],
    fits: xr.Dataset,
) -> List[Figure]:
    """
    Plot raw Ramsey/echo-like traces for Control (left col) and Target (right col),
    and overlay the fitted curve ONLY on the Target axes.

    For each qubit-pair:
      rows -> control_state in {0,1}
      cols -> Control (C), Target (T)

    The observable is 'state' if present, otherwise 'I'.
    """
    figs: List[Figure] = []
    val = "state" if "state" in ds.data_vars else "I"

    for qp in qubit_pairs:
        qc, qt = qp.qubit_control, qp.qubit_target

        # Slice once per pair
        ds_pair = ds.sel(qubit_pair=qp.name)
        ts_ns = ds_pair.idle_time.data  # time in ns
        data_c = ds_pair.sel(control_target="c")
        data_t = ds_pair.sel(control_target="t")

        # Fit slice for this pair/target (may be absent or partially missing)
        fit_t: Optional[xr.Dataset]
        try:
            fit_t = fits.sel(qubit_pair=qp.name, control_target="t")
        except Exception:
            fit_t = None

        fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True)

        for row, st in enumerate((0, 1)):
            # raw data
            y_c = data_c.sel(control_state=st)[val].data
            y_t = data_t.sel(control_state=st)[val].data

            axes[row, 0].plot(ts_ns, y_c, lw=1, label=f"raw (qc={st})")
            axes[row, 1].plot(ts_ns, y_t, lw=1, label=f"raw (qc={st})")

            # overlay fitted curve for the TARGET at this control_state
            if fit_t is not None:
                _plot_fit_on_axis(
                    ax=axes[row, 1],
                    t_ns=ts_ns,
                    fit_slice=fit_t.sel(control_state=st, drop=True),
                    label=f"fit (qc={st})",
                )

            # y-labels on left
            axes[row, 0].set_ylabel(f"{val} (qc={st})")

        # Column titles (top row)
        axes[0, 0].set_title(f"Qc: {qc.name}")
        axes[0, 1].set_title(f"Qt: {qt.name}")

        # Bottom x-labels
        axes[-1, 0].set_xlabel("time [ns]")
        axes[-1, 1].set_xlabel("time [ns]")

        # Legends only on the Target column
        axes[0, 1].legend(loc="best", fontsize=8)
        axes[1, 1].legend(loc="best", fontsize=8)

        fig.tight_layout()
        figs.append(fig)

    return figs


def _plot_fit_on_axis(
    ax: Axes, t_ns: np.ndarray, fit_slice: xr.Dataset, label: str = "fit"
) -> None:
    """
    Evaluate and plot the decaying-oscillation fit for one control_state on the given axis.

    Expects `fit_slice.fit` to contain parameters a, f, phi, offset, decay under `fit_vals` dim.

    - t_ns: times in ns
    - f: assumed to be in MHz in the stored fit (common convention in your pipeline).
         Change to 1e6 if your fitter returns Hz.
    """
    if "fit" not in fit_slice:
        return

    # Pull scalars safely; fall back to None if missing
    def _get(name: str) -> Optional[float]:
        try:
            return float(fit_slice.fit.sel(fit_vals=name).values)
        except Exception:
            return None

    a = _get("a")
    f = _get("f")  # MHz (per your existing code)
    phi = _get("phi")
    offset = _get("offset")
    decay = _get("decay")  # 1/ns (your echo code uses -1/decay for T2)

    # If any parameter is missing, skip plotting the fit
    if None in (a, f, phi, offset, decay):
        return

    # Build fitted curve. `oscillation_decay_exp` is assumed to accept t (ns) and scalar params.
    # If it expects SI units, convert here; currently we keep your convention:
    #   f in MHz, t in ns, decay in 1/ns, consistent with earlier code.
    t_ns = np.linspace(t_ns[0], t_ns[-1], 100)
    y_fit = oscillation_decay_exp(
        xr.DataArray(t_ns),
        xr.DataArray(a),
        xr.DataArray(f),  # MHz
        xr.DataArray(phi),
        xr.DataArray(offset),
        xr.DataArray(decay),  # 1/ns
    )

    ax.plot(t_ns, np.asarray(y_fit.data), lw=1.2, label=label)

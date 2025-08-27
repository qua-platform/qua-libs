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
    figss: List[List[Figure]] = []
    val = "state" if "state" in ds.data_vars else "I"

    for qp in qubit_pairs:
        qc, qt = qp.qubit_control, qp.qubit_target

        # Slice once per pair
        ds_pair = ds.sel(qubit_pair=qp.name)
        ts_ns = ds_pair.idle_time.data  # time in ns
        rel_phases = ds_pair.relative_phase.data  # in 2pi
        data_c = ds_pair.sel(control_target="c")
        data_t = ds_pair.sel(control_target="t")

        # Fit slice for this pair/target (may be absent or partially missing)
        fit_t: Optional[xr.Dataset]
        try:
            fit_t = fits.sel(qubit_pair=qp.name, control_target="t")
        except Exception:
            fit_t = None

        figs: List[Figure] = []
        for ph in rel_phases:
            fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
            fig.suptitle(f"Qc: {qc.name}, Qt: {qt.name} @ {ph:6.5f}")
            for row, st in enumerate((0, 1)):
                # raw data
                y_c = data_c.sel(control_state=st).sel(relative_phase=ph, method="nearest")[val].data
                y_t = data_t.sel(control_state=st).sel(relative_phase=ph, method="nearest")[val].data

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
        figss.append(figs)

    return figss


def _plot_fit_on_axis(ax: Axes, t_ns: np.ndarray, fit_slice: xr.Dataset, label: str = "fit") -> None:
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
    y_fit = oscillation_decay_exp(
        xr.DataArray(t_ns),
        xr.DataArray(a),
        xr.DataArray(f),  # MHz
        xr.DataArray(phi),
        xr.DataArray(offset),
        xr.DataArray(decay),  # 1/ns
    )

    ax.plot(t_ns, np.asarray(y_fit.data), lw=1.2, label=label)


def plot_fit_summary(
    ds: xr.Dataset,
    qubit_pairs: List[AnyTransmonPair],
    ds_fit: xr.Dataset,
) -> List[plt.Figure]:
    """
    For each qubit pair, create a 1x2 figure:
      (0) relative phase vs freq offset for control_state = 0 and 1
      (1) relative phase vs zz_coeff

    Uses variables prepared in ds_fit:
      - freq_offset(qubit_pair, control_target, control_state, relative_phase) [MHz]
      - zz_coeff(qubit_pair, relative_phase) [MHz if you convert, else same as freq_offset units]
      - best_phase(qubit_pair)
      - best_zz_coeff(qubit_pair)
    """
    figs: List[plt.Figure] = []

    for qp in qubit_pairs:
        qc, qt = qp.qubit_control, qp.qubit_target

        # --- Pull data for this pair
        # freq offset for control states
        f0 = ds_fit.freq_offset.sel(qubit_pair=qp.name, control_target="t", control_state=0)
        f1 = ds_fit.freq_offset.sel(qubit_pair=qp.name, control_target="t", control_state=1)
        # ensure we're 1D over relative_phase
        f0 = f0.squeeze(drop=True)
        f1 = f1.squeeze(drop=True)

        # Slice once per pair
        ds_pair = ds.sel(qubit_pair=qp.name)
        ts_ns = ds_pair.idle_time.data  # time in ns
        rel_phases = ds_pair.relative_phase.data  # in 2pi
        data_t = ds_pair.sel(control_target="t")
        val = "state" if "state" in ds.data_vars else "I"

        # zz coefficient
        zz = ds_fit.zz_coeff.sel(qubit_pair=qp.name).squeeze(drop=True)

        # phases and bests
        phases = f0["relative_phase"].values  # same coord across arrays
        best_ph = float(ds_fit.best_phase.sel(qubit_pair=qp.name).values)
        best_zz = float(ds_fit.best_zz_coeff.sel(qubit_pair=qp.name).values)

        # --- Make figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
        ax_c0, ax_f, ax_c1, ax_zz = axes.ravel()
        fig.suptitle(f"Qc: {qc.name}, Qt: {qt.name}")

        ax_c0.pcolor(ts_ns, rel_phases, data_t.sel(control_state=0)[val].data, shading="auto")
        ax_c0.set_aspect("auto")
        ax_c0.set_xlabel("Idle times (ns)")
        ax_c0.set_ylabel("Relative phase (×2π)")
        ax_c0.set_title(f"control_state = 0")

        ax_c1.pcolor(ts_ns, rel_phases, data_t.sel(control_state=1)[val].data, shading="auto")
        ax_c1.set_aspect("auto")
        ax_c1.set_xlabel("Idle times (ns)")
        ax_c1.set_ylabel("Relative phase (×2π)")
        ax_c1.set_title(f"control_state = 1")

        # Panel 1: freq offset vs phase, both control states
        ax_f.plot(phases, f0.values, lw=1.8, label="qc=0")
        ax_f.plot(phases, f1.values, lw=1.8, label="qc=1")
        ax_f.axvline(best_ph, ls="--", alpha=0.6)
        ax_f.set_xlabel("Relative phase (×2π)")
        ax_f.set_ylabel("Freq offset [MHz]")
        ax_f.set_title(f"Extracted frequency")
        ax_f.legend(frameon=False, fontsize=9)
        ax_f.set_aspect("auto")

        # Panel 3: zz_coeff vs phase
        ax_zz.plot(zz["relative_phase"].values, zz.values, lw=1.8)
        ax_zz.axvline(best_ph, ls="--", alpha=0.6)
        ax_zz.annotate(
            f"best={best_zz:.3f}",
            xy=(
                best_ph,
                np.interp(
                    best_ph,
                    zz["relative_phase"].values,
                    np.interp(
                        np.argsort(zz["relative_phase"].values), np.argsort(zz["relative_phase"].values), zz.values
                    ),
                ),
            ),  # safe-ish placement; will move if NaNs
            xytext=(10, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
            arrowprops=dict(arrowstyle="->", lw=1, alpha=0.6),
            fontsize=9,
            annotation_clip=True,
        )
        ax_zz.set_xlabel("Relative phase (×2π)")
        ax_zz.set_ylabel("ZZ coeff [MHz]")
        ax_zz.set_title("ZZ coefficient")
        ax_zz.set_aspect("auto")

        # shared x range if phases exist
        if phases.size:
            x_min = float(np.nanmin(phases))
            x_max = float(np.nanmax(phases))
            ax_f.set_xlim(x_min, x_max)
            ax_zz.set_xlim(x_min, x_max)

        fig.tight_layout()
        figs.append(fig)

    return figs

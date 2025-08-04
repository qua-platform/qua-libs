import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import List, Dict
from qualibrate import QualibrationNode
from calibration_utils.swap_amp_calibration.analysis import FitParameters
from quam_builder.architecture.superconducting.qubit import AnyQubitPair  # TODO: FIX THIS


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: List[AnyQubitPair],
    ds_fit: xr.Dataset,
    fit_results: Dict[str, Dict],
    node: QualibrationNode,
) -> plt.Figure:
    """
    Plot raw data (control and target) and overlay fits or optimal amplitude lines.
    """
    n = len(qubit_pairs)
    fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(15, 4 * n))

    if n == 1:
        axs = np.array([[axs[0], axs[1]]])

    for i, qp in enumerate(qubit_pairs):
        name = qp.name
        da_ctrl = ds.data_var_control.sel(qubit=name)
        da_trgt = ds.data_var_target.sel(qubit=name)

        amp_mV = ds.amp_full.sel(qubit=name) * 1e3

        for j, (da, label) in enumerate([(da_ctrl, "Control"), (da_trgt, "Target")]):
            ax = axs[i][j]
            if node.parameters.max_number_pulses_per_sweep == 1:
                da.plot(ax=ax, x="amplitude")
                if "fit_amp_control" in ds_fit[name]:
                    ds_fit[name]["fit_amp_control"].plot(ax=ax, x="amplitude", ls="--", color="r")
            else:
                da.plot(ax=ax, x="amplitude", y="N_pi_vec")
                amp_opt = fit_results[name]["optimal_amplitude"]
                ax.axvline(x=amp_opt * 1e3, ls="--", lw=0.8, color="r")

            ax.set_title(f"{name} - {label}")
            ax.set_xlabel("Amplitude [mV]" if node.parameters.max_number_pulses_per_sweep == 1 else "Amplitude [arb.]")
            if node.parameters.max_number_pulses_per_sweep > 1:
                ax.set_ylabel("Number of pulses")

    fig.tight_layout()
    fig.suptitle("SWAP Oscillations - Control and Target", y=1.02)
    return fig

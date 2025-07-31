import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import List, Dict
from qualibrate import QualibrationNode
from calibration_utils.swap_amp_calibration.analysis import FitParameters
from quam_builder.architecture.superconducting.qubit import AnyQubitPair #TODO: FIX THIS


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: List[AnyQubitPair],
    ds_fit: xr.Dataset,
    fit_results: Dict[str, Dict],
    node: QualibrationNode
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot raw data (control and target) with fits or optimal amplitude lines using QubitPairGrid.
    Returns both control and target figures.
    """
    from qualibration.helpers.plotting_tools import grid_pair_names, grid_iter, QubitPairGrid #TODO: FIX THIS

    ds = ds.assign_coords(amp_mV=ds.amp_full * 1e3)

    # === Control Data Plot ===
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)

    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        if node.parameters.max_number_pulses_per_sweep == 1:
            ds.loc[qubit].data_var_control.plot(ax=ax, x="amp_mV")
            if "fit_amp_control" in ds_fit[name]:
                (ds_fit[name].fit_amp_control * 1e3).plot(ax=ax, x="amp_mV", ls="--", color="r")
        else:
            ds.loc[qubit].data_var_control.plot(ax=ax, x="amp_mV", y="N_pi_vec")
            amp = 1e3 * node.results["results"][name]["SWAP_amplitude"]
            ax.axvline(amp, color="r", lw=0.5, ls='--')
            ax.set_ylabel("num. of pulses")

        ax.set_xlabel("Amplitude [mV]")
        ax.set_title(name)

    grid.fig.suptitle("SWAP amplitude calibration, control")
    plt.tight_layout()
    plt.show()
    node.results["figure_control"] = grid.fig

    # === Target Data Plot ===
    grid = QubitPairGrid(grid_names, qubit_pair_names)

    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        if node.parameters.max_number_pulses_per_sweep == 1:
            ds.loc[qubit].data_var_target.plot(ax=ax, x="amp_mV")
            if "fit_amp_target" in ds_fit[name]:
                (ds_fit[name].fit_amp_target * 1e3).plot(ax=ax, x="amp_mV", ls="--", color="r")
        else:
            ds.loc[qubit].data_var_target.plot(ax=ax, x="amp_mV", y="N_pi_vec")
            amp = 1e3 * node.results["results"][name]["SWAP_amplitude"]
            ax.axvline(amp, color="r", lw=0.5, ls='--')
            ax.set_ylabel("num. of pulses")

        ax.set_xlabel("Amplitude [mV]")
        ax.set_title(name)

    grid.fig.suptitle("SWAP amplitude calibration, target")
    plt.tight_layout()
    plt.show()
    node.results["figure_target"] = grid.fig

    return node.results["figure_control"], node.results["figure_target"]


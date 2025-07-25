from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
# from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair
from calibration_utils.cr_utils import *

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubit_pairs: List[AnyTransmonPair], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """

    figs = []
    for qp in qubit_pairs:
        qc = qp.qubit_control
        qt = qp.qubit_target
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        fig.suptitle(f"Qc: {qc.name}, Qt: {qt.name}")

        # Prepare the figure for live plotting
        ds_sliced = ds.sel(qubit_pair=qp.name, control_target="c")
        corr_phase = ds_sliced.coords["corr_phase"].values
        # plotting data
        ax.plot(corr_phase, ds_sliced.sel(control_state=0)["state"].data)
        ax.plot(corr_phase, ds_sliced.sel(control_state=1)["state"].data)
        ax.set_xlabel("correction phase on Qc [2pi]")
        ax.set_ylabel("control state")
        ax.legend(["Qc=0", "Qc=1"])
        plt.tight_layout()
        figs.append(fig)
    return figs

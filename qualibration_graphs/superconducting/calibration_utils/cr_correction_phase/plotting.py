from typing import List, Literal
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
        corr_qubits = ds_pair.correction_qubit.data
        corr_phases = ds_pair.correction_phase.data
        control_targets = ds_pair.control_target.data
        control_states = ds_pair.control_state.data

        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

        for row, tt in enumerate(corr_qubits):

            for col, ct in enumerate(control_targets):
                ax = axes[row, col]

                for st in control_states:
                    y = ds_pair.sel(correction_qubit=tt, control_target=ct, control_state=st)[val].data
                    ax.plot(corr_phases, y, label=f"q{ct}=|{st}>", marker="o", linestyle="-")
                ax.set_ylim([-0.05, 1.05])
                # y-labels on left
                ax.set_ylabel("<Z>") if col == 0 else None
                # titles (top row)
                ax.set_title(f"Q{ct}: {qc.name}") if row == 0 and ct == "c" else None
                ax.set_title(f"Q{ct}: {qt.name}") if row == 0 and ct == "t" else None
                # Bottom x-labels
                ax.set_xlabel(f"{tt} Correction Phase [2pi]")
                ax.legend()

        # fig.tight_layout()
        figs.append(fig)

    return figs










# def plot_raw_data_with_fit(ds: xr.Dataset, qubit_pairs: List[AnyTransmonPair], fits: xr.Dataset):
#     """
#     Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

#     Parameters
#     ----------
#     ds : xr.Dataset
#         The dataset containing the quadrature data.
#     qubits : list of AnyTransmon
#         A list of qubits to plot.
#     fits : xr.Dataset
#         The dataset containing the fit parameters.

#     Returns
#     -------
#     Figure
#         The matplotlib figure object containing the plots.

#     Notes
#     -----
#     - The function creates a grid of subplots, one for each qubit.
#     - Each subplot contains the raw data and the fitted curve.
#     """

#     figs = []
#     for qp in qubit_pairs:
#         qc = qp.qubit_control
#         qt = qp.qubit_target
#         fig, ax = plt.subplots(1, 1, figsize=(7, 6))
#         fig.suptitle(f"Qc: {qc.name}, Qt: {qt.name}")

#         # Prepare the figure for live plotting
#         ds_sliced = ds.sel(qubit_pair=qp.name, control_target="c")
#         correction_phase = ds_sliced.coords["correction_phase"].values
#         # plotting data
#         ax.plot(correction_phase, ds_sliced.sel(control_state=0)["state"].data)
#         ax.plot(correction_phase, ds_sliced.sel(control_state=1)["state"].data)
#         ax.set_xlabel("correction phase on Qc [2pi]")
#         ax.set_ylabel("control state")
#         ax.legend(["Qc=0", "Qc=1"])
#         plt.tight_layout()
#         figs.append(fig)
#     return figs

from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair
# from cr_utils import 


u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(node, ds: xr.Dataset, qubit_pairs: List[AnyTransmonPair], fits: xr.Dataset):
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
        fig, axss = plt.subplots(3, 2, figsize=(8, 8), sharex=True)
        # Plots
        plt.suptitle(f"CR Time Rabi")
        for i, (axs, bss) in enumerate(zip(axss, ["X", "Y", "Z"])):
            for j, stc in enumerate(["g", "e"]):
                ds_sliced = ds.sel(qubit_pair=qp.name, qst_basis=i, control_state=j)
                if node.parameters.use_state_discrimination:
                    axs[0].plot(ds_sliced.pulse_duration.data, ds_sliced.sel(control_target="c")["state"].data, label=[f"qc=|{stc}>"])
                    axs[1].plot(ds_sliced.pulse_duration.data, ds_sliced.sel(control_target="t")["state"].data, label=[f"qt=|{stc}>"])
                    axs[0].set_ylabel(f"State in <{bss}>")
                else:
                    axs[0].plot(ds_sliced.pulse_duration.data, ds_sliced.sel(control_target="c")["I"].data, label=[f"qc=|{stc}>"])
                    axs[1].plot(ds_sliced.pulse_duration.data, ds_sliced.sel(control_target="t")["I"].data, label=[f"qt=|{stc}>"])
                    axs[0].set_ylabel(f"rotated 'I' quadrature <{bss}>")
                axs[0].set_title(f"control: {qp.qubit_control.name}") if i == 0 else None
                axs[1].set_title(f"target: {qp.qubit_target.name}") if i == 0 else None
                for ax in axs:
                    ax.set_xlabel("cr durations [ns]") if i == 2 else None
                    ax.legend([f"{qc.name}=g", f"{qc.name}=e"])
        plt.tight_layout()
        figs.append(fig)
    return figs

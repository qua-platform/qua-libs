from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the raw data and optimal operation (power and frequency) point for the TWPA pump.

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
    data_dsnr = "snr_delta_db" if "snr_delta_db" in ds else None
    data_gain = "gain_db" if "gain_db" in ds else None

    to_plot = [d for d in (data_dsnr, data_gain) if d is not None]
    ncols = len(to_plot)

    if ncols == 0:
        raise RuntimeError("The dataset must contain either 'snr_delta_db' or 'gain_db'.")

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharex=True, sharey=True)

    # Ensure axes is iterable
    if ncols == 1:
        axes = [axes]

    for ax, var in zip(axes, to_plot):
        (ds.assign_coords(full_pump_freq_GHz=ds.full_pump_freq / u.GHz)[var]).plot(
            ax=ax, y="pump_power_dBm", x="full_pump_freq_GHz", add_colorbar=True
        )
        ax.set_title("ΔSNR [dB]" if var == "snr_delta_db" else "Gain [dB]")
        ax.set_xlabel("Pump tone frequency [GHz]")
        ax.set_ylabel("Pump tone power [dBm]")

    fig.tight_layout()
    fig.suptitle("TWPA Pump Calibration")
    return fig

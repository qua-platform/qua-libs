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
    if hasattr(ds, "snr_delta_db"):
        data_dsnr = "snr_delta_db"
    elif hasattr(ds, "gain_db"):
        data_gain = "gain_db"
    else:
        raise RuntimeError(
            "The dataset must contain either 'snr_delta_db' or 'gain_db' for the plotting function to work."
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    # Plot snr_delta_db
    (ds.assign_coords(pump_frequency_GHz=ds.pump_frequency / u.GHz)[data_dsnr]).plot(
        ax=axes[0], y="pump_amp", x="pump_frequency", add_colorbar=True
    )
    axes[0].set_title("ΔSNR [dB]")

    # Plot gain_db
    (ds.assign_coords(pump_frequency_GHz=ds.pump_frequency / u.GHz)[data_gain]).plot(
        ax=axes[0], y="pump_amp", x="pump_frequency", add_colorbar=True
    )
    axes[1].set_title("Gain [dB]")

    for ax in axes:
        ax.set_xlabel("Pump frequency [GHz]")
        ax.set_ylabel("Pump amplitude [dBm]")

    fig.tight_layout()
    fig.suptitle("TWPA Pump Calibration")
    return fig

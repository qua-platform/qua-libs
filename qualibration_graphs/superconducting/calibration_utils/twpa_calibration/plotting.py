from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from qualibrate import QualibrationNode

u = unit(coerce_to_integer=True)

def plot_raw_data_with_fit(ds: xr.Dataset, node:QualibrationNode):
    """
    Plot raw TWPA calibration data (ΔSNR, Gain), mark max gain point,
    and show qubits attached to each TWPA in the titles.
    """
    to_plot = [v for v in ("snr_delta_db", "gain_db") if v in ds]
    ncols = len(to_plot)
    if ncols == 0:
        raise RuntimeError("Dataset must contain 'snr_delta_db' or 'gain_db'.")

    twpas = list(ds.coords["twpa"].values)
    nrows = len(twpas)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True
    )

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for i, twpa in enumerate(twpas):
        qubits = node.namespace["twpa_group"].get(str(twpa), [])   # grab list of qubits
        for j, var in enumerate(to_plot):
            ax = axes[i][j]

            data_to_plot = ds[var].sel(twpa=twpa).isel(pump_amp=slice(1, None))
            data_to_plot = data_to_plot.assign_coords(
                full_pump_freq_GHz=("pump_frequency", ds.full_pump_freq.sel(twpa=twpa).values / 1e9)
            )

            data_to_plot.plot(
                ax=ax, y="pump_power_dBm", x="full_pump_freq_GHz", add_colorbar=True
            )

            if var == "gain_db" and "gain_max_freq" in ds and "gain_max_power" in ds:
                opt_freq = ds["gain_max_freq"].sel(twpa=twpa).item() / 1e9
                opt_power = ds["gain_max_power"].sel(twpa=twpa).item()
                ax.scatter(opt_freq, opt_power, s=150, c="red", edgecolors="black", zorder=10)

            label = "ΔSNR [dB]" if var == "snr_delta_db" else "Gain [dB]"
            # 👉 TWPA name + qubit list in title
            ax.set_title(f"{twpa} ({', '.join(qubits)}): {label}")

            ax.set_xlabel("Pump frequency [GHz]")
            ax.set_ylabel("Pump power [dBm]")

    fig.tight_layout()
    fig.suptitle("TWPA Pump Calibration", y=1.02)
    return fig

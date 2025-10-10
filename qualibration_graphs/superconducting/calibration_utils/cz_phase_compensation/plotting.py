import matplotlib.pyplot as plt
import xarray as xr
from qualibrate import QualibrationNode
from quam_config import Quam

from .parameters import Parameters


def plot_raw_data_with_fit(ds_raw: xr.Dataset, qubit_pairs: Quam, ds_fit: xr.Dataset = None):
    """
    Plot the raw data with the fit for each qubit pair in a single figure.
    """
    n_pairs = len(qubit_pairs)

    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4))
    if n_pairs == 1:
        axes = [axes]

    for i, qp_name in enumerate(qubit_pairs):
        ax = axes[i]

        # Select data for this qubit pair
        qp_data = ds_raw.sel(qubit_pair=qp_name.name)

        # Plot raw data
        if "state_control" in ds_raw.data_vars:
            qp_data.state_control.plot(ax=ax, marker="o", linestyle="", color="blue", label="Control")
            qp_data.state_target.plot(ax=ax, marker="o", linestyle="", color="red", label="Target")
        else:
            qp_data.I_control.sel(control_target="c").plot(
                ax=ax, marker="o", linestyle="", color="blue", label="Control"
            )
            qp_data.I_target.sel(control_target="t").plot(ax=ax, marker="o", linestyle="", color="red", label="Target")

        # Plot fitted data if available and fit was successful
        if ds_fit is not None:
            qp_fit_data = ds_fit.sel(qubit_pair=qp_name.name)
            if qp_fit_data.success.values:
                if "fitted_control" in ds_fit.data_vars:
                    qp_fit_data.fitted_control.plot(ax=ax, color="blue", alpha=0.5)
                if "fitted_target" in ds_fit.data_vars:
                    qp_fit_data.fitted_target.plot(ax=ax, color="red", alpha=0.5)
        if "state_control" in ds_raw.data_vars:
            ax.set_ylabel("Measured State")
        else:
            ax.set_ylabel("Rotated I Quadrature [V]")

        ax.set_xlabel(r"x90 frame rotation [$\mathrm{rad}/2\pi$]")
        ax.legend()

    plt.tight_layout()
    return fig

from typing import Literal, Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.cz_iswap_flux_bootstrap.parameters import moving_qubit, stationary_qubit
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: list,
    fits: xr.Dataset,
    qubit_role: Literal["stationary", "moving"] = "stationary",
) -> Figure:
    """Plot the chevron 2D map for every qubit pair on a chip-topology grid.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset (must contain ``amp_full``, ``detuning``, and either
        ``state_{qubit_role}`` or ``I_{qubit_role}``).
    qubit_pairs : list
        Qubit pair objects used for grid placement.
    fits : xr.Dataset
        Dataset containing fit parameters (``cz_len``, ``cz_amp``).
    qubit_role : {"stationary", "moving"}
        Which qubit's data to display. Defaults to ``"stationary"``.

    Returns
    -------
    Figure
        Matplotlib figure with one chevron panel per pair.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    qp_map = {qp.name: qp for qp in qubit_pairs}
    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        try:
            fit_data = fits.sel(qubit_pair=qp_name) if fits is not None else None
        except (KeyError, ValueError):
            fit_data = None
        plot_individual_qubit_chevron(ax, ds, qp_name, qubit_role, fit_data, qp_map.get(qp_name))

    grid.fig.suptitle(f"CZ Chevron — {qubit_role} qubit")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_qubit_chevron(
    ax: Axes,
    ds: xr.Dataset,
    qp_name: str,
    qubit_role: Literal["stationary", "moving"],
    fit: Optional[xr.Dataset] = None,
    qp=None,
):
    """Plot the chevron 2D map for one qubit pair and one qubit role.

    Parameters
    ----------
    ax : Axes
        Axis to draw on.
    ds : xr.Dataset
        Processed dataset with ``amp_full`` and ``detuning`` coordinates.
    qp_name : str
        Qubit pair name used to select data from ``ds``.
    qubit_role : {"stationary", "moving"}
        Which qubit to display.  ``"stationary"`` uses ``state_stationary`` /
        ``I_stationary``; ``"moving"`` uses ``state_moving`` / ``I_moving``.
    fit : xr.Dataset, optional
        Fit parameters for this pair.  When valid, the optimal CZ point
        (``cz_len``, ``cz_amp``) is marked with a red star.
    qp : qubit pair object, optional
        When provided, the moving and stationary qubit names are shown in
        the panel title.
    """
    state_var = f"state_{qubit_role}"
    iq_var = f"I_{qubit_role}"
    data = getattr(ds, state_var) if hasattr(ds, state_var) else getattr(ds, iq_var)
    data.sel(qubit_pair=qp_name).plot(y="amp_full", ax=ax)

    if fit is not None:
        try:
            cz_len = float(fit.cz_len)
            cz_amp = float(fit.cz_amp)
            if not (np.isnan(cz_len) or np.isnan(cz_amp)):
                ax.axvline(cz_len, color="red", ls="--", lw=1.2)
                ax.axhline(cz_amp, color="orange", ls="--", lw=1.2)
                ax.scatter(cz_len, cz_amp, color="red", marker="*", s=200, zorder=5, label="CZ point")
                ax.legend(fontsize=8)
        except (ValueError, TypeError, AttributeError, KeyError):
            pass

    if qp is not None:
        qb_name = moving_qubit(qp).name if qubit_role == "moving" else stationary_qubit(qp).name
        ax.set_title(f"{qp_name}\n{qubit_role}: {qb_name}")
    else:
        ax.set_title(qp_name)

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Amplitude [V]")

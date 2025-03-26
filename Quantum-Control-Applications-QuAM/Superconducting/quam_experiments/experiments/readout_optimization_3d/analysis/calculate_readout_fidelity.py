import numpy as np
import xarray as xr

from qualang_tools.analysis import two_state_discriminator


def _apply_discriminator(I_g, Q_g, I_e, Q_e) -> float:
    """Wrapper function to apply two_state_discriminator and return only fidelity."""
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(I_g, Q_g, I_e, Q_e, False, b_plot=False)

    return fidelity


def calculate_readout_fidelity(ds: xr.Dataset) -> xr.DataArray:
    """Perform two-state discrimination for every coordinate across all runs"""
    fidelity_da = xr.apply_ufunc(
        _apply_discriminator,
        ds.I_g,
        ds.Q_g,
        ds.I_e,
        ds.Q_e,
        input_core_dims=[["run"]] * 4,  # Collapse the 'run' dimension
        vectorize=True,  # Ensure broadcasting over remaining dimensions
    )

    return fidelity_da


def get_maximum_fidelity_per_qubit(ds: xr.Dataset):
    """Returns a dataset where each qubit retains its max fidelity point."""
    da = ds.fidelity  # Extract the fidelity data array

    max_points = []
    for qubit in ds.qubit:
        da_q = da.sel(qubit=qubit)  # Select fidelity for this qubit

        # Get index of maximum fidelity
        max_idx = np.unravel_index(da_q.argmax(), da_q.shape)
        max_coords = {dim: da_q.coords[dim][idx] for dim, idx in zip(da_q.dims, max_idx)}

        # Select the maximum fidelity point while keeping coordinates
        da_max = da_q.sel(**max_coords)

        # Add back the qubit dimension
        da_max = da_max.expand_dims("qubit").assign_coords(qubit=[qubit.item()])

        max_points.append(da_max)

    # Combine into a single dataset
    return xr.concat(max_points, dim="qubit").to_dataset(name="optimal_readout_point")

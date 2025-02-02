import xarray as xr

from qualang_tools.analysis import two_state_discriminator


def _apply_discriminator(I_g, Q_g, I_e, Q_e) -> float:
    """Wrapper function to apply two_state_discriminator and return only fidelity."""
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
        I_g, Q_g, I_e, Q_e, False, b_plot=False
    )

    return fidelity


def calculate_readout_fidelity(ds: xr.Dataset) -> xr.DataArray:
    """ Perform two-state discrimination for every coordinate across all runs"""
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
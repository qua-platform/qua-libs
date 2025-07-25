import xarray as xr


def reshape_control_target_val2dim(ds: xr.Dataset, state_discrimination: bool = False) -> xr.Dataset:
    """
    Transforms a dataset with variables I_c, Q_c, I_t, Q_t (or state_c/state_t)
    into a dataset with a new 'control_target' dimension ('c' or 't') and
    renamed variables ('I', 'Q') or 'state'.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing either:
        - ['I_c', 'Q_c', 'I_t', 'Q_t'] if state_discrimination is False, or
        - ['state_c', 'state_t'] if state_discrimination is True.
    state_discrimination : bool
        If True, convert ['state_c', 'state_t'] into a single 'state' variable.
        If False, convert ['I_c', 'Q_c', 'I_t', 'Q_t'] into 'I' and 'Q'.

    Returns
    -------
    xr.Dataset
        Reshaped dataset with a 'control_target' dimension (values 'c', 't'),
        and reordered dimensions: ('qubit_pair', 'control_target', ...)
    """
    control_target = ["c", "t"]
    target_dims = ("qubit_pair", "control_target")

    if state_discrimination:
        state = xr.concat(
            [ds["state_c"], ds["state_t"]],
            dim=xr.DataArray(control_target, dims="control_target", name="control_target"),
        ).transpose("qubit_pair", "control_target", ...)

        new_ds = xr.Dataset({"state": state})

    else:
        I = xr.concat(
            [ds["I_c"], ds["I_t"]], dim=xr.DataArray(control_target, dims="control_target", name="control_target")
        ).transpose("qubit_pair", "control_target", ...)

        Q = xr.concat(
            [ds["Q_c"], ds["Q_t"]], dim=xr.DataArray(control_target, dims="control_target", name="control_target")
        ).transpose("qubit_pair", "control_target", ...)

        new_ds = xr.Dataset({"I": I, "Q": Q})

    # Add missing coordinates from the original dataset (if not already present)
    for coord in ds.coords:
        if coord not in new_ds.coords:
            new_ds = new_ds.assign_coords({coord: ds.coords[coord]})

    # Make sure to order the dims
    new_ds = new_ds.transpose(*target_dims, ...)

    return new_ds

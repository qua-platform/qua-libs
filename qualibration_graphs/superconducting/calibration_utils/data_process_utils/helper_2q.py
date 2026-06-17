import xarray as xr
from typing import Iterable, List, Optional, Tuple


# def reshape_control_target_val2dim(
#     ds: xr.Dataset,
#     state_discrimination: bool = False,
#     control_target=["c", "t"],
#     control_target_dim_name="control_target",
# ) -> xr.Dataset:
#     """
#     Transforms a dataset with variables I_c, Q_c, I_t, Q_t (or state_c/state_t)
#     into a dataset with a new 'control_target' dimension ('c' or 't') and
#     renamed variables ('I', 'Q') or 'state'.

#     Parameters
#     ----------
#     ds : xr.Dataset
#         Input dataset containing either:
#         - ['I_c', 'Q_c', 'I_t', 'Q_t'] if state_discrimination is False, or
#         - ['state_c', 'state_t'] if state_discrimination is True.
#     state_discrimination : bool
#         If True, convert ['state_c', 'state_t'] into a single 'state' variable.
#         If False, convert ['I_c', 'Q_c', 'I_t', 'Q_t'] into 'I' and 'Q'.

#     Returns
#     -------
#     xr.Dataset
#         Reshaped dataset with a 'control_target' dimension (values 'c', 't'),
#         and reordered dimensions: ('qubit_pair', 'control_target', ...)
#     """
#     c_, t_ = control_target
#     target_dims = ("qubit_pair", control_target_dim_name)

#     if state_discrimination:
#         state = xr.concat(
#             [ds[f"state_{c_}"], ds[f"state_{t_}"]],
#             dim=xr.DataArray(control_target, dims=control_target_dim_name, name=control_target_dim_name),
#         ).transpose("qubit_pair", control_target_dim_name, ...)

#         new_ds = xr.Dataset({"state": state})

#     else:
#         I = xr.concat(
#             [ds[f"I_{c_}"], ds[f"I_{t_}"]], dim=xr.DataArray(control_target, dims=control_target_dim_name, name=control_target_dim_name)
#         ).transpose("qubit_pair", control_target_dim_name, ...)

#         Q = xr.concat(
#             [ds[f"Q_{c_}"], ds[f"Q_{t_}"]], dim=xr.DataArray(control_target, dims=control_target_dim_name, name=control_target_dim_name)
#         ).transpose("qubit_pair", control_target_dim_name, ...)

#         new_ds = xr.Dataset({"I": I, "Q": Q})

#     # Add missing coordinates from the original dataset (if not already present)
#     for coord in ds.coords:
#         if coord not in new_ds.coords:
#             new_ds = new_ds.assign_coords({coord: ds.coords[coord]})

#     # Make sure to order the dims
#     new_ds = new_ds.transpose(*target_dims, ...)

#     return new_ds



def reshape_control_target_val2dim(
    ds: xr.Dataset,
    state_discrimination: bool = False,
    control_target: List[str] = ["c", "t"],
    control_target_dim_name: str = "control_target",
    *,
    # New (optional) generalizations — all keyword-only to preserve compatibility:
    suffixes: Optional[Iterable[str]] = None,
    suffix_dim_name: Optional[str] = None,
    prefixes: Optional[Iterable[str]] = None,
    leading_dims: Optional[Tuple[str, ...]] = None,
) -> xr.Dataset:
    """
    Generalized reshape: stack variables with suffixes (e.g., I_c, Q_c, I_t, Q_t)
    into a new suffix dimension, and rename to base names (I, Q, ...).

    Backward compatible defaults:
      - suffixes          -> control_target
      - suffix_dim_name   -> control_target_dim_name
      - prefixes          -> ['I','Q'] unless state_discrimination=True (then ['state'])
      - leading_dims      -> ('qubit_pair', suffix_dim_name) if 'qubit_pair' exists;
                             otherwise ('qubit', suffix_dim_name) if 'qubit' exists;
                             otherwise (suffix_dim_name,) only.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing variables like '<prefix>_<suffix>'.
    state_discrimination : bool
        If True, expects only 'state_<suffix>' variables and outputs a single 'state' var.
    control_target : list[str]
        (Legacy) The pair of suffixes to use; kept for backward compatibility.
    control_target_dim_name : str
        (Legacy) Name of the new suffix dimension; kept for backward compatibility.
    suffixes : Iterable[str], optional
        Override the list of suffixes to gather (e.g., ['c','t'] and ['d','p']).
    suffix_dim_name : str, optional
        Override the suffix dimension name (e.g., 'qubit_pair_role' or 'qubit').
    prefixes : Iterable[str], optional
        Base variable names to gather (e.g., ['I','Q','Amp']). If
        state_discrimination=True, defaults to ['state']; else ['I','Q'].
    leading_dims : tuple[str, ...], optional
        Reorder dimensions so these appear first, followed by remaining dims.
        Typical choices: ('qubit_pair', suffix_dim_name) or ('qubit', suffix_dim_name).

    Returns
    -------
    xr.Dataset
        Dataset with a new suffix dimension, variables renamed to base prefixes,
        and dimensions reordered with the requested leading dims.
    """
    # Resolve legacy/default parameters (keeps old behavior by default)
    suffixes = list(suffixes) if suffixes is not None else list(control_target)
    suffix_dim_name = suffix_dim_name or control_target_dim_name

    if prefixes is None:
        prefixes = ["state"] if state_discrimination else ["I", "Q"]
    else:
        prefixes = list(prefixes)

    # Infer a sensible default leading dim if not specified
    if leading_dims is None:
        if "qubit_pair" in ds.dims:
            leading_dims = ("qubit_pair", suffix_dim_name)
        elif "qubit" in ds.dims:
            leading_dims = ("qubit", suffix_dim_name)
        else:
            leading_dims = (suffix_dim_name,)

    # Build the coordinate for the new suffix dimension
    suffix_coord = xr.DataArray(
        suffixes, dims=suffix_dim_name, name=suffix_dim_name
    )

    # Validate that required variables exist
    missing = []
    for p in prefixes:
        for s in suffixes:
            name = f"{p}_{s}"
            if name not in ds:
                missing.append(name)
    if missing:
        raise KeyError(
            "The following expected variables are missing from the dataset: "
            + ", ".join(missing)
        )

    # Concatenate for each prefix across the new suffix dimension
    data_vars = {}
    for p in prefixes:
        stacked = xr.concat(
            [ds[f"{p}_{s}"] for s in suffixes], dim=suffix_coord
        )
        # Just in case, transpose to move suffix dim next to qubit-like dim(s)
        # (final strict ordering handled below)
        data_vars[p] = stacked.transpose(...)

    new_ds = xr.Dataset(data_vars)

    # Carry over any coords that are present in ds but missing in new_ds
    for coord in ds.coords:
        if coord not in new_ds.coords:
            new_ds = new_ds.assign_coords({coord: ds.coords[coord]})

    # Final dimension ordering: requested leading dims first, then the rest
    # (preserving their relative order).
    def _ordered_dims(obj: xr.Dataset) -> List[str]:
        seen = set()
        out = []
        # Start with the dims that actually exist in the result
        existing_leads = [d for d in leading_dims if d in obj.dims]
        for d in existing_leads:
            if d not in seen:
                out.append(d)
                seen.add(d)
        for d in obj.dims:
            if d not in seen:
                out.append(d)
                seen.add(d)
        return out

    new_ds = new_ds.transpose(*_ordered_dims(new_ds), ...)

    return new_ds
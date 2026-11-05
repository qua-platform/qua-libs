"""Reading back the saved Standard RB run that an interleaved RB node compares against."""

from __future__ import annotations

import xarray as xr


def load_saved_ds_fit(load_id: int) -> xr.Dataset | None:
    """The ``ds_fit`` saved by node run ``load_id``, or None if that run saved none.

    Raises whatever qualibrate raises when the run cannot be found or read; the caller logs it and
    falls back to the reference alpha stored in the state.
    """
    from qualibrate.core.utils.node.content import read_node_data
    from qualibrate.core.utils.node.path_solver import get_node_dir_path
    from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path

    base_path = get_qualibrate_config(get_qualibrate_config_path()).storage.location
    node_dir = get_node_dir_path(int(load_id), base_path)
    data = read_node_data(node_dir, int(load_id), base_path)
    if data is None:
        return None
    return data.get("ds_fit")

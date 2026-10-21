"""Qualibrate node storage helpers for flux-distortion calibrations."""

from __future__ import annotations

from typing import Any


def read_node_data_dict(run_id: int) -> dict[str, Any]:
    """Load a Qualibrate node's saved results dict for ``run_id``.

    Resolves the configured Qualibrate storage location and reads the payload
    written by ``node.save()`` (typically includes ``ds_raw``, ``ds_fit``, …).

    Parameters
    ----------
    run_id :
        Qualibrate snapshot / run index.

    Returns
    -------
    dict
        Parsed node data as returned by ``qualibrate.core.utils.node.content.read_node_data``.
    """
    from qualibrate.core.utils.node.content import read_node_data
    from qualibrate.core.utils.node.path_solver import get_node_dir_path
    from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path

    base_path = get_qualibrate_config(get_qualibrate_config_path()).storage.location
    node_dir = get_node_dir_path(run_id, base_path)
    return read_node_data(node_dir, run_id, base_path)

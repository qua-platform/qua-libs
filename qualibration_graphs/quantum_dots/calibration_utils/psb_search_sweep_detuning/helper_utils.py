from __future__ import annotations

import xarray as xr
from typing import List


def assemble_ds_raw(dataset: xr.Dataset, pair_names: List[str]) -> xr.Dataset:
    """Convert fetched per-pair streams into the canonical 06a ``ds_raw`` layout."""
    i_arr = xr.concat([dataset[f"I_{pair_name}"] for pair_name in pair_names], dim="qubit_pair")
    q_arr = xr.concat([dataset[f"Q_{pair_name}"] for pair_name in pair_names], dim="qubit_pair")
    i_arr = i_arr.assign_coords(qubit_pair=pair_names)
    q_arr = q_arr.assign_coords(qubit_pair=pair_names)
    return xr.Dataset({"I": i_arr, "Q": q_arr})

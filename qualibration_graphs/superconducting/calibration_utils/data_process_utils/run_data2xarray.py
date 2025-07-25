import numpy as np
import xarray as xr
from typing import Dict, Tuple, Union, Literal, Optional, List


class CloudXarrayDataBuilder:
    def __init__(
        self,
        data_dict: dict,
        sweep_axes: Dict[str, xr.DataArray],
        outer_key: Literal["qubit", "qubit_pair"],
        fetch_names: Union[str, List[str]],
    ):
        """
        Class to convert cloud API results into an xarray.Dataset.

        Args:
            data_dict: Raw data dictionary from the cloud API.
            sweep_axes: Axes metadata as xr.DataArray.
            outer_key: Name of the outer axis ("qubit" or "qubit_pair").
            state_discrimination: Expect "state" instead of I/Q.
            adc_trace: Expect ADC trace data.
        """
        self.data_dict = dict(data_dict)
        self.sweep_axes = sweep_axes
        self.outer_key = outer_key
        self.fetch_names = fetch_names

        for key, da in self.sweep_axes.items():
            if da.dims and da.dims[0] != key:
                da = da.rename({da.dims[0]: key})
            self.sweep_axes[key] = da

    def _prepare_data_dict(self) -> dict:
        axis_values = self.sweep_axes[self.outer_key].data
        
        return {
            (qb_qp, fetch_name): np.array(self.data_dict[f"{fetch_name}{i + 1}"])
            for i, qb_qp in enumerate(axis_values)
            for fetch_name in self.fetch_names
        }

    def _init_data_vars(self, var_names: list, shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        """Initialize data arrays filled with NaNs."""
        return {name: np.full(shape, np.nan, dtype=np.float64) for name in var_names}

    def _insert_array(
        self,
        data_array: np.ndarray,
        key: Union[str, Tuple],
        array: Optional[np.ndarray],
        outer_labels: np.ndarray,
        all_axes: list,
        outer_axis: str,
    ):
        """Insert one array into the right slice of a full array."""
        if array is None:
            return

        coord_dict = self.sweep_axes
        coord_sizes = {k: v.size for k, v in coord_dict.items()}
        inner_axes = [ax for ax in all_axes if ax != outer_axis]
        expected_shape = tuple(coord_sizes[ax] for ax in inner_axes)

        if array.shape != expected_shape:
            raise ValueError(f"Shape mismatch for {key}: expected {expected_shape}, got {array.shape}")

        slices = [slice(None)] * len(all_axes)
        slices[0] = outer_labels.tolist().index(key)
        data_array[tuple(slices)] = array

    def _build_full_data(self, all_axes: list, shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        """Build the full data variables dictionary based on the prepared data."""
        outer_labels = self.sweep_axes[self.outer_key].values
        full_data = self._init_data_vars(self.fetch_names, shape)

        for q in outer_labels:
            for fetch_name in self.fetch_names:
                self._insert_array(
                    full_data[fetch_name],
                    q,
                    self._prepared_data.get((q, fetch_name)),
                    outer_labels,
                    all_axes,
                    self.outer_key,
                )

        return full_data

    def build_dataset(self) -> xr.Dataset:
        """Build and return an xarray.Dataset based on the initialized parameters."""

        self._prepared_data = self._prepare_data_dict()
        coord_dict = dict(self.sweep_axes)
        all_axes = list(coord_dict.keys())
        coord_sizes = {k: v.size for k, v in coord_dict.items()}
        full_shape = [coord_sizes[ax] for ax in all_axes]

        full_data = self._build_full_data(all_axes, tuple(full_shape))

        self._dataset = xr.Dataset(
            data_vars={k: (all_axes, v) for k, v in full_data.items()},
            coords=coord_dict,
        )
        return self._dataset

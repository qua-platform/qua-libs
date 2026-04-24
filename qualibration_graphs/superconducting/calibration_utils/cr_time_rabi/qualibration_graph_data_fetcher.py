import time
import logging
import re

import numpy as np
import xarray as xr

from typing import Any, Dict, List, Optional, Union
from qm.jobs.qm_job import QmJob

__all__ = ["XarrayDataFetcher"]

logger = logging.getLogger(__name__)


def timer_decorator(func):
    """
    Decorator to time the execution of a function and log the elapsed time.

    Args:
        func (callable): The function to be timed.

    Returns:
        callable: Wrapped function with timing.
    """

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        logger.debug(f"Function {func.__name__} started.")
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug(f"Function {func.__name__} finished in {elapsed:.3f} seconds.")
        return result

    return wrapper


class XarrayDataFetcher:
    """
    Class to fetch data using a QmJob and update a xarray.Dataset with the acquired data.
    """

    ignore_handles = [
        "readout",
        "readout_timestamps",
        "__qpu_execution_time_seconds",
        "__total_python_runtime_seconds",
    ]
    missing_data_value = 0  # np.nan

    def __init__(
        self,
        job: QmJob,
        axes: Optional[Dict[str, Union[xr.DataArray, np.ndarray]]] = None,
    ):
        """
        Initialize the data fetcher.

        Args:
            job (QmJob): A QmJob instance with result_handles for data acquisition.
            axes (Optional[Dict[str, xr.DataArray]]): Dictionary of coordinate axes.
                If None, no coordinates are used.
        """
        logger.debug("Initializing XarrayDataFetcher.")
        self.job = job
        # Make a copy of the axes so that they aren’t modified elsewhere.
        self.axes = self.preprocess_axes(axes)

        self._started_acquisition: bool = False
        self._finished_acquisition: bool = False
        self.t_start: Optional[float] = None

        # _raw_data now holds keys mapping to either np.ndarray or None.
        self._raw_data: Dict[str, Any] = {}

        self.dataset = self.initialize_dataset()
        self.data = {"dataset": self.dataset}
        logger.debug("XarrayDataFetcher initialized.")

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @staticmethod
    def preprocess_axes(
        axes: Optional[Dict[str, Union[xr.DataArray, np.ndarray]]],
    ) -> Optional[Dict[str, xr.DataArray]]:
        """
        Preprocess the axes dictionary to ensure all values are xr.DataArray instances.

        Args:
            axes (Optional[Dict[str, Union[xr.DataArray, np.ndarray]]]): Dictionary of coordinate axes.
        """
        if axes is None:
            return None
        if not isinstance(axes, dict):
            logger.warning("Axes must be a dictionary; ignoring axes.")
            return None

        new_axes = {}
        for key, val in axes.items():
            if isinstance(val, xr.DataArray):
                if val.dims == ("dim_0",):
                    val = val.rename({"dim_0": key})
                val.name = key
                new_axes[key] = val

            elif isinstance(val, np.ndarray):
                new_axes[key] = xr.DataArray(val, name=key, dims=(key,))
            else:
                logger.warning(
                    f"Axes must be a dictionary of xr.DataArray or np.ndarray instances; ignoring axis {key}."
                )
        return new_axes

    @timer_decorator
    def retrieve_latest_data(self):
        """
        Retrieve the latest data from the QmJob result handles.
        Skips handles listed in ignore_handles.
        """
        logger.debug("Starting to retrieve latest data from job result handles.")
        for data_label in self.job.result_handles.keys():
            if data_label in self.ignore_handles:
                logger.debug(f"Skipping ignored handle: {data_label}")
                continue

            logger.debug(f"Fetching data for handle: {data_label}")

            data_handle = self.job.result_handles.get(data_label)
            if data_handle is None or data_handle.count_so_far() == 0:
                self._raw_data[data_label] = None
            else:
                self._raw_data[data_label] = data_handle.fetch_all()
            logger.debug(
                f"Data fetched for {data_label}: shape {np.shape(self._raw_data[data_label])}"
            )

    def initialize_dataset(self):
        """
        Initialize a xarray.Dataset using the provided axes.

        Returns:
            xr.Dataset: An empty dataset with coordinates if axes is provided.
        """
        logger.debug("Initializing dataset with axes: {}".format(self.axes))
        return xr.Dataset(coords=self.axes)

    def update_dataset(self):
        """
        Update the xarray.Dataset with the latest raw data.
        This method uses the entire raw data dictionary (including None values)
        and delegates to the appropriate update function based on the available axes.

        Returns:
            xr.Dataset: The updated dataset.
        """
        logger.debug("Updating dataset with raw data.")

        # Add any non-array data to the data dictionary.
        for data_label, data_component in self._raw_data.items():
            if not isinstance(data_component, (np.ndarray, type(None))):
                self.data[data_label] = data_component

        raw_data_arrays = {}
        for label, array in self._raw_data.items():
            if isinstance(array, (np.ndarray, type(None))):
                raw_data_arrays[label] = array
            elif isinstance(array, list):
                raw_data_arrays[label] = np.array(array)
            else:
                continue

        if not raw_data_arrays:
            logger.debug("No raw data entries to update; returning current dataset.")
            return self.dataset

        # Case: no axes provided.
        if self.axes is None:
            logger.debug("No axes provided; updating dataset without coordinates.")
            self._update_no_axes_data_arrays(raw_data_arrays)
            return self.dataset

        dims_order = list(self.axes.keys())
        axes_shape = tuple(self.axes[dim].size for dim in dims_order)
        logger.debug(f"Axes shape: {axes_shape}")

        if dims_order[0] not in ["qubit", "qubit_pair"]:
            logger.error("first axis must be either qubit or qubit_pair")

        # Determine reference shape from non-None entries.
        data_arrays = [d for d in raw_data_arrays.values() if isinstance(d, np.ndarray)]
        if data_arrays:
            ref_shape = data_arrays[0].shape
            for d in data_arrays:
                if d.shape != ref_shape:
                    logger.error(
                        "Mismatch in shapes of raw data arrays: {}".format(
                            [d.shape for d in data_arrays]
                        )
                    )
                    raise ValueError("All arrays must have the same shape")
        else:
            # All entries are None; use axes shape as the reference.
            ref_shape = axes_shape

        logger.debug(f"Reference shape for raw data arrays: {ref_shape}")

        # Delegate to the correct update method.
        if axes_shape == ref_shape:
            logger.debug(
                "Axes shape matches raw data shape. Updating regular data arrays."
            )
            self._update_regular_data_arrays(raw_data_arrays, dims_order, axes_shape)
        elif len(axes_shape) == len(ref_shape) + 1 and axes_shape[1:] == ref_shape:
            logger.debug(
                "Axes shape has an extra dimension (qubit axis). Updating qubit data arrays."
            )
            self._update_qubit_data_arrays(raw_data_arrays, dims_order)
        else:
            logger.error(
                f"Axes and raw data arrays have incompatible shapes: axes_shape: {axes_shape}, ref_shape: {ref_shape}"
            )
            raise ValueError("Axes and arrays have incompatible shapes")

        logger.debug("Dataset update complete.")
        return self.dataset

    def _fill_missing_data(
        self, data: Optional[np.ndarray], shape: tuple
    ) -> np.ndarray:
        """
        Helper function to fill missing data.

        Args:
            data (Optional[np.ndarray]): The raw data (or None).
            shape (tuple): Desired shape if data is missing.

        Returns:
            np.ndarray: Original data if available; otherwise an array of NaNs.
        """
        if data is None:
            logger.debug(f"Data is None; filling with NaN array of shape {shape}.")
            return np.full(shape, self.missing_data_value)
        return data

    def _update_no_axes_data_arrays(self, raw_data: Dict[str, Any]):
        """
        Update the dataset by assigning each raw data entry without using any coordinates.
        For any entry that is None, a scalar NaN is used.

        Args:
            raw_data (Dict[str, Any]): Raw data entries (including None values).
        """
        logger.debug("Updating dataset without axes.")
        for label, data in raw_data.items():
            if data is None:
                logger.debug(
                    f"Data for variable '{label}' is None; filling with scalar NaN."
                )
                data = np.nan
            else:
                logger.debug(f"Updating variable '{label}' with shape {data.shape}.")
            self.dataset[label] = xr.DataArray(data)

    def _update_regular_data_arrays(
        self, raw_data: Dict[str, Any], dims_order: List[str], fill_shape: tuple
    ):
        """
        Update the dataset by directly assigning each raw data entry as a new variable.
        If a raw data entry is None, it is replaced with an array of NaNs with the given fill_shape.

        Args:
            raw_data (Dict[str, Any]): Raw data entries (including None values).
            dims_order (List[str]): Ordered list of dimension names.
            fill_shape (tuple): The expected shape of each raw data array.
        """
        logger.debug("Updating regular data arrays with coordinates.")
        for label, data in raw_data.items():
            filled_data = self._fill_missing_data(data, fill_shape)
            logger.debug(
                f"Updating variable '{label}' with dims {dims_order} and shape {filled_data.shape}."
            )
            self.dataset[label] = xr.DataArray(
                filled_data, dims=dims_order, coords=self.axes
            )

    def _update_qubit_data_arrays(
        self, raw_data: Dict[str, Any], dims_order: List[str]
    ):
        """
        Group raw data keys matching the pattern {label}{idx} and stack them along a new dimension.
        If a raw data entry is None, it is replaced with an array of NaNs.
        The shape for filling is determined from the sizes of the non-qubit dimensions (dims_order[1:]).

        Args:
            raw_data (Dict[str, Any]): Raw data entries (including None values).
            dims_order (List[str]): Ordered list of dimension names, where the first dimension
                                    represents the qubit axis.
        """
        logger.debug("Updating qubit data arrays by grouping and stacking.")

        grouped = {}
        non_qubit_shape = tuple(self.axes[dim].size for dim in dims_order[1:])
        for key, data in raw_data.items():
            if not isinstance(data, (np.ndarray, type(None))):
                continue
            m = re.match(r"([a-zA-Z_]+)(\d+)$", key)
            if m:
                base = m.group(1)
                idx = int(m.group(2))
                filled_data = self._fill_missing_data(data, non_qubit_shape)
                logger.debug(f"Grouping key '{key}': base '{base}', index {idx}")
                grouped.setdefault(base, []).append((idx, filled_data))
            else:
                filled_data = self._fill_missing_data(data, non_qubit_shape)
                logger.debug(
                    f"Key '{key}' does not match pattern; updating without qubit axis."
                )
                self.dataset[key] = xr.DataArray(
                    filled_data,
                    dims=dims_order[1:],
                    coords={dim: self.axes[dim] for dim in dims_order[1:]},
                )
        # Process each grouped variable by stacking the arrays along the qubit axis.
        for base, items in grouped.items():
            items.sort(key=lambda x: x[0])
            arrays = [item[1] for item in items]
            logger.debug(f"Stacking {len(arrays)} arrays for variable '{base}'.")
            stacked = np.stack(arrays, axis=0)
            self.dataset[base] = xr.DataArray(
                stacked, dims=[dims_order[0]] + dims_order[1:], coords=self.axes
            )

    @timer_decorator
    def is_processing(self):
        """
        Check if the job is still processing results.

        Returns:
            bool: True if the job is processing or just finished (one final yield), otherwise False.
        """
        logger.debug("Checking processing status of the job.")
        if not self._started_acquisition:
            logger.debug("Acquisition not started yet; marking as started.")
            self._started_acquisition = True
            self._finished_acquisition = False
            self.t_start = time.time()
            return True

        is_processing = self.job.result_handles.is_processing()
        logger.debug(f"Job is_processing status: {is_processing}")
        if is_processing:
            return True

        logger.debug("Job processing complete; returning False.")
        self._finished_acquisition = True
        return False

    def acquire_data(self):
        """
        Convenience method: retrieve data, update the dataset, and return the processing status.
        """
        self.retrieve_latest_data()
        self.update_dataset()
        return self.is_processing()

    def __iter__(self):
        """
        Make the XarrayDataFetcher iterable.
        Yields:
            xr.Dataset: Updated dataset each iteration until job processing is complete.
        """
        if not self._started_acquisition:
            self._started_acquisition = True
            self.t_start = time.time()

        # Continuously update and yield the dataset while the job is processing
        while self.job.result_handles.is_processing():
            self.retrieve_latest_data()
            self.update_dataset()
            yield self.dataset

        # Final update and yield after processing is complete
        self.retrieve_latest_data()
        self.update_dataset()
        yield self.dataset

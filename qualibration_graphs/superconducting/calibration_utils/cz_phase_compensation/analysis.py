import xarray as xr
from qualibrate import QualibrationNode
from quam_config import Quam

from .parameters import Parameters


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode[Parameters, Quam]):
    """
    Process the raw dataset to extract the relevant information.
    """
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode[Parameters, Quam]):
    """
    Fit the raw data to extract the relevant parameters.
    """
    return ds, {}


def log_fitted_results(fit_results: dict, log_callable: callable):
    """
    Log the fitted results.
    """
    pass

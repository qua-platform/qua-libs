import matplotlib.pyplot as plt
import xarray as xr
from qualibrate import QualibrationNode
from quam_config import Quam

from .parameters import Parameters


def plot_raw_data_with_fit(ds_raw: xr.Dataset, qubits: Quam, ds_fit: xr.Dataset = None):
    """
    Plot the raw data with the fit.
    """
    fig = plt.figure()
    return fig

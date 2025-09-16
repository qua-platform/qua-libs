# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict
from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.z_crosstalk_coupling_validation import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.data_process_utils import *
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# TODO: Implement this protocol

"""
Validates the effectiveness of Z crosstalk cancellation.

After applying the compensation determined from Z_crosstalk_detuning
(e.g., virtual-Z feed-forward or counter-drives), Ramsey experiments are
repeated on spectator qubits while driving the source qubit. Successful
cancellation is indicated when the residual Stark-induced detuning on the
probed qubits remains below the specified tolerance.
"""

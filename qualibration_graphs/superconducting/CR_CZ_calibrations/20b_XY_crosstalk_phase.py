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
from calibration_utils.xy_crosstalk_coupling_phase import (
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
Following the style of `XY_crosstalk_coupling_strength.py`, this protocol
characterizes the **phase** component of the microwave crosstalk matrix.

The experiment is a Ramsey-type sequence where the phase of the drive is
swept before the second π/2 pulse. By comparing the resulting phase response
to the reference case where the probed qubit is driven directly, the relative
crosstalk phase can be extracted.
"""
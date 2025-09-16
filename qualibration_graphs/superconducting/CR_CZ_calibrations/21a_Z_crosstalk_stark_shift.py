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
from calibration_utils.z_crosstalk_coupling_stark_shift import (
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
Characterizes the Stark-induced Z crosstalk between qubits.

The driven qubit (Qd) is pulsed at its own resonance frequency (or at the
off-resonant tone used in two-qubit gates), while Ramsey experiments are run
on a probed qubit (Qp). The resulting frequency shifts Δω_p are extracted as
a function of drive amplitude and phase. These measurements quantify the
Z-type crosstalk (dispersive Stark shifts and effective ZZ terms) and provide
the data needed to compute virtual-Z feed-forward or cancellation tones.
"""
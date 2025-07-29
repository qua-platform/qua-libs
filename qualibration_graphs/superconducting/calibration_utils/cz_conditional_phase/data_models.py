from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class FitResults:
    """Stores the relevant CZ conditional phase experiment fit parameters for a single qubit pair"""

    optimal_amplitude: float
    phase_diff: xr.DataArray
    fitted_curve: np.ndarray
    leakage: xr.DataArray
    success: bool

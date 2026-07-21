"""IQ analysis for PSB experiments that sweep readout integration time.

Uses the same PCA + two-Gaussian EM path as :mod:`calibration_utils.iq_sweep` (see
``06a_PSB_search_opx_sweep_detuning``); the sweep axis is whatever
``node.parameters.sweep_name`` names (default ``readout_length`` in ns).
"""

from __future__ import annotations

from typing import Dict, Tuple

import xarray as xr

from qualibrate.core import QualibrationNode

from calibration_utils.iq_sweep import fit_raw_data_pca_gaussian, log_fitted_results

__all__ = [
    "fit_measure_duration_raw_data",
    "log_fitted_results",
]


def fit_measure_duration_raw_data(
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict]:
    """Fit ``node.results['ds_raw']`` per qubit pair and per sweep point.

    Expects ``ds_raw`` with variables ``I``, ``Q`` and dimensions including
    ``qubit_pair``, ``n_runs``, and ``node.parameters.sweep_name``.

    Returns
    -------
    ds_fit, fit_results
        Same contract as :func:`calibration_utils.iq_sweep.fit_raw_data_pca_gaussian`.
    """
    return fit_raw_data_pca_gaussian(node.results["ds_raw"], node)

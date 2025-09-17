import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualang_tools.bakery import baking

__all__ = [
    "FitParameters",
    "process_raw_dataset",
    "fit_raw_data",
    "log_fitted_results",
    "baked_flux_xy_segments",
]


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool
    flux_delay: int


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        success = res["success"] if isinstance(res, dict) else res.success
        flux_delay = res["flux_delay"] if isinstance(res, dict) else res.flux_delay
        status = "SUCCESS" if success else "FAIL"
        log_callable(f"Qubit {q}: {status} | flux_delay = {flux_delay} ns")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    data = "state" if hasattr(ds, "state") else "I"

    difference = ds[data].sel(init_state="e") - ds[data].sel(init_state="g")

    if data == "I":
        difference -= difference.mean()

    ds["difference"] = difference
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """

    dfit = ds.groupby("qubit").apply(fit_routine, node=node)

    fit_results = _extract_relevant_fit_parameters(dfit, node)

    return dfit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    fit_results = {}
    for q in fit.qubit.data:
        fit_results[q] = FitParameters(
            success=fit.success.sel(qubit=q).data, flux_delay=fit.flux_delay.sel(qubit=q).data
        )
    return fit_results


def fit_routine(da, node):

    x = da.relative_time.data
    y = da.difference.data[0]

    try:
        sign_changes = np.sign(y)
        crossings = np.where(np.diff(sign_changes) != 0)[0]
        assert len(crossings) == 2, "Expected exactly two sign change points in the data."
        flux_delay = x[(crossings[1] + crossings[0]) // 2]
        da = da.assign(flux_delay=flux_delay)
        da = da.assign(success=True)

    except AssertionError as e:
        print(f"Error processing {da.qubit.data}: {e}")
        flux_delay = 0
        da = da.assign(flux_delay=flux_delay)
        da = da.assign(success=False)
        return da

    print(f"Flux delay for {da.qubit.data}: {flux_delay:.2f} ns")

    return da


def baked_flux_xy_segments(config: dict, waveform: List[float], qb, zeros_each_side: int):
    """Create baked XY+Z (flux) pulse segments for all relative shifts.

    Parameters
    ----------
    config : dict
        Full QUA configuration dict.
    waveform : list[float]
        Flux (Z) pulse samples (without padding) matching x180 length.
    qb : AnyTransmon-like
        Qubit object providing access to xy and z channels.
    zeros_each_side : int
        Number of zeros before and after (total scan range = 2 * zeros_each_side).

    Returns
    -------
    list
        List of baking objects, each representing one relative timing segment.
    """
    pulse_segments = []
    total = 2 * zeros_each_side
    i_key = qb.xy.name + ".x180_DragCosine.wf.I"
    q_key = qb.xy.name + ".x180_DragCosine.wf.Q"
    I_samples = config["waveforms"][i_key]["samples"]
    Q_samples = config["waveforms"][q_key]["samples"]
    for i in range(total):
        with baking(config, padding_method="none") as b:
            wf = [0.0] * i + waveform + [0.0] * (total - i)
            zeros = [0.0] * zeros_each_side
            I_wf = zeros + I_samples + zeros
            Q_wf = zeros + Q_samples + zeros
            assert len(wf) == len(I_wf) == len(Q_wf), "Flux and XY padded waveforms must have identical length"
            b.add_op("flux_pulse", qb.z.name, wf)
            b.add_op("x180", qb.xy.name, [I_wf, Q_wf])
            b.play("flux_pulse", qb.z.name)
            b.play("x180", qb.xy.name)
        pulse_segments.append(b)
    return pulse_segments

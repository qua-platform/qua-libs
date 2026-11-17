"""Analysis helpers for the G-E-F readout duration optimization.

Every shot is acquired with accumulated demodulation, so the dataset holds the |g>, |e> and |f>
IQ shots at every integration duration up to the sweep maximum. At each duration the three blobs
are classified exactly as node 15 does it: the centers come from ``iq_blobs_ef``'s Gaussian fit
and every shot goes to the center with the smallest ``|dI| + |dQ|``, which is the rule
``readout_state_gef`` applies on the OPX. The three-state assignment fidelity is the mean of the
diagonal of the resulting confusion matrix.

The chosen duration is the shortest one whose fidelity is within ``fidelity_tolerance`` of the
best on the axis: past the saturation knee a longer readout costs time in every downstream
experiment for no fidelity gain, while T1 decay during the readout eventually lowers it.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from calibration_utils.iq_blobs_ef import fit_raw_data as fit_gef_blobs
from calibration_utils.readout_duration_power_optimization.analysis import to_volts_per_duration

IQ_KEYS = ["Ig", "Qg", "Ie", "Qe", "If", "Qf"]
STATES = ["g", "e", "f"]


@dataclass
class FitParameters:
    """Fitted G-E-F readout duration for a single qubit."""

    optimal_duration: float
    optimal_fidelity: float
    best_duration: float
    best_fidelity: float
    confusion_matrix: list
    success: bool
    note: str = ""


@dataclass
class DurationChoice:
    """The integration duration a qubit's G-E-F readout should run at."""

    duration: float = float("nan")
    fidelity: float = float("nan")
    best_duration: float = float("nan")
    best_fidelity: float = float("nan")
    success: bool = False
    note: str = ""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the chosen G-E-F readout duration and its fidelity for every qubit."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
            s = (
                f"\tG-E-F readout duration: {fit_results[q]['optimal_duration']:.0f} ns "
                f"(F = {100 * fit_results[q]['optimal_fidelity']:.2f} %)\n"
                f"\tBest on the duration axis: {fit_results[q]['best_duration']:.0f} ns "
                f"(F = {100 * fit_results[q]['best_fidelity']:.2f} %)\n"
            )
        else:
            s_qubit += " FAIL!\n"
            s = ""
        if fit_results[q]["note"]:
            s += f"\t{fit_results[q]['note']}\n"
        log_callable(s_qubit + s)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert the accumulated demodulation results to Volts, one divisor per duration slice."""
    del node
    return to_volts_per_duration(ds, IQ_KEYS)


def select_duration(
    fidelity: xr.DataArray, fidelity_tolerance: float, min_assignment_fidelity: float
) -> DurationChoice:
    """Pick the shortest duration whose fidelity is within ``fidelity_tolerance`` of the best.

    Args:
        fidelity: Three-state assignment fidelity along ``duration`` (NaN where the fit failed).
        fidelity_tolerance: Absolute fidelity the readout may give up to be shorter.
        min_assignment_fidelity: Below this best fidelity the qubit fails.
    """
    durations = np.asarray(fidelity.duration.values, dtype=float)
    values = np.asarray(fidelity.values, dtype=float)
    if not np.isfinite(values).any():
        return DurationChoice(note="the g/e/f blobs could not be fitted at any duration")

    best_index = int(np.nanargmax(values))
    choice = DurationChoice(best_duration=float(durations[best_index]), best_fidelity=float(values[best_index]))
    if choice.best_fidelity < min_assignment_fidelity:
        choice.note = (
            f"the best three-state assignment fidelity {choice.best_fidelity:.3f} is below "
            f"min_assignment_fidelity={min_assignment_fidelity}"
        )
        return choice

    # Durations are ascending, so the first one inside the tolerance band is the shortest.
    index = int(np.flatnonzero(np.nan_to_num(values, nan=-np.inf) >= choice.best_fidelity - fidelity_tolerance)[0])
    choice.duration = float(durations[index])
    choice.fidelity = float(values[index])
    choice.success = True
    if index == len(durations) - 1:
        choice.note = "the fidelity is still rising at max_duration_in_ns; consider a longer sweep"
    return choice


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, xr.Dataset, dict[str, FitParameters]]:
    """Classify the g/e/f shots at every duration and pick each qubit's readout duration.

    Returns:
        ``(ds_fit, ds_iq_blobs, fit_results)``. ``ds_fit`` carries the fidelity and confusion
        matrix along the duration axis plus the chosen duration; ``ds_iq_blobs`` is node 15's
        fit at the chosen duration, used for the IQ-blob and confusion-matrix figures.
    """
    qubit_names = [str(q) for q in ds.qubit.values]
    fidelity = np.full((len(qubit_names), ds.duration.size), np.nan)
    confusion = np.full((len(qubit_names), ds.duration.size, len(STATES), len(STATES)), np.nan)
    for i, q in enumerate(qubit_names):
        for j, duration in enumerate(ds.duration.values):
            fit = _fit_gef_blobs(ds[IQ_KEYS].sel(qubit=[q], duration=duration).drop_vars("duration"))
            if fit is not None:
                confusion[i, j] = fit.confusion_matrix.sel(qubit=q).values
                fidelity[i, j] = np.mean(np.diag(confusion[i, j]))

    ds_fit = ds.assign(
        assignment_fidelity=(["qubit", "duration"], fidelity),
        confusion_matrix=(["qubit", "duration", "prepared", "measured"], confusion),
    ).assign_coords(prepared=STATES, measured=STATES)
    ds_fit.assignment_fidelity.attrs = {"long_name": "three-state assignment fidelity"}

    return _extract_relevant_fit_parameters(ds_fit, node)


def _fit_gef_blobs(ds: xr.Dataset) -> Optional[xr.Dataset]:
    """Node 15's g/e/f classification, or None if the blobs cannot be fitted."""
    try:
        ds_fit, _ = fit_gef_blobs(ds, None)
    except (RuntimeError, ValueError) as error:  # curve_fit failing on unresolved blobs
        logging.getLogger(__name__).debug("g/e/f fit failed: %s", error)
        return None
    return ds_fit


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Select each qubit's readout duration and fit its IQ blobs there."""
    qubit_names = [str(q) for q in ds_fit.qubit.values]
    choices = {
        q: select_duration(
            ds_fit.assignment_fidelity.sel(qubit=q),
            node.parameters.fidelity_tolerance,
            node.parameters.min_assignment_fidelity,
        )
        for q in qubit_names
    }
    ds_fit = ds_fit.assign(
        {
            "optimal_duration": xr.DataArray([choices[q].duration for q in qubit_names], coords={"qubit": qubit_names}),
            "optimal_fidelity": xr.DataArray([choices[q].fidelity for q in qubit_names], coords={"qubit": qubit_names}),
            "best_duration": xr.DataArray(
                [choices[q].best_duration for q in qubit_names], coords={"qubit": qubit_names}
            ),
            "best_fidelity": xr.DataArray(
                [choices[q].best_fidelity for q in qubit_names], coords={"qubit": qubit_names}
            ),
            "success": xr.DataArray([choices[q].success for q in qubit_names], coords={"qubit": qubit_names}),
        }
    )

    # The IQ-blob fit needs one duration slice per qubit, and each qubit has its own. A qubit that
    # failed still has to occupy its slot in the stacked dataset, so it borrows the longest duration;
    # its results are discarded by the success flag below.
    blob_slices = []
    for q in qubit_names:
        duration = choices[q].duration if choices[q].success else ds_fit.duration.values[-1]
        blob_slices.append(ds_fit[IQ_KEYS].sel(qubit=[q], duration=duration).drop_vars("duration"))
    ds_iq_blobs = _fit_gef_blobs(xr.concat(blob_slices, dim="qubit"))

    fit_results = {}
    for q in qubit_names:
        choice = choices[q]
        confusion = (
            ds_fit.confusion_matrix.sel(qubit=q, duration=choice.duration).values.tolist() if choice.success else []
        )
        fit_results[q] = FitParameters(
            optimal_duration=choice.duration,
            optimal_fidelity=choice.fidelity,
            best_duration=choice.best_duration,
            best_fidelity=choice.best_fidelity,
            confusion_matrix=confusion,
            success=choice.success and ds_iq_blobs is not None,
            note=choice.note,
        )
    return ds_fit, ds_iq_blobs, fit_results

"""Analysis helpers for the joint readout duration x power optimization.

The grid is (readout amplitude prefactor) x (integration duration). At every grid point a
two-component spherical Gaussian mixture is fitted to the |g> and |e> IQ samples, which
yields three numbers: the assignment fidelity, the non-outlier fraction, and the ratio
between the two fitted blob variances.

The operating point is the global fidelity maximum, but only over the points whose blobs
survive both gates. The variance ratio is the gate that the non-outlier fraction alone
does not provide: the characteristic high-power failure spreads the excited blob into an
arc while the ground blob stays tight, which keeps the non-outlier fraction healthy while
making the fidelity number meaningless.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from sklearn.mixture import GaussianMixture

from qualibrate import QualibrationNode
from calibration_utils.iq_blobs import fit_raw_data as fit_iq_blobs
from calibration_utils.iq_blobs.analysis import FitParameters as FitParametersIQblobs

# Raw demodulation units are converted to Volts by scaling with 2**12 over the integration
# duration in nanoseconds. Unlike a conventional readout there is no single per-qubit
# length here, so the divisor is the duration of each slice.
DEMOD_TO_VOLTS_NUMERATOR = 2**12

FIT_VALS = ["meas_fidelity", "outliers", "variance_ratio"]


@dataclass
class FitParameters(FitParametersIQblobs):
    """Fit parameters for the joint duration / amplitude optimization on a single qubit."""

    optimal_amplitude: float = 0.0
    optimal_duration: float = 0.0
    optimal_fidelity: float = float("nan")
    note: str = ""


@dataclass
class OperatingPoint:
    """The grid point a qubit's readout should be operated at."""

    amp_prefactor: float
    duration: float
    fidelity: float
    success: bool
    note: str


def to_volts_per_duration(ds: xr.Dataset, keys: Sequence[str]) -> xr.Dataset:
    """Convert accumulated demodulation results to Volts, slice by slice.

    ``qualibration_libs.data.convert_IQ_to_V`` divides by the readout length held in the
    state, which is a single number per qubit. Accumulated demodulation instead produces one
    integral per integration duration, so each slice carries its own divisor. Applying a
    shared divisor would leave a scale error that is invisible in the fidelity map --
    fidelity is scale invariant -- but corrupts the threshold written back to the state.
    """
    return ds.assign({key: ds[key] * DEMOD_TO_VOLTS_NUMERATOR / ds.duration for key in keys})


def blob_statistics(I: np.ndarray, Q: np.ndarray) -> Tuple[float, float, float]:
    """Fit one grid point's IQ blobs and report fidelity, non-outlier fraction, variance ratio.

    Args:
        I: Shape ``(2, n_runs)``, the in-phase samples for the prepared |g> and |e> states.
        Q: Shape ``(2, n_runs)``, the quadrature samples in the same layout.

    Returns:
        ``(assignment fidelity, non-outlier fraction, variance ratio)``. The variance ratio
        is the wider fitted blob over the narrower one, so it is never below 1.
    """
    I = np.asarray(I)
    Q = np.asarray(Q)
    I_mean = np.mean(I, axis=1)
    Q_mean = np.mean(Q, axis=1)
    means_init = [[I_mean[0], Q_mean[0]], [I_mean[1], Q_mean[1]]]
    precisions_init = [1 / ((np.mean(np.var(I, axis=1)) + np.mean(np.var(Q, axis=1))) / 2)] * 2
    clf = GaussianMixture(
        n_components=2,
        covariance_type="spherical",
        means_init=means_init,
        precisions_init=precisions_init,
        tol=1e-5,
        reg_covar=1e-12,
    )
    X = np.array([np.asarray(I).flatten(), np.asarray(Q).flatten()]).T
    clf.fit(X)

    ground_pred = clf.predict(np.array([I[0], Q[0]]).T)
    excited_pred = clf.predict(np.array([I[1], Q[1]]).T)
    ground_hits = ground_pred.size - np.count_nonzero(ground_pred)
    excited_hits = np.count_nonzero(excited_pred)
    meas_fidelity = (ground_hits / len(I[0]) + excited_hits / len(I[1])) / 2

    loglikelihood = clf.score_samples(X)
    max_ll = np.max(loglikelihood)
    non_outliers = np.sum(loglikelihood > np.log(0.01) + max_ll) / len(X)

    covariances = np.asarray(clf.covariances_, dtype=float)
    smallest = float(np.min(covariances))
    variance_ratio = float(np.max(covariances) / smallest) if smallest > 0 else float("inf")

    return float(meas_fidelity), float(non_outliers), variance_ratio


def _blob_statistics_array(I: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """``blob_statistics`` as a single stacked array, which is what ``apply_ufunc`` needs.

    A vectorized ufunc with one output core dimension must return one array, not a tuple.
    The ordering matches :data:`FIT_VALS`.
    """
    return np.array(blob_statistics(I, Q))


def select_operating_point(
    fidelity: xr.DataArray,
    non_outlier: xr.DataArray,
    variance_ratio: xr.DataArray,
    outliers_threshold: float,
    max_variance_ratio: float,
    fixed_duration: Optional[float] = None,
) -> OperatingPoint:
    """Pick the highest-fidelity grid point whose blobs pass both quality gates.

    Args:
        fidelity: Assignment fidelity over ``(amp_prefactor, duration)``.
        non_outlier: Non-outlier fraction over the same grid.
        variance_ratio: Blob variance ratio over the same grid.
        outliers_threshold: Minimum non-outlier fraction for a point to be eligible.
        max_variance_ratio: Maximum blob variance ratio for a point to be eligible.
        fixed_duration: If given, only grid points at this integration duration are
            considered, and the search reduces to an amplitude sweep at that duration. This
            is what keeps the selection honest when the readout length is not going to be
            updated: every quantity derived from the chosen point -- the thresholds, the
            rotation angle, the confusion matrix -- then describes the integration duration
            the readout will actually run at.

    Returns:
        The selected point, or an unsuccessful :class:`OperatingPoint` whose note names the
        gate (or gates) that rejected every point.

    Ties matter here and are broken deliberately. Assignment fidelity is quantised to one
    part in twice the shot count, so in the saturated region many points carry exactly the
    same value. The search runs in row-major order over (amplitude, duration) with both axes
    ascending, so a tie resolves to the lowest amplitude and then the shortest duration --
    the cheapest point that still buys the available fidelity.
    """
    if fixed_duration is not None:
        matches = np.isclose(np.asarray(fidelity.duration.values, dtype=float), float(fixed_duration))
        if not matches.any():
            return OperatingPoint(
                amp_prefactor=float("nan"),
                duration=float("nan"),
                fidelity=float("nan"),
                success=False,
                note=(
                    f"the readout length is held at {float(fixed_duration):.0f} ns because "
                    f"update_readout_length is off, and that duration is not on the swept axis "
                    f"{[float(d) for d in fidelity.duration.values]} ns"
                ),
            )
        on_grid = fidelity.duration.values[matches][:1]
        fidelity = fidelity.sel(duration=on_grid)
        non_outlier = non_outlier.sel(duration=on_grid)
        variance_ratio = variance_ratio.sel(duration=on_grid)

    passes_outliers = non_outlier >= outliers_threshold
    passes_variance = variance_ratio <= max_variance_ratio
    eligible = passes_outliers & passes_variance

    if not bool(eligible.any()):
        reasons: List[str] = []
        if not bool(passes_outliers.any()):
            reasons.append(f"the non-outlier fraction never reached {outliers_threshold}")
        if not bool(passes_variance.any()):
            reasons.append(f"the blob variance ratio never fell to {max_variance_ratio}")
        if not reasons:
            reasons.append("no point passed the non-outlier fraction and the blob variance ratio at the same time")
        if fixed_duration is not None:
            reasons.append(
                f"the search was restricted to {float(fixed_duration):.0f} ns because update_readout_length is off"
            )
        return OperatingPoint(
            amp_prefactor=float("nan"),
            duration=float("nan"),
            fidelity=float("nan"),
            success=False,
            note="; ".join(reasons),
        )

    masked = fidelity.where(eligible)
    flat_index = int(np.nanargmax(masked.values))
    amp_index, duration_index = np.unravel_index(flat_index, masked.shape)

    return OperatingPoint(
        amp_prefactor=float(masked.amp_prefactor.values[amp_index]),
        duration=float(masked.duration.values[duration_index]),
        fidelity=float(masked.values[amp_index, duration_index]),
        success=True,
        note="",
    )


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the chosen operating point and its fidelity for every qubit."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
            s = (
                f"\tOptimal readout amplitude: {1e3 * fit_results[q]['optimal_amplitude']:.3f} mV\n"
                f"\tOptimal integration duration: {fit_results[q]['optimal_duration']:.0f} ns\n"
                f"\tAssignment fidelity there: {100 * fit_results[q]['optimal_fidelity']:.2f} %\n"
            )
        else:
            s_qubit += " FAIL!\n"
            s = f"\tNo grid point passed the blob quality gates: {fit_results[q]['note']}\n"
        log_callable(s_qubit + s)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert to Volts per integration duration and stack |g> and |e> onto a state axis."""
    # Skip if the data has already been processed
    if ~np.all([var in ds.data_vars for var in ["Ig", "Qg", "Ie", "Qe"]]):
        return ds

    ds = to_volts_per_duration(ds, ["Ig", "Qg", "Ie", "Qe"])

    operation = node.parameters.operation
    readout_amplitudes = np.array(
        [ds.amp_prefactor * q.resonator.operations[operation].amplitude for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(readout_amplitude=(["qubit", "amp_prefactor"], readout_amplitudes))
    ds.readout_amplitude.attrs = {"long_name": "readout amplitude", "units": "V"}

    # Rearrange the data to combine Ig and Ie into I, and Qg and Qe into Q
    ds_rearranged = xr.Dataset()
    ds_rearranged["I"] = xr.concat([ds.Ig, ds.Ie], dim="state").assign_coords(state=[0, 1])
    ds_rearranged["Q"] = xr.concat([ds.Qg, ds.Qe], dim="state").assign_coords(state=[0, 1])
    for var in ds.coords:
        if var not in ds_rearranged.coords:
            ds_rearranged[var] = ds[var]
    for var in ds.data_vars:
        if var not in ["Ig", "Ie", "Qg", "Qe"]:
            ds_rearranged[var] = ds[var]

    return ds_rearranged


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, xr.Dataset, dict[str, FitParameters]]:
    """Fit every grid point, gate on blob quality, and pick each qubit's operating point.

    Returns:
        ``(ds_fit, ds_iq_blobs, fit_results)``. ``ds_fit`` carries the fidelity, non-outlier
        and variance-ratio maps plus the selected point; ``ds_iq_blobs`` is the IQ-blob fit
        evaluated at the selected point, which supplies the angle, thresholds and confusion
        matrix written to the state.
    """
    fit_data = xr.apply_ufunc(
        _blob_statistics_array,
        ds.I,
        ds.Q,
        input_core_dims=[["state", "n_runs"], ["state", "n_runs"]],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    fit_data = fit_data.assign_coords(fit_vals=FIT_VALS)
    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])

    return _extract_relevant_fit_parameters(ds_fit, node)


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Select each qubit's operating point and fit its IQ blobs there."""
    qubit_names = [str(q) for q in ds_fit.qubit.values]
    operation_name = node.parameters.operation

    # With `update_readout_length` off the readout keeps the length it already has, so the
    # duration axis is not free to move: a point picked at some other duration would hand the
    # state a threshold, a rotation angle and a confusion matrix describing an integration
    # time the readout will never run at. Pinning the search to the current length turns the
    # node into an amplitude sweep at that length, which is what the parameter asks for.
    fixed_durations: Dict[str, Optional[float]] = {q: None for q in qubit_names}
    if not node.parameters.update_readout_length:
        for q in qubit_names:
            fixed_durations[q] = float(node.machine.qubits[q].resonator.operations[operation_name].length)

    operating_points: Dict[str, OperatingPoint] = {}
    for q in qubit_names:
        per_qubit = ds_fit.fit_data.sel(qubit=q)
        operating_points[q] = select_operating_point(
            per_qubit.sel(fit_vals="meas_fidelity"),
            per_qubit.sel(fit_vals="outliers"),
            per_qubit.sel(fit_vals="variance_ratio"),
            outliers_threshold=node.parameters.outliers_threshold,
            max_variance_ratio=node.parameters.max_variance_ratio,
            fixed_duration=fixed_durations[q],
        )
    # The absolute amplitude the chosen prefactor corresponds to. It is carried on the fit
    # dataset as well as in the fit results because the plots annotate with it.
    optimal_amplitudes = {
        q: (
            operating_points[q].amp_prefactor * node.machine.qubits[q].resonator.operations[operation_name].amplitude
            if operating_points[q].success
            else float("nan")
        )
        for q in qubit_names
    }
    ds_fit = ds_fit.assign(
        {
            "optimal_amplitude": xr.DataArray(
                [optimal_amplitudes[q] for q in qubit_names], coords={"qubit": ds_fit.qubit.data}
            ),
            "optimal_amp_prefactor": xr.DataArray(
                [operating_points[q].amp_prefactor for q in qubit_names], coords={"qubit": ds_fit.qubit.data}
            ),
            "optimal_duration": xr.DataArray(
                [operating_points[q].duration for q in qubit_names], coords={"qubit": ds_fit.qubit.data}
            ),
            "optimal_fidelity": xr.DataArray(
                [operating_points[q].fidelity for q in qubit_names], coords={"qubit": ds_fit.qubit.data}
            ),
            "quality_gate_passed": xr.DataArray(
                [operating_points[q].success for q in qubit_names], coords={"qubit": ds_fit.qubit.data}
            ),
        }
    )

    # The IQ-blob fit needs one (amplitude, duration) slice per qubit, and each qubit has its
    # own. A qubit that failed the gate still has to occupy its slot in the stacked dataset,
    # so it borrows the first grid point; its results are discarded by the success flag below.
    blob_slices = []
    for q in qubit_names:
        point = operating_points[q]
        selector = (
            dict(amp_prefactor=point.amp_prefactor, duration=point.duration)
            if point.success
            else dict(amp_prefactor=ds_fit.amp_prefactor.values[0], duration=ds_fit.duration.values[0])
        )
        at_point = ds_fit.sel(qubit=q, **selector)
        blob_slices.append(
            xr.Dataset(
                {
                    "Ig": at_point.I.sel(state=0).drop_vars("state"),
                    "Qg": at_point.Q.sel(state=0).drop_vars("state"),
                    "Ie": at_point.I.sel(state=1).drop_vars("state"),
                    "Qe": at_point.Q.sel(state=1).drop_vars("state"),
                }
            ).expand_dims(qubit=[q])
        )
    ds_blob_input = xr.concat(blob_slices, dim="qubit")
    # `amp_prefactor` and `duration` differ per qubit, so they are no longer shared axes.
    ds_blob_input = ds_blob_input.drop_vars(
        [v for v in ("amp_prefactor", "duration", "readout_amplitude") if v in ds_blob_input.coords]
    )
    ds_iq_blobs, iq_fit_results = fit_iq_blobs(ds_blob_input, node)

    fit_results = {}
    for q in qubit_names:
        point = operating_points[q]
        params_dict = dict(iq_fit_results[q].__dict__)
        params_dict["success"] = bool(params_dict["success"]) and point.success
        params_dict["optimal_amplitude"] = float(optimal_amplitudes[q])
        params_dict["optimal_duration"] = point.duration
        params_dict["optimal_fidelity"] = point.fidelity
        params_dict["note"] = point.note
        fit_results[q] = FitParameters(**params_dict)

    return ds_fit, ds_iq_blobs, fit_results

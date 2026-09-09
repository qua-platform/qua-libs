"""Shot-resolved GMM selection, following readout_power_optimization (08b)."""

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from calibration_utils.iq_blobs.analysis import fit_raw_data as fit_ge_blobs


@dataclass
class FitParameters:
    success: bool = False
    optimal_duration: int = 0
    readout_fidelity: float = float("nan")  # percent, matching 08b
    iw_angle: float = float("nan")
    ge_threshold: float = float("nan")
    rus_threshold: float = float("nan")
    confusion_matrix: list = field(default_factory=list)


def process_raw_dataset(ds, node):
    if "I" in ds and "Q" in ds:
        return ds
    states = "gef" if node.parameters.operation == "readout_GEF" else "ge"
    # The library converter uses the machine's fixed readout length. Here each
    # sweep point must instead be normalized by its own integration duration.
    result = xr.Dataset(attrs=ds.attrs)
    for quadrature in "IQ":
        result[quadrature] = xr.concat([ds[f"{quadrature}{state}"] for state in states], dim="state")
        result[quadrature] = result[quadrature].assign_coords(state=list(range(len(states)))) * 2**12 / ds.duration
        result[quadrature].attrs["units"] = "V"
    result.duration.attrs = {"long_name": "readout duration", "units": "ns"}
    return result


def fit_gmm(samples):
    """Return fidelity, non-outlier fraction and centers ordered by prepared state.

    samples has shape (state, shot, IQ). For GEF, score the nearest-center
    classifier used by the hardware, using the centers obtained from the GMM.
    """
    n_states, n_shots, _ = samples.shape
    if not np.isfinite(samples).all() or n_shots < 10:
        raise ValueError("Need at least ten finite shots per state")
    variance = float(np.mean(np.var(samples, axis=1)))
    if variance <= 0:
        raise ValueError("Zero-variance IQ data")
    means = samples.mean(axis=1)
    model = GaussianMixture(
        n_components=n_states, covariance_type="spherical", means_init=means,
        precisions_init=np.full(n_states, 1 / variance), tol=1e-5, reg_covar=1e-12, random_state=0,
    ).fit(samples.reshape(-1, 2))
    if not model.converged_:
        raise ValueError("GMM did not converge")
    # GMM component indices are arbitrary; match them back to prepared states.
    rows, columns = linear_sum_assignment(np.sum((means[:, None, :] - model.means_[None, :, :]) ** 2, axis=2))
    permutation = columns[np.argsort(rows)]
    centers = model.means_[permutation]
    labels = np.argsort(permutation)[model.predict(samples.reshape(-1, 2))].reshape(n_states, n_shots)
    if n_states == 3:
        labels = np.argmin(np.sum((samples[:, :, None, :] - centers[None, None, :, :]) ** 2, axis=-1), axis=-1)
    confusion = np.array([[np.mean(labels[s] == measured) for measured in range(n_states)] for s in range(n_states)])
    likelihood = model.score_samples(samples.reshape(-1, 2))
    non_outliers = float(np.mean(likelihood > likelihood.max() + np.log(0.01)))
    return float(np.trace(confusion) / n_states), non_outliers, centers, confusion


def fit_raw_data(ds, node):
    names = ds.qubit.values.tolist()
    durations = ds.duration.values
    n_states = ds.sizes["state"]
    metrics = np.full((len(names), len(durations), 2), np.nan)
    selected_iq = np.full((len(names), n_states, ds.sizes["n_runs"], 2), np.nan)
    centers = np.full((len(names), n_states, 2), np.nan)
    confusion = np.full((len(names), n_states, n_states), np.nan)
    optimum = np.full(len(names), np.nan)
    results = {}
    for qi, name in enumerate(names):
        result = results[name] = FitParameters()
        fitted_points = {}
        for di in range(len(durations)):
            point = ds.sel(qubit=name).isel(duration=di)
            samples = np.stack([point.I.transpose("state", "n_runs"), point.Q.transpose("state", "n_runs")], axis=-1)
            try:
                fidelity, non_outliers, point_centers, point_confusion = fit_gmm(samples)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            metrics[qi, di] = fidelity, non_outliers
            fitted_points[di] = samples, point_centers, point_confusion
        valid = np.isfinite(metrics[qi, :, 0]) & (metrics[qi, :, 1] >= node.parameters.outliers_threshold)
        if not valid.any():
            continue
        # Sorted durations and argmax resolve equal fidelities in favor of shorter readout.
        best = int(np.argmax(np.where(valid, metrics[qi, :, 0], -np.inf)))
        samples, point_centers, point_confusion = fitted_points[best]
        result.optimal_duration = int(durations[best])
        if n_states == 2:
            best_ds = xr.Dataset(
                {f"{quadrature}{state}": (("qubit", "n_runs"), samples[si, :, axis][None, :])
                 for si, state in enumerate("ge") for axis, quadrature in enumerate("IQ")},
                coords={"qubit": [name], "n_runs": ds.n_runs},
            )
            # Fit each qubit separately so IQ rotations are independent of other qubits.
            qubit = next(q for q in node.namespace["qubits"] if q.name == name)
            fit, ge_results = fit_ge_blobs(best_ds, SimpleNamespace(namespace={"qubits": [qubit]}))
            ge = ge_results[name]
            result.success = ge.success
            result.iw_angle, result.ge_threshold, result.rus_threshold = ge.iw_angle, ge.ge_threshold, ge.rus_threshold
            result.readout_fidelity = ge.readout_fidelity
            point_confusion = np.array(ge.confusion_matrix)
            for si, state in enumerate("ge"):
                samples[si, :, 0] = fit[f"I{state}_rot"].values[0]
                samples[si, :, 1] = fit[f"Q{state}_rot"].values[0]
            point_centers = samples.mean(axis=1)
        else:
            result.success = bool(np.isfinite(point_centers).all())
            result.readout_fidelity = 100 * float(np.trace(point_confusion) / n_states)
        result.confusion_matrix = point_confusion.tolist()
        if result.success:
            optimum[qi] = result.optimal_duration
            selected_iq[qi], centers[qi], confusion[qi] = samples, point_centers, point_confusion
    ds_fit = ds.assign(
        fit_data=(("qubit", "duration", "fit_vals"), metrics),
        optimal_duration=("qubit", optimum),
    ).assign_coords(fit_vals=["meas_fidelity", "non_outliers"])
    ds_fit["valid_duration"] = ds_fit.fit_data.sel(fit_vals="non_outliers") >= node.parameters.outliers_threshold
    blobs = xr.Dataset(
        {
            "I": (("qubit", "state", "n_runs"), selected_iq[..., 0]),
            "Q": (("qubit", "state", "n_runs"), selected_iq[..., 1]),
            "centers": (("qubit", "state", "quadrature"), centers),
            "confusion_matrix": (("qubit", "state", "measured_state"), confusion),
            "ge_threshold": ("qubit", [results[q].ge_threshold for q in names]),
            "rus_threshold": ("qubit", [results[q].rus_threshold for q in names]),
        },
        coords={"qubit": names, "state": ds.state, "n_runs": ds.n_runs,
                "quadrature": ["I", "Q"], "measured_state": range(n_states)},
    )
    return ds_fit, blobs, results


def log_fitted_results(results, log_callable=None):
    log = log_callable or logging.getLogger(__name__).info
    for name, result in results.items():
        if result["success"]:
            log(f"{name}: duration = {result['optimal_duration']} ns; assignment fidelity = {result['readout_fidelity']:.2f}%")
        else:
            log(f"{name}: FAILED — no valid duration or discriminator fit; state unchanged")

"""Shot-resolved GMM selection, following readout_power_optimization (08b)."""

import logging
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture


@dataclass
class FitParameters:
    """Per-qubit calibration values selected from the duration sweep."""

    success: bool = False
    optimal_duration: int = 0
    readout_fidelity: float = float("nan")
    """Assignment fidelity in percent, matching readout-power optimization."""

    ge_fidelity: float = float("nan")
    """Mean of P(g|g) and P(e|e), in percent."""

    ef_fidelity: float = float("nan")
    """Mean of P(e|e) and P(f|f), in percent."""

    g_assignment: float = float("nan")
    e_assignment: float = float("nan")
    f_assignment: float = float("nan")
    g_non_outliers: float = float("nan")
    e_non_outliers: float = float("nan")
    f_non_outliers: float = float("nan")
    ge_reference_max: float = float("nan")
    ge_required: float = float("nan")

    confusion_matrix: list = field(default_factory=list)


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Combine state streams and normalize each duration point into volts."""
    if "I" in ds and "Q" in ds:
        return ds
    states = "gef"
    # The library converter uses the machine's fixed readout length. Here each
    # sweep point must instead be normalized by its own integration duration.
    result = xr.Dataset(attrs=ds.attrs)
    for quadrature in "IQ":
        result[quadrature] = xr.concat([ds[f"{quadrature}{state}"] for state in states], dim="state")
        result[quadrature] = result[quadrature].assign_coords(state=list(range(len(states)))) * 2**12 / ds.duration
        result[quadrature].attrs["units"] = "V"
    result.duration.attrs = {"long_name": "readout duration", "units": "ns"}
    return result


def fit_gmm(samples: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return assignment and state-balanced Gaussian quality metrics.

    Samples have shape (3, shot, IQ). Score the nearest-center
    classifier using the centers obtained from the GMM.  The non-outlier
    fraction is evaluated separately around each prepared state's matched
    Gaussian.  This avoids rejecting a legitimate broad blob merely because a
    different state has a narrower, higher peak likelihood.
    """
    n_states, n_shots, _ = samples.shape
    if not np.isfinite(samples).all() or n_shots < 10:
        raise ValueError("Need at least ten finite shots per state")
    variance = float(np.mean(np.var(samples, axis=1)))
    if variance <= 0:
        raise ValueError("Zero-variance IQ data")
    means = samples.mean(axis=1)
    model = GaussianMixture(
        n_components=n_states,
        covariance_type="spherical",
        means_init=means,
        precisions_init=np.full(n_states, 1 / variance),
        tol=1e-5,
        reg_covar=1e-12,
        random_state=0,
    ).fit(samples.reshape(-1, 2))
    if not model.converged_:
        raise ValueError("GMM did not converge")
    # GMM component indices are arbitrary; match them back to prepared states.
    rows, columns = linear_sum_assignment(np.sum((means[:, None, :] - model.means_[None, :, :]) ** 2, axis=2))
    permutation = columns[np.argsort(rows)]
    centers = model.means_[permutation]
    labels = np.argmin(
        np.sum(
            (samples[:, :, None, :] - centers[None, None, :, :]) ** 2,
            axis=-1,
        ),
        axis=-1,
    )
    confusion = np.array(
        [[np.mean(labels[state] == measured) for measured in range(n_states)] for state in range(n_states)]
    )
    matched_variances = model.covariances_[permutation]
    radius_squared = np.sum((samples - centers[:, None, :]) ** 2, axis=-1) / matched_variances[:, None]
    # A spherical Gaussian falls to 1% of its peak at r^2=-2*ln(0.01).
    state_non_outliers = np.mean(radius_squared <= -2 * np.log(0.01), axis=1)
    # Duration eligibility belongs to the first (GE) stage of the selector, so
    # only g/e Gaussian quality gates the candidate set.  The f quality remains
    # recorded and its failures reduce the subsequent EF objective naturally.
    non_outliers = float(np.min(state_non_outliers[:2]))
    return (
        float(np.trace(confusion) / n_states),
        non_outliers,
        centers,
        confusion,
        state_non_outliers,
    )


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, xr.Dataset, dict[str, FitParameters]]:
    """Fit every duration, hold GE near its maximum, then maximize EF.

    Selection is intentionally lexicographic instead of maximizing the mean
    diagonal of the three-state confusion matrix.  First discard failed GMM
    points.  Next form a GE plateau whose fidelity is both above
    ``minimum_ge_fidelity`` and within ``ge_fidelity_tolerance`` of the best GE
    value for that qubit.  Finally maximize EF fidelity only on that plateau.
    Because durations are sorted, exact ties choose the shorter readout.

    Gaussian inlier fractions are always retained as diagnostics.  They are an
    optional hard gate because real e->g relaxation or f leakage during a long
    measurement creates non-Gaussian tails that are already reflected in the
    assignment confusion matrix; rejecting them twice can hide the best usable
    discriminator.
    """
    names = ds.qubit.values.tolist()
    durations = ds.duration.values
    n_states = 3
    metric_names = [
        "gef_fidelity",
        "ge_fidelity",
        "ef_fidelity",
        "g_assignment",
        "e_assignment",
        "f_assignment",
        "non_outliers",
        "g_non_outliers",
        "e_non_outliers",
        "f_non_outliers",
    ]
    metrics = np.full((len(names), len(durations), len(metric_names)), np.nan)
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
            samples = np.stack(
                [
                    point.I.transpose("state", "n_runs"),
                    point.Q.transpose("state", "n_runs"),
                ],
                axis=-1,
            )
            try:
                (
                    fidelity,
                    non_outliers,
                    point_centers,
                    point_confusion,
                    state_non_outliers,
                ) = fit_gmm(samples)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            diagonal = np.diag(point_confusion)
            ge_fidelity = float(np.mean(diagonal[:2]))
            ef_fidelity = float(np.mean(diagonal[1:]))
            metrics[qi, di] = (
                fidelity,
                ge_fidelity,
                ef_fidelity,
                diagonal[0],
                diagonal[1],
                diagonal[2],
                non_outliers,
                state_non_outliers[0],
                state_non_outliers[1],
                state_non_outliers[2],
            )
            fitted_points[di] = samples, point_centers, point_confusion
        base_valid = np.isfinite(metrics[qi, :, 0])
        if node.parameters.enforce_outliers_threshold:
            base_valid &= metrics[qi, :, 6] >= node.parameters.outliers_threshold
        if not base_valid.any():
            continue
        ge_reference_max = float(np.max(metrics[qi, base_valid, 1]))
        ge_required = max(
            float(node.parameters.minimum_ge_fidelity),
            ge_reference_max - float(node.parameters.ge_fidelity_tolerance),
        )
        ge_valid = base_valid & (metrics[qi, :, 1] >= ge_required)
        if not ge_valid.any():
            result.ge_reference_max = 100 * ge_reference_max
            result.ge_required = 100 * ge_required
            continue
        # The objective is EF only after GE has passed. Sorted durations and
        # argmax resolve exact EF ties in favor of the shorter readout.
        best = int(np.argmax(np.where(ge_valid, metrics[qi, :, 2], -np.inf)))
        samples, point_centers, point_confusion = fitted_points[best]
        diagonal = np.diag(point_confusion)
        result.optimal_duration = int(durations[best])
        result.success = bool(np.isfinite(point_centers).all())
        result.readout_fidelity = 100 * float(np.trace(point_confusion) / n_states)
        result.ge_fidelity = 100 * float(np.mean(diagonal[:2]))
        result.ef_fidelity = 100 * float(np.mean(diagonal[1:]))
        result.g_assignment = 100 * float(diagonal[0])
        result.e_assignment = 100 * float(diagonal[1])
        result.f_assignment = 100 * float(diagonal[2])
        result.g_non_outliers = 100 * float(metrics[qi, best, 7])
        result.e_non_outliers = 100 * float(metrics[qi, best, 8])
        result.f_non_outliers = 100 * float(metrics[qi, best, 9])
        result.ge_reference_max = 100 * ge_reference_max
        result.ge_required = 100 * ge_required
        result.confusion_matrix = point_confusion.tolist()
        if result.success:
            optimum[qi] = result.optimal_duration
            selected_iq[qi], centers[qi], confusion[qi] = samples, point_centers, point_confusion
    ds_fit = ds.assign(
        fit_data=(("qubit", "duration", "fit_vals"), metrics),
        optimal_duration=("qubit", optimum),
    ).assign_coords(fit_vals=metric_names)
    ds_fit["base_valid_duration"] = np.isfinite(ds_fit.fit_data.sel(fit_vals="gef_fidelity"))
    if node.parameters.enforce_outliers_threshold:
        ds_fit["base_valid_duration"] &= (
            ds_fit.fit_data.sel(fit_vals="non_outliers") >= node.parameters.outliers_threshold
        )
    ge_reference = ds_fit.fit_data.sel(fit_vals="ge_fidelity").where(ds_fit.base_valid_duration).max("duration")
    ds_fit["ge_reference_max"] = ge_reference
    ds_fit["ge_required"] = xr.apply_ufunc(
        np.maximum,
        ge_reference - node.parameters.ge_fidelity_tolerance,
        node.parameters.minimum_ge_fidelity,
    )
    ds_fit["valid_duration"] = ds_fit.base_valid_duration & (
        ds_fit.fit_data.sel(fit_vals="ge_fidelity") >= ds_fit.ge_required
    )
    blobs = xr.Dataset(
        {
            "I": (("qubit", "state", "n_runs"), selected_iq[..., 0]),
            "Q": (("qubit", "state", "n_runs"), selected_iq[..., 1]),
            "centers": (("qubit", "state", "quadrature"), centers),
            "confusion_matrix": (("qubit", "state", "measured_state"), confusion),
        },
        coords={
            "qubit": names,
            "state": ds.state,
            "n_runs": ds.n_runs,
            "quadrature": ["I", "Q"],
            "measured_state": range(n_states),
        },
    )
    return ds_fit, blobs, results


def log_fitted_results(results: dict, log_callable=None) -> None:
    """Log the two-stage selection and selected confusion-matrix metrics."""
    log = log_callable or logging.getLogger(__name__).info
    for name, result in results.items():
        if result["success"]:
            log(
                f"{name}: duration = {result['optimal_duration']} ns; "
                f"GE = {result['ge_fidelity']:.2f}% "
                f"(required {result['ge_required']:.2f}%, best {result['ge_reference_max']:.2f}%); "
                f"EF = {result['ef_fidelity']:.2f}%; "
                f"GEF mean = {result['readout_fidelity']:.2f}%; "
                f"diag(g,e,f) = ({result['g_assignment']:.2f}%, "
                f"{result['e_assignment']:.2f}%, {result['f_assignment']:.2f}%); "
                f"inliers(g,e,f) = ({result['g_non_outliers']:.2f}%, "
                f"{result['e_non_outliers']:.2f}%, {result['f_non_outliers']:.2f}%)"
            )
        else:
            log(f"{name}: FAILED — no valid duration or discriminator fit; state unchanged")

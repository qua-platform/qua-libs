"""Per-qubit-pair RB fitting mechanics.

Survival statistics, exponential curve fit, fit-quality validation, and the
Standard / Interleaved pair fitters. Alpha -> fidelity conversion is delegated
to ``fidelity.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.common_utils.rb import (
    fit_survival,
    rb_decay_curve,
    validate_fit_parameters,
    validate_residuals,
    SHALLOW_LEAKAGE_THRESHOLD,
    LeakageFit,
    correct_populations,
    fit_leakage,
    leakage_decay_curve,
    leakage_per_gate,
    mean_and_stderr_over_sequences,
    read_confusion_matrix,
)
from calibration_utils.two_qubit_rb import fidelity

# Empirical tolerances for soft (log-only) survival-curve sanity checks — not
# derived from a model; chosen to flag clearly-off data without false-positiving
# on ordinary shot noise.
_MONOTONIC_DECAY_MIN_TOLERANCE = 0.03
_SHALLOW_SURVIVAL_THRESHOLD = 0.35


def _survival_over_sequences(ds_qp: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Mean P(|00>) and its standard error vs circuit depth, from the per-sequence survival."""
    per_sequence = _survival_per_sequence(ds_qp).transpose(..., "sequence")
    mean, stderr = mean_and_stderr_over_sequences(per_sequence.values)
    template = per_sequence.isel(sequence=0, drop=True)
    return template.copy(data=mean), template.copy(data=stderr)


def _survival_probability(ds_qp: xr.Dataset) -> xr.DataArray:
    """P(|00>) vs circuit depth, averaged over sequence and shots."""
    return _survival_over_sequences(ds_qp)[0]


def _survival_per_sequence(ds_qp: xr.Dataset) -> xr.DataArray:
    """P(|00>) per random sequence at each depth, averaged over shots."""
    survival = (ds_qp.state == 0).mean(dim="shots")
    if "qubit_pair" in survival.dims:
        survival = survival.squeeze("qubit_pair", drop=True)
    return survival


def _survival_stderr(ds_qp: xr.Dataset) -> xr.DataArray:
    """Standard error of the mean survival probability at each circuit depth.

    Shot-averaged survival per random sequence, then SEM across sequences only
    (shots are repeats of the same circuit, not independent RB samples).
    """
    return _survival_over_sequences(ds_qp)[1]


def _check_survival_soft_warnings(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray,
) -> list[str]:
    """Soft checks on raw survival data (warnings only; do not fail the fit)."""
    warnings: list[str] = []
    depths = np.asarray(circuit_depths, dtype=float)
    surv = np.asarray(survival, dtype=float)
    err = np.asarray(stderr, dtype=float)

    finite = np.isfinite(surv)
    if not finite.any():
        return warnings

    shallow_idx = int(np.argmin(depths))
    deep_idx = int(np.argmax(depths))
    shallow_depth = float(depths[shallow_idx])
    deep_depth = float(depths[deep_idx])
    shallow_survival = float(surv[shallow_idx])
    deep_survival = float(surv[deep_idx])

    if deep_depth > shallow_depth and np.isfinite(deep_survival) and np.isfinite(shallow_survival):
        tolerance = max(
            float(err[shallow_idx]) if np.isfinite(err[shallow_idx]) else 0.0,
            float(err[deep_idx]) if np.isfinite(err[deep_idx]) else 0.0,
            _MONOTONIC_DECAY_MIN_TOLERANCE,
        )
        if deep_survival > shallow_survival + tolerance:
            warnings.append(
                f"P(|00>) at max depth {deep_depth:g} ({deep_survival:.3f}) exceeds "
                f"shallow depth {shallow_depth:g} ({shallow_survival:.3f}) by more than "
                f"{tolerance:.3f} — decay is not monotonic; check SPAM/readout or shot count."
            )

    if np.isfinite(shallow_survival) and shallow_survival < _SHALLOW_SURVIVAL_THRESHOLD:
        warnings.append(
            f"P(|00>) at depth {shallow_depth:g} is {shallow_survival:.3f} (well below 0.5) — "
            "suggests readout, reset, or SPAM issues; fitted fidelity may be unreliable."
        )

    return warnings


def _validate_rb_fit(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray,
    fit_amplitude: float,
    alpha: float,
    fit_offset: float,
    *,
    interleaved: bool,
    fit_fidelity: float,
    fit_fidelity_stderr: float | None = None,
    standard_rb_alpha: float | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Run all hard-fail RB fit validations."""
    issues: list[str] = []
    issues.extend(validate_fit_parameters(fit_amplitude, alpha, fit_offset))
    issues.extend(fidelity.validate_fidelity_bounds(fit_fidelity, interleaved=interleaved))
    issues.extend(fidelity.validate_fidelity_uncertainty(fit_fidelity, fit_fidelity_stderr, interleaved=interleaved))
    if interleaved and standard_rb_alpha is not None:
        issues.extend(fidelity.validate_interleaved_alpha(alpha, standard_rb_alpha))
    if not issues:
        issues.extend(validate_residuals(circuit_depths, survival, stderr, fit_amplitude, alpha, fit_offset))
    return len(issues) == 0, tuple(issues)


def _fit_survival_for_pair(da: xr.Dataset) -> dict:
    """Compute survival stats + exponential fit for one qubit pair."""
    survival = _survival_probability(da)
    survival_per_sequence = _survival_per_sequence(da)
    stderr = _survival_stderr(da)
    circuit_depths = survival.circuit_depth.values
    survival_vals = survival.values
    stderr_vals = stderr.values

    fit_warnings = tuple(_check_survival_soft_warnings(circuit_depths, survival_vals, stderr_vals))
    fit_params = fit_survival(circuit_depths, survival_vals, stderr_vals)

    return {
        "survival": survival,
        "survival_per_sequence": survival_per_sequence,
        "stderr": stderr,
        "circuit_depths": circuit_depths,
        "survival_vals": survival_vals,
        "stderr_vals": stderr_vals,
        "fit_warnings": fit_warnings,
        "fit_params": fit_params,
    }


def _fitted_curve_dataarray(circuit_depths, fit_params) -> xr.DataArray:
    if fit_params is not None:
        fit_amplitude, alpha, fit_offset, _, _, _ = fit_params
        fitted_curve = rb_decay_curve(circuit_depths, fit_amplitude, alpha, fit_offset)
    else:
        fitted_curve = np.full_like(circuit_depths, np.nan, dtype=float)
    return xr.DataArray(fitted_curve, dims=["circuit_depth"], coords={"circuit_depth": circuit_depths})


def _standard_rb_reference(node: QualibrationNode, qp_name: str) -> dict:
    """The 37a Standard RB reference 37b compares against: ``alpha``, ``fidelity``, ``load_id`` and ``source``.

    A leakage run (any readout or reset) writes the same ``StandardRB*`` as a normal one: the RB decay
    does not depend on the readout.
    """
    fidelity_dict = node.machine.qubit_pairs[qp_name].macros[node.parameters.operation].fidelity
    return {
        "alpha": fidelity_dict.get("StandardRB_alpha"),
        "fidelity": fidelity_dict.get("StandardRB"),
        "load_id": fidelity_dict.get("StandardRB_load_id"),
        "source": "StandardRB_alpha (run 37a_two_qubit_standard_rb first)",
    }


def _try_load_standard_rb_overlay(node: QualibrationNode, qp_name: str) -> dict | None:
    """Load saved 37a ``ds_fit`` for interleaved overlay plots (no refit)."""
    standard_rb_load_id = _standard_rb_reference(node, qp_name)["load_id"]
    if standard_rb_load_id is None:
        return None

    from qualibrate.core.utils.node.content import read_node_data
    from qualibrate.core.utils.node.path_solver import get_node_dir_path
    from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path

    try:
        base_path = get_qualibrate_config(get_qualibrate_config_path()).storage.location
        node_dir = get_node_dir_path(int(standard_rb_load_id), base_path)
        std_rb_data = read_node_data(node_dir, int(standard_rb_load_id), base_path)
    except Exception as exc:
        node.log(f"Could not load StandardRB overlay for {qp_name}: failed to read saved node data ({exc}).")
        return None

    if std_rb_data is None:
        return None

    ds_fit = std_rb_data.get("ds_fit")
    if ds_fit is None or qp_name not in ds_fit.qubit_pair.values:
        node.log(
            f"Could not load StandardRB overlay for {qp_name}: "
            f"saved 37a run {standard_rb_load_id} has no ds_fit for this pair."
        )
        return None

    fr = ds_fit.sel(qubit_pair=qp_name)
    if "survival_probability" not in fr or "fitted_curve" not in fr or "fit_alpha" not in fr:
        node.log(
            f"Could not load StandardRB overlay for {qp_name}: "
            f"saved 37a ds_fit is missing survival or fit variables."
        )
        return None

    alpha_stderr = np.nan
    if "alpha_stderr" in fr and np.isfinite(fr.alpha_stderr.values):
        alpha_stderr = float(fr.alpha_stderr.values)

    epc_reference = None
    if "epc" in fr and np.isfinite(fr.epc.values):
        epc_reference = float(fr.epc.values)

    survival_per_sequence = fr.survival_per_sequence if "survival_per_sequence" in fr else None

    return {
        "circuit_depth": np.asarray(fr.circuit_depth.values, dtype=float),
        "survival": np.asarray(fr.survival_probability.values, dtype=float),
        "survival_per_sequence": survival_per_sequence,
        "fitted_curve": np.asarray(fr.fitted_curve.values, dtype=float),
        "alpha": float(fr.fit_alpha.values),
        "alpha_stderr": alpha_stderr,
        "epc_reference": epc_reference,
    }


def _resolve_irb_reference(
    node: QualibrationNode, qp_name: str, overlay: dict | None
) -> tuple[float, float | None, float | None]:
    """Resolve reference Standard RB alpha and EPC for interleaved analysis.

    Prefer saved 37a ``ds_fit`` overlay; fall back to QUAM (see :func:`_standard_rb_reference`).
    """
    if overlay is not None:
        alpha_stderr = overlay.get("alpha_stderr")
        stderr = (
            float(alpha_stderr) if alpha_stderr is not None and np.isfinite(alpha_stderr) and alpha_stderr > 0 else None
        )
        return float(overlay["alpha"]), stderr, overlay.get("epc_reference")

    reference = _standard_rb_reference(node, qp_name)
    if reference["alpha"] is None:
        raise KeyError(
            f"Qubit pair {qp_name}: missing reference in macros[{node.parameters.operation!r}].fidelity: "
            f"{reference['source']} for this operation."
        )
    epc_reference = 1 - float(reference["fidelity"]) if reference["fidelity"] is not None else None
    return float(reference["alpha"]), None, epc_reference


def _assign_standard_rb_overlay(
    da: xr.Dataset,
    overlay: dict,
    *,
    node: QualibrationNode | None = None,
    qp_name: str | None = None,
) -> xr.Dataset:
    """Align a Standard RB overlay onto the interleaved dataset circuit depths."""
    circuit_depths = da.circuit_depth.values
    survival_on_depths = np.full(len(circuit_depths), np.nan)
    fitted_on_depths = np.full(len(circuit_depths), np.nan)
    overlay_depths = overlay["circuit_depth"]
    overlay_per_sequence = overlay.get("survival_per_sequence")
    n_sequence = da.sizes.get("sequence")
    per_sequence_on_depths = None
    if overlay_per_sequence is not None and n_sequence is not None:
        overlay_n_sequence = overlay_per_sequence.sizes.get("sequence")
        if (
            overlay_n_sequence == n_sequence
        ):  # if the number of sequences is the same, then we can use the per-sequence data
            per_sequence_on_depths = np.full((len(circuit_depths), n_sequence), np.nan)
        elif node is not None:
            node.log(
                f"Standard RB overlay"
                f"{f' for {qp_name}' if qp_name else ''}: skipping per-sequence points "
                f"(saved 37a num_circuits_per_depth={overlay_n_sequence} != "
                f"current num_circuits_per_depth={n_sequence}). Depth-mean overlay kept."
            )

    for idx, depth in enumerate(circuit_depths):
        match = np.where(overlay_depths == depth)[0]
        if match.size:
            survival_on_depths[idx] = overlay["survival"][match[0]]
            fitted_on_depths[idx] = overlay["fitted_curve"][match[0]]
            if per_sequence_on_depths is not None:
                per_sequence_on_depths[idx, :] = overlay_per_sequence.isel(circuit_depth=match[0]).values

    assign_kwargs = {
        "standard_rb_overlay_survival": ("circuit_depth", survival_on_depths),
        "standard_rb_overlay_fitted": ("circuit_depth", fitted_on_depths),
        "standard_rb_fit_alpha": float(overlay["alpha"]),
        "standard_rb_fit_alpha_stderr": float(overlay.get("alpha_stderr", np.nan)),
    }
    if per_sequence_on_depths is not None:
        assign_kwargs["standard_rb_overlay_survival_per_sequence"] = (
            ["circuit_depth", "sequence"],
            per_sequence_on_depths,
        )

    return da.assign(**assign_kwargs)


def fit_srb_pair(da: xr.Dataset, average_gates_per_clifford: float | None) -> xr.Dataset:
    """Fit Standard RB survival probability vs circuit depth for one qubit pair."""
    stats = _fit_survival_for_pair(da)
    circuit_depths, survival_vals, stderr_vals = stats["circuit_depths"], stats["survival_vals"], stats["stderr_vals"]
    fit_params = stats["fit_params"]

    if fit_params is None:
        fit_amplitude = alpha = fit_offset = alpha_stderr = amplitude_stderr = offset_stderr = np.nan
        srb = fidelity.SRBFidelity(
            fidelity=np.nan,
            fidelity_stderr=None,
            epc=np.nan,
            epc_stderr=None,
            epg=None,
            epg_stderr=None,
            average_gate_fidelity=None,
        )
        success, fit_issues = False, ("Exponential fit did not converge.",)
    else:
        fit_amplitude, alpha, fit_offset, alpha_stderr, amplitude_stderr, offset_stderr = fit_params
        srb = fidelity.compute_srb_fidelity(alpha, alpha_stderr, average_gates_per_clifford, n_qubits=2)
        success, fit_issues = _validate_rb_fit(
            circuit_depths,
            survival_vals,
            stderr_vals,
            fit_amplitude,
            alpha,
            fit_offset,
            interleaved=False,
            fit_fidelity=srb.fidelity,
            fit_fidelity_stderr=srb.fidelity_stderr,
        )

    fitted_curve_da = _fitted_curve_dataarray(circuit_depths, fit_params)

    assign_kwargs = {
        "survival_probability": stats["survival"],
        "survival_per_sequence": stats["survival_per_sequence"],
        "survival_stderr": stats["stderr"],
        "fitted_curve": fitted_curve_da,
        "fit_amplitude": fit_amplitude,
        "fit_alpha": alpha,
        "fit_offset": fit_offset,
        "alpha_stderr": alpha_stderr,
        "fit_amplitude_stderr": amplitude_stderr,
        "fit_offset_stderr": offset_stderr,
        "fidelity": srb.fidelity,
        "fidelity_stderr": srb.fidelity_stderr if srb.fidelity_stderr is not None else np.nan,
        "epc": srb.epc,
        "epc_stderr": srb.epc_stderr if srb.epc_stderr is not None else np.nan,
        "epg": srb.epg if srb.epg is not None else np.nan,
        "epg_stderr": srb.epg_stderr if srb.epg_stderr is not None else np.nan,
        "success": success,
        "fit_issues": "\n".join(fit_issues),
        "fit_warnings": "\n".join(stats["fit_warnings"]),
    }
    if srb.average_gate_fidelity is not None:
        assign_kwargs["average_gate_fidelity"] = srb.average_gate_fidelity
    if average_gates_per_clifford is not None:
        assign_kwargs["average_gates_per_clifford"] = average_gates_per_clifford

    return da.assign(**assign_kwargs)


def fit_irb_pair(da: xr.Dataset, node: QualibrationNode, qp_name: str) -> xr.Dataset:
    """Fit Interleaved RB survival probability vs circuit depth for one qubit pair."""
    overlay = _try_load_standard_rb_overlay(node, qp_name)
    standard_rb_alpha, standard_rb_alpha_stderr, epc_reference = _resolve_irb_reference(node, qp_name, overlay)

    stats = _fit_survival_for_pair(da)
    circuit_depths, survival_vals, stderr_vals = stats["circuit_depths"], stats["survival_vals"], stats["stderr_vals"]
    fit_params = stats["fit_params"]

    if fit_params is None:
        fit_amplitude = alpha = fit_offset = alpha_stderr = amplitude_stderr = offset_stderr = np.nan
        irb = fidelity.IRBFidelity(
            fidelity=np.nan, fidelity_stderr=None, epc=np.nan, epc_stderr=None, epg=np.nan, epg_stderr=None
        )
        success, fit_issues = False, ("Exponential fit did not converge.",)
    else:
        fit_amplitude, alpha, fit_offset, alpha_stderr, amplitude_stderr, offset_stderr = fit_params
        irb = fidelity.compute_irb_fidelity(
            alpha,
            alpha_stderr,
            standard_rb_alpha,
            n_qubits=2,
            standard_rb_alpha_stderr=standard_rb_alpha_stderr,
            epc_reference=epc_reference,
        )
        success, fit_issues = _validate_rb_fit(
            circuit_depths,
            survival_vals,
            stderr_vals,
            fit_amplitude,
            alpha,
            fit_offset,
            interleaved=True,
            fit_fidelity=irb.fidelity,
            fit_fidelity_stderr=irb.fidelity_stderr,
            standard_rb_alpha=standard_rb_alpha,
        )

    fitted_curve_da = _fitted_curve_dataarray(circuit_depths, fit_params)

    da = da.assign(
        survival_probability=stats["survival"],
        survival_per_sequence=stats["survival_per_sequence"],
        survival_stderr=stats["stderr"],
        fitted_curve=fitted_curve_da,
        fit_amplitude=fit_amplitude,
        fit_alpha=alpha,
        fit_offset=fit_offset,
        alpha_stderr=alpha_stderr,
        fit_amplitude_stderr=amplitude_stderr,
        fit_offset_stderr=offset_stderr,
        fidelity=irb.fidelity,
        fidelity_stderr=irb.fidelity_stderr if irb.fidelity_stderr is not None else np.nan,
        epc=irb.epc,
        epc_stderr=irb.epc_stderr if irb.epc_stderr is not None else np.nan,
        epg=irb.epg,
        epg_stderr=irb.epg_stderr if irb.epg_stderr is not None else np.nan,
        success=success,
        fit_issues="\n".join(fit_issues),
        fit_warnings="\n".join(stats["fit_warnings"]),
        standard_rb_alpha=standard_rb_alpha,
        standard_rb_fit_alpha_stderr=standard_rb_alpha_stderr if standard_rb_alpha_stderr is not None else np.nan,
    )

    if overlay is not None:
        da = _assign_standard_rb_overlay(da, overlay, node=node, qp_name=qp_name)
    else:
        nan_overlay = np.full(len(circuit_depths), np.nan)
        da = da.assign(
            standard_rb_overlay_survival=("circuit_depth", nan_overlay),
            standard_rb_overlay_fitted=("circuit_depth", nan_overlay.copy()),
            standard_rb_fit_alpha=np.nan,
            standard_rb_fit_alpha_stderr=np.nan,
        )
    return da


# --- Leakage RB (measure_leakage=True) -------------------------------------------------------
#
# The leakage model, its fit and the per-gate subtraction are shared with the single-qubit RB
# (``calibration_utils.common_utils.rb``); this section reads the two-qubit g/e/f states.


_LEAKAGE_SERIES = ("pair", "control", "target")  # either qubit in f, control in f, target in f


def _leaked_populations(states, confusion_control=None, confusion_target=None) -> dict[str, np.ndarray]:
    """Leaked populations from per-shot states ``3 * control + target`` (shots on the last axis).

    The joint 9-outcome distribution is corrected with the Kronecker product of the two confusion
    matrices (rows prepared; identity for a missing one), see
    :func:`~calibration_utils.common_utils.rb.correct_populations`.

    Returns:
        ``pair`` (either qubit in f), ``control`` and ``target`` (that qubit in f).
    """
    states = np.asarray(states)
    measured = np.stack([(states == k).mean(axis=-1) for k in range(9)], axis=-1)
    measured = correct_populations(measured, [confusion_control, confusion_target])
    joint = measured.reshape(*measured.shape[:-1], 3, 3)
    control, target = joint[..., 2, :].sum(axis=-1), joint[..., :, 2].sum(axis=-1)
    return {"pair": control + target - joint[..., 2, 2], "control": control, "target": target}


def _resolve_leakage_reference(node: QualibrationNode, qp_name: str) -> dict:
    """The 37a leakage reference (``macros[operation].extras["leakage_per_2q_clifford"]``) for this pair, or ``{}``."""
    fidelity_dict = node.machine.qubit_pairs[qp_name].macros[node.parameters.operation].fidelity
    extras = getattr(node.machine.qubit_pairs[qp_name].macros[node.parameters.operation], "extras", None) or {}
    return extras.get("leakage_per_2q_clifford") or {}


# Names of the per-CZ variables in ``ds_leakage`` for each key of ``leakage_per_gate``.
_PER_CZ_VARIABLES = {
    "l1": "l1_per_cz",
    "l1_stderr": "l1_per_cz_stderr",
    "l2": "l2_per_cz",
    "l2_stderr": "l2_per_cz_stderr",
}


def fit_leakage_pair(da: xr.Dataset, node: QualibrationNode, qp_name: str, *, interleaved: bool) -> xr.Dataset:
    """Fit leaked population vs circuit depth for one qubit pair (``measure_leakage=True``).

    ``state`` holds ``3 * control + target`` per shot with each qubit read as g, e or f. Three
    series are fitted (``leakage_series``: either qubit in f, control in f, target in f). The pair
    fails, so nothing is written to the state, if either qubit's leaked population at the
    shallowest depth is above ``SHALLOW_LEAKAGE_THRESHOLD``: there the qubits are only reset and
    read, so that is the g/e/f readout or reset failing, not leakage. For the interleaved node the
    per-CZ leakage is taken against ``macros[operation].extras["leakage_per_2q_clifford"]`` from 37a.
    """
    qp = next(qp for qp in node.namespace["qubit_pairs"] if qp.name == qp_name)
    state = da.state.squeeze("qubit_pair", drop=True) if "qubit_pair" in da.state.dims else da.state
    states = state.transpose("circuit_depth", "sequence", "shots").values
    depths = np.asarray(state.circuit_depth.values, dtype=float)
    names = {"control": qp.qubit_control.name, "target": qp.qubit_target.name}
    confusion = {"control": read_confusion_matrix(qp.qubit_control), "target": read_confusion_matrix(qp.qubit_target)}
    leaked = _leaked_populations(states, confusion["control"], confusion["target"])

    means, stderrs, fits = [], [], []
    for series in _LEAKAGE_SERIES:
        mean, stderr = mean_and_stderr_over_sequences(leaked[series], shots_per_sequence=states.shape[-1])
        means.append(mean)
        stderrs.append(stderr)
        fits.append(fit_leakage(depths, mean, stderr))

    shallow = int(np.argmin(depths))
    suspect = [
        names[series]
        for series in ("control", "target")
        if means[_LEAKAGE_SERIES.index(series)][shallow] > SHALLOW_LEAKAGE_THRESHOLD
    ]
    failure = ""
    if suspect:
        failure = (
            f"Leaked population at depth {depths[shallow]:g} is above {SHALLOW_LEAKAGE_THRESHOLD:g} "
            f"for {', '.join(suspect)}: the g/e/f readout or reset is suspect, so nothing is written "
            "to the state. Recalibrate the g/e/f readout (15_iq_blobs_gef) and rerun."
        )

    series_coords = {"leakage_series": list(_LEAKAGE_SERIES)}
    curve_coords = {**series_coords, "circuit_depth": state.circuit_depth.values}

    def per_series(values) -> xr.DataArray:
        return xr.DataArray(np.asarray(values), dims=["leakage_series"], coords=series_coords)

    def per_series_curve(values) -> xr.DataArray:
        return xr.DataArray(np.asarray(values), dims=["leakage_series", "circuit_depth"], coords=curve_coords)

    assign_kwargs = {
        "leaked": per_series_curve(means),
        "leaked_stderr": per_series_curve(stderrs),
        "leaked_fit": per_series_curve([fit.curve(depths) for fit in fits]),
        "leakage_l1": per_series([fit.l1 for fit in fits]),
        "leakage_l1_stderr": per_series([fit.l1_stderr for fit in fits]),
        "leakage_l2": per_series([fit.l2 for fit in fits]),
        "leakage_l2_stderr": per_series([fit.l2_stderr for fit in fits]),
        "leakage_p_inf": per_series([fit.p_inf for fit in fits]),
        "leakage_lam": per_series([fit.lam for fit in fits]),
        "leakage_p0": per_series([fit.p0 for fit in fits]),
        "leakage_method": per_series([fit.method for fit in fits]),
        "leakage_fit_success": per_series([fit.success for fit in fits]),
        "leakage_fit_issues": per_series(["\n".join(fit.fit_issues) for fit in fits]),
        "leakage_fit_warnings": per_series(["\n".join(fit.fit_warnings) for fit in fits]),
        "control_qubit": names["control"],
        "target_qubit": names["target"],
        "confusion_corrected_control": confusion["control"] is not None,
        "confusion_corrected_target": confusion["target"] is not None,
        "success": fits[0].success and not suspect,
        "failure": failure,
    }
    if interleaved:
        reference = _resolve_leakage_reference(node, qp_name)
        per_cz = leakage_per_gate(fits, [reference.get(series) for series in _LEAKAGE_SERIES])
        assign_kwargs.update({_PER_CZ_VARIABLES[key]: per_series(values) for key, values in per_cz.items()})
        assign_kwargs["reference_updated_at"] = str(reference.get("updated_at") or "")
        assign_kwargs["per_cz_note"] = (
            ""
            if reference.get("pair")
            else f"No extras['leakage_per_2q_clifford'] for {qp_name}: run 37a_two_qubit_standard_rb with "
            "measure_leakage=True first. Only the leakage per interleaved Clifford is reported."
        )
    return da.drop_vars("state").assign(**assign_kwargs)


__all__ = [
    "rb_decay_curve",
    "fit_srb_pair",
    "fit_irb_pair",
    "leakage_decay_curve",
    "fit_leakage_pair",
]

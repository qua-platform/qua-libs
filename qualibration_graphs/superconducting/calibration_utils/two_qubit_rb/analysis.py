"""Analysis utilities for two-qubit randomized benchmarking experiments.

Per-qubit-pair fitting lives in ``fit_utils.py``. Alpha -> fidelity conversion
lives in ``fidelity.py``. Result types and logging live in ``reporting.py``.

"""

from __future__ import annotations

import enum
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.two_qubit_rb import fidelity, fit_utils
from calibration_utils.two_qubit_rb.coherence_limit import try_coherence_limit_epg
from calibration_utils.two_qubit_rb.reporting import (
    IRBFitResult,
    SRBFitResult,
)


class RBMode(enum.Enum):
    """Which RB protocol produced a dataset — passed explicitly by the caller."""

    STANDARD = "standard"
    INTERLEAVED = "interleaved"


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode | None = None) -> xr.Dataset:
    """Normalize raw RB dataset layout for downstream analysis."""
    if node is not None and node.parameters.use_input_stream:
        for name in ds.data_vars:
            dims = list(ds[name].dims)
            if {"circuit_depth", "shots", "sequence"}.issubset(dims):
                other_dims = [d for d in dims if d not in ("circuit_depth", "shots", "sequence")]
                ds[name] = ds[name].transpose(*other_dims, "shots", "circuit_depth", "sequence")
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode, *, mode: RBMode
) -> Tuple[xr.Dataset, Dict[str, SRBFitResult] | Dict[str, IRBFitResult]]:
    """Fit RB survival curves for each qubit pair and return an augmented dataset."""
    if mode is RBMode.STANDARD:
        average_gates_per_clifford = node.namespace.get("average_gates_per_clifford")
        ds_fit = ds.groupby("qubit_pair").apply(lambda da: fit_utils.fit_srb_pair(da, average_gates_per_clifford))
        ds_fit, fit_results = _extract_srb_results(ds_fit, node)
    else:
        ds_fit = ds.groupby("qubit_pair").apply(
            lambda da: fit_utils.fit_irb_pair(da, node, str(np.asarray(da.qubit_pair.values).item()))
        )
        ds_fit, fit_results = _extract_irb_results(ds_fit, node)

    ds_fit = _attach_shared_coherence_limits(ds_fit, node)
    return ds_fit, fit_results


def _annotate_shared_attrs(ds_fit: xr.Dataset) -> None:
    """Attach display attrs shared by both modes (mutates in place)."""
    attrs_by_var = {
        "survival_probability": {"long_name": "P(|00>)", "units": "a.u."},
        "survival_per_sequence": {"long_name": "P(|00>) per random sequence", "units": "a.u."},
        "fitted_curve": {"long_name": "exponential RB fit", "units": "a.u."},
        "fidelity": {"long_name": "RB fidelity", "units": "a.u."},
        "epc": {"long_name": "error per Clifford", "units": "a.u."},
        "epg": {"long_name": "error per gate", "units": "a.u."},
        "fit_alpha": {"long_name": "RB decay constant alpha", "units": "a.u."},
    }
    for var, attrs in attrs_by_var.items():
        if var in ds_fit.data_vars:
            ds_fit[var].attrs = attrs


def _attach_shared_coherence_limits(ds_fit: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    qubit_pairs = node.namespace["qubit_pairs"]
    coherence_limits = [try_coherence_limit_epg(qp, node.parameters.operation) or np.nan for qp in qubit_pairs]
    ds_fit = ds_fit.assign(coherence_limit_epg=("qubit_pair", np.asarray(coherence_limits, dtype=float)))
    ds_fit.coherence_limit_epg.attrs = {"long_name": "coherence-limited EPG", "units": "a.u."}
    return ds_fit


def _extract_srb_results(ds_fit: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, SRBFitResult]]:
    """Build ``SRBFitResult`` per qubit pair, including the optional cKay implied-CZ check."""
    _annotate_shared_attrs(ds_fit)
    qubit_pairs = node.namespace["qubit_pairs"]
    n_1q = node.namespace.get("avg_1q_per_clifford")
    n_cz = node.namespace.get("avg_cz_per_clifford")

    fit_results: Dict[str, SRBFitResult] = {}
    for qp in qubit_pairs:
        qp_name = qp.name
        qp_data = ds_fit.sel(qubit_pair=qp_name)

        alpha = float(qp_data.fit_alpha.values)
        alpha_stderr = float(qp_data.alpha_stderr.values) if "alpha_stderr" in qp_data else None
        fidelity_stderr = (
            float(qp_data.fidelity_stderr.values)
            if "fidelity_stderr" in qp_data and np.isfinite(qp_data.fidelity_stderr.values)
            else None
        )

        implied_cz = fidelity.compute_implied_cz(
            alpha_2q=alpha,
            alpha_2q_stderr=alpha_stderr,
            qubit_control=qp.qubit_control,
            qubit_target=qp.qubit_target,
            n_1=n_1q,
            n_2=n_cz,
        )

        fit_amplitude_stderr = None
        if "fit_amplitude_stderr" in qp_data:
            stderr = float(qp_data.fit_amplitude_stderr.values)
            if np.isfinite(stderr) and stderr > 0:
                fit_amplitude_stderr = stderr
        fit_offset_stderr = None
        if "fit_offset_stderr" in qp_data:
            stderr = float(qp_data.fit_offset_stderr.values)
            if np.isfinite(stderr) and stderr > 0:
                fit_offset_stderr = stderr

        fit_results[qp_name] = SRBFitResult(
            alpha=alpha,
            alpha_stderr=alpha_stderr,
            fidelity=float(qp_data.fidelity.values),
            fidelity_stderr=fidelity_stderr,
            fit_amplitude=float(qp_data.fit_amplitude.values),
            fit_amplitude_stderr=fit_amplitude_stderr,
            fit_offset=float(qp_data.fit_offset.values),
            fit_offset_stderr=fit_offset_stderr,
            success=bool(qp_data.success.values),
            fit_issues=_split_lines(qp_data, "fit_issues"),
            fit_warnings=_split_lines(qp_data, "fit_warnings"),
            coherence_limit_epg=try_coherence_limit_epg(qp, node.parameters.operation),
            epc=float(qp_data.epc.values) if "epc" in qp_data else None,
            epg=float(qp_data.epg.values) if "epg" in qp_data else None,
            average_gate_fidelity=(
                float(qp_data.average_gate_fidelity.values) if "average_gate_fidelity" in qp_data else None
            ),
            average_gates_per_clifford=(
                float(qp_data.average_gates_per_clifford.values) if "average_gates_per_clifford" in qp_data else None
            ),
            implied_cz=implied_cz,
        )

    return ds_fit, fit_results


def _extract_irb_results(ds_fit: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, IRBFitResult]]:
    """Build ``IRBFitResult`` per qubit pair."""
    _annotate_shared_attrs(ds_fit)
    qubit_pairs = node.namespace["qubit_pairs"]

    fit_results: Dict[str, IRBFitResult] = {}
    for qp in qubit_pairs:
        qp_name = qp.name
        qp_data = ds_fit.sel(qubit_pair=qp_name)

        fidelity_stderr = (
            float(qp_data.fidelity_stderr.values)
            if "fidelity_stderr" in qp_data and np.isfinite(qp_data.fidelity_stderr.values)
            else None
        )
        standard_rb_alpha_stderr = None
        if "standard_rb_fit_alpha_stderr" in qp_data:
            stderr = float(qp_data.standard_rb_fit_alpha_stderr.values)
            if np.isfinite(stderr) and stderr > 0:
                standard_rb_alpha_stderr = stderr

        fit_amplitude_stderr = None
        if "fit_amplitude_stderr" in qp_data:
            stderr = float(qp_data.fit_amplitude_stderr.values)
            if np.isfinite(stderr) and stderr > 0:
                fit_amplitude_stderr = stderr
        fit_offset_stderr = None
        if "fit_offset_stderr" in qp_data:
            stderr = float(qp_data.fit_offset_stderr.values)
            if np.isfinite(stderr) and stderr > 0:
                fit_offset_stderr = stderr

        fit_results[qp_name] = IRBFitResult(
            alpha=float(qp_data.fit_alpha.values),
            alpha_stderr=(float(qp_data.alpha_stderr.values) if "alpha_stderr" in qp_data else None),
            fidelity=float(qp_data.fidelity.values),
            fidelity_stderr=fidelity_stderr,
            fit_amplitude=float(qp_data.fit_amplitude.values),
            fit_amplitude_stderr=fit_amplitude_stderr,
            fit_offset=float(qp_data.fit_offset.values),
            fit_offset_stderr=fit_offset_stderr,
            success=bool(qp_data.success.values),
            fit_issues=_split_lines(qp_data, "fit_issues"),
            fit_warnings=_split_lines(qp_data, "fit_warnings"),
            coherence_limit_epg=try_coherence_limit_epg(qp, node.parameters.operation),
            epc=float(qp_data.epc.values) if "epc" in qp_data else None,
            epg=float(qp_data.epg.values) if "epg" in qp_data else None,
            epg_stderr=fidelity_stderr,
            standard_rb_alpha=(float(qp_data.standard_rb_alpha.values) if "standard_rb_alpha" in qp_data else None),
            standard_rb_alpha_stderr=standard_rb_alpha_stderr,
        )

    return ds_fit, fit_results


def _split_lines(qp_data: xr.Dataset, var: str) -> tuple[str, ...]:
    if var not in qp_data:
        return ()
    return tuple(line for line in str(qp_data[var].values).split("\n") if line)

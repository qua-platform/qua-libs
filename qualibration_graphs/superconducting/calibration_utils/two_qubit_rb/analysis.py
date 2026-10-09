"""Analysis utilities for two-qubit randomized benchmarking experiments.

Per-qubit-pair fitting lives in ``fit_utils.py``. Alpha -> fidelity conversion
lives in ``fidelity.py``. Result types and logging live in ``reporting.py``.

Standard / interleaved 37a/37b use analog-XY transpilation
``{cz, sx, x, ry, y}`` (encoding v3). Error per Clifford (EPC) is the RB
observable. Error per gate (EPG) divides by the physical-gate count, which
grew when former virtual-Z layers became analog Y — do not compare EPG to a
ZX-basis (``{rz, sx, x, cz}``) 37a run.

"""

from __future__ import annotations

import enum
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.two_qubit_rb import fidelity, fit_utils
from calibration_utils.two_qubit_rb.coherence_limit import try_coherence_limit_epg
from calibration_utils.two_qubit_rb.packing import RB_ENCODING_VERSION
from calibration_utils.two_qubit_rb.parameters import (
    CANONICAL_ANALYSIS_DIMS,
    DECLARED_RAW_DIMS,
    STREAMED_RAW_DIMS,
)
from calibration_utils.two_qubit_rb.rb_cache import DEFAULT_RB_BASIS_GATES
from calibration_utils.two_qubit_rb.reporting import (
    IRBFitResult,
    SRBFitResult,
)

# Dataset attr for new acquisitions. Bump together with :data:`RB_ENCODING_VERSION`
# when the science-result layout or gate-only executor contract changes.
RB_EXECUTION_FORMAT_VERSION = 2

IRB_RERUN_37A_MESSAGE = (
    "Interleaved CZ RB (37b) must use a Standard RB (37a) reference acquired with "
    "the same analog-XY encoding (v3: basis {cz, sx, x, ry, y}, gate-only circuits, "
    "readout outside the unsafe switch). Rerun 37a_two_qubit_standard_rb before "
    "trusting CZ fidelity if the saved overlay is ZX-basis (virtual Z / encoding v2) "
    "or otherwise predates this format. EPG is not comparable to a pre-XY 37a run "
    "(former virtual Z now counts as a physical Y pulse); EPC remains the RB "
    "observable."
)


class RBMode(enum.Enum):
    """Which RB protocol produced a dataset — passed explicitly by the caller."""

    STANDARD = "standard"
    INTERLEAVED = "interleaved"


def stamp_execution_format(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Record encoding/execution version and raw acquisition order on a new dataset."""
    use_input_stream = bool(node.parameters.use_input_stream)
    ds.attrs["rb_execution_format_version"] = RB_EXECUTION_FORMAT_VERSION
    ds.attrs["rb_encoding_version"] = RB_ENCODING_VERSION
    ds.attrs["rb_basis_gates"] = list(DEFAULT_RB_BASIS_GATES)
    ds.attrs["rb_raw_acquisition_order"] = list(STREAMED_RAW_DIMS if use_input_stream else DECLARED_RAW_DIMS)
    ds.attrs["rb_canonical_order"] = list(CANONICAL_ANALYSIS_DIMS)
    ds.attrs["rb_use_input_stream"] = use_input_stream
    return ds


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode | None = None) -> xr.Dataset:
    """Normalize raw RB dataset layout for downstream analysis.

    Streamed fetches arrive as ``circuit_depth, sequence, shots``. Named
    transposition yields canonical ``shots, circuit_depth, sequence`` for both
    modes. Dimension names, not position, drive the transpose.
    """
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
    if mode is RBMode.INTERLEAVED:
        node.log(IRB_RERUN_37A_MESSAGE)
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
        "survival_probability": {"long_name": "P(|00>)"},
        "survival_per_sequence": {"long_name": "P(|00>) per random sequence"},
        "fitted_curve": {"long_name": "exponential RB fit"},
        "fidelity": {"long_name": "RB fidelity"},
        "epc": {"long_name": "error per Clifford"},
        # Analog-XY denominator; not comparable to ZX-basis (virtual-Z) 37a EPG.
        "epg": {"long_name": "error per gate (analog XY basis)"},
        "fit_alpha": {"long_name": "RB decay constant alpha"},
    }
    for var, attrs in attrs_by_var.items():
        if var in ds_fit.data_vars:
            ds_fit[var].attrs = attrs


def _attach_shared_coherence_limits(ds_fit: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    qubit_pairs = node.namespace["qubit_pairs"]
    coherence_limits = [try_coherence_limit_epg(qp, node.parameters.operation) or np.nan for qp in qubit_pairs]
    ds_fit = ds_fit.assign(coherence_limit_epg=("qubit_pair", np.asarray(coherence_limits, dtype=float)))
    ds_fit.coherence_limit_epg.attrs = {"long_name": "coherence-limited EPG"}
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
            epc_stderr=(
                float(qp_data.epc_stderr.values)
                if "epc_stderr" in qp_data and np.isfinite(qp_data.epc_stderr.values)
                else None
            ),
            epg=float(qp_data.epg.values) if "epg" in qp_data else None,
            epg_stderr=(
                float(qp_data.epg_stderr.values)
                if "epg_stderr" in qp_data and np.isfinite(qp_data.epg_stderr.values)
                else None
            ),
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
            epc_stderr=(
                float(qp_data.epc_stderr.values)
                if "epc_stderr" in qp_data and np.isfinite(qp_data.epc_stderr.values)
                else None
            ),
            epg=float(qp_data.epg.values) if "epg" in qp_data else None,
            epg_stderr=(
                float(qp_data.epg_stderr.values)
                if "epg_stderr" in qp_data and np.isfinite(qp_data.epg_stderr.values)
                else None
            ),
            standard_rb_alpha=(float(qp_data.standard_rb_alpha.values) if "standard_rb_alpha" in qp_data else None),
            standard_rb_alpha_stderr=standard_rb_alpha_stderr,
        )

    return ds_fit, fit_results


def _split_lines(qp_data: xr.Dataset, var: str) -> tuple[str, ...]:
    if var not in qp_data:
        return ()
    return tuple(line for line in str(qp_data[var].values).split("\n") if line)

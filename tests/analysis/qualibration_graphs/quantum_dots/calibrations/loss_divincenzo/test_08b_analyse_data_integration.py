"""Integration test: full analyse_data trace for 08b_qubit_spectroscopy.

Covers both parity_pre_measurement modes.  Uses qubit names and larmor
frequencies from quam_machine_state/state_old.json (q1-q4, 5.25 GHz each).

Key design notes
----------------
* No QOP connection required — ds_raw is built from a synthetic Lorentzian
  peak so peaks_dips() reliably finds a fit.
* node is a SimpleNamespace stub; only the fields read by process_raw_dataset,
  fit_raw_data, and _extract_relevant_fit_parameters are populated.
* Qubit loop in fit_raw_data: peaks_dips() calls xr.apply_ufunc(...,
  vectorize=True) which automatically maps over the "qubit" dim.
  _extract_relevant_fit_parameters then uses fit.sel(qubit=q) in a dict
  comprehension over fit.qubit.values to build per-qubit FitParameters.
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from calibration_utils.common_utils.parity_streams import process_parity_streams
from calibration_utils.qubit_spectroscopy_parity_diff.analysis import (
    fit_raw_data,
    log_fitted_results,
)

# ── Constants matching quam_machine_state/state_old.json ─────────────────────────

QUBIT_NAMES = ["q1", "q2", "q3", "q4"]
LARMOR_FREQ_HZ = 5_250_000_000.0      # 5.25 GHz — same for all 4 qubits in state

# Different peak centres per qubit so we can distinguish them in assertions
PEAK_CENTERS_HZ = {"q1": 5_000_000.0, "q2": -3_000_000.0, "q3": 2_000_000.0, "q4": -7_000_000.0}
PEAK_FWHM_HZ    = 2_000_000.0
PEAK_AMPLITUDE  = 0.4

SPAN_MHZ  = 50.0
STEP_MHZ  = 0.25
DETUNINGS = np.arange(-SPAN_MHZ / 2 * 1e6, SPAN_MHZ / 2 * 1e6, STEP_MHZ * 1e6)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _lorentzian(x: np.ndarray, center: float, fwhm: float, amplitude: float) -> np.ndarray:
    return amplitude / (1.0 + 4.0 * ((x - center) / fwhm) ** 2)


def _make_qubit(name: str) -> SimpleNamespace:
    """Stub with the attributes read by process_raw_dataset and fit_raw_data."""
    xy = SimpleNamespace(RF_frequency=LARMOR_FREQ_HZ)
    return SimpleNamespace(name=name, xy=xy)


def _make_node(parity_pre_measurement: bool, analysis_signal: str = "E_p2_given_p1_0"):
    qubits = [_make_qubit(n) for n in QUBIT_NAMES]
    params = SimpleNamespace(
        parity_pre_measurement=parity_pre_measurement,
        analysis_signal=analysis_signal,
        frequency_span_in_mhz=SPAN_MHZ,
    )
    return SimpleNamespace(
        parameters=params,
        namespace={"qubits": qubits},
        results={},
        outcomes={},
        log=lambda msg: None,
    )


def _detuning_coord() -> xr.DataArray:
    return xr.DataArray(
        DETUNINGS, dims="detuning", attrs={"long_name": "drive frequency", "units": "Hz"}
    )


def _signal(qname: str) -> np.ndarray:
    """Lorentzian centred at PEAK_CENTERS_HZ[qname]."""
    return _lorentzian(DETUNINGS, PEAK_CENTERS_HZ[qname], PEAK_FWHM_HZ, PEAK_AMPLITUDE)


def _build_single_stream_ds() -> xr.Dataset:
    """ds_raw when parity_pre_measurement=False: one averaged p_{qname} per qubit."""
    return xr.Dataset(
        {f"p_{q}": xr.DataArray(_signal(q), dims="detuning") for q in QUBIT_NAMES},
        coords={"detuning": _detuning_coord()},
    )


def _build_joint_stream_ds(analysis_signal: str = "E_p2_given_p1_0") -> xr.Dataset:
    """ds_raw when parity_pre_measurement=True: four joint-outcome streams per qubit.

    We set E[p2|p1=0] = signal by construction:
        p0_p1 / (p0_p0 + p0_p1) = signal
    achieved by: p0_p1 = signal, p0_p0 = 1 - signal.
    The p1=1 branch is left flat (50 / 50) since only E_p2_given_p1_0 is used.
    """
    data_vars = {}
    for q in QUBIT_NAMES:
        sig = _signal(q)
        data_vars[f"p0_p0_{q}"] = xr.DataArray(1.0 - sig, dims="detuning")
        data_vars[f"p0_p1_{q}"] = xr.DataArray(sig,       dims="detuning")
        data_vars[f"p1_p0_{q}"] = xr.DataArray(np.full_like(sig, 0.5), dims="detuning")
        data_vars[f"p1_p1_{q}"] = xr.DataArray(np.full_like(sig, 0.5), dims="detuning")
    return xr.Dataset(data_vars, coords={"detuning": _detuning_coord()})


# ── Helper: run the full analyse_data trace ───────────────────────────────────


def _run_analyse_data(node):
    """Mirrors the body of the process_raw_data + analyse_data run_actions."""
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [q.name for q in node.namespace["qubits"]],
        parity_pre_measurement=node.parameters.parity_pre_measurement,
        sweep_dims=("detuning",),
    )
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qname: ("successful" if fr["success"] else "failed")
        for qname, fr in node.results["fit_results"].items()
    }
    return node


# ── Tests: parity_pre_measurement=False ──────────────────────────────────────


@pytest.fixture
def single_stream_node():
    node = _make_node(parity_pre_measurement=False)
    node.results["ds_raw"] = _build_single_stream_ds()
    return _run_analyse_data(node)


def test_single_stream_fit_results_populated_for_all_qubits(single_stream_node):
    assert set(single_stream_node.results["fit_results"].keys()) == set(QUBIT_NAMES)


def test_single_stream_fit_succeeds_for_all_qubits(single_stream_node):
    for q in QUBIT_NAMES:
        assert single_stream_node.results["fit_results"][q]["success"], (
            f"Fit failed for {q}: {single_stream_node.results['fit_results'][q]}"
        )


def test_single_stream_fit_finds_correct_peak_frequency(single_stream_node):
    """Fitted frequency should be within 2x the step size of the true peak centre."""
    tolerance = 2 * STEP_MHZ * 1e6
    for q in QUBIT_NAMES:
        fitted_rel = single_stream_node.results["fit_results"][q]["relative_freq"]
        expected = PEAK_CENTERS_HZ[q]
        assert abs(fitted_rel - expected) < tolerance, (
            f"{q}: expected peak at {expected/1e6:.2f} MHz, "
            f"got {fitted_rel/1e6:.2f} MHz (tol={tolerance/1e6:.2f} MHz)"
        )


def test_single_stream_outcomes_are_successful(single_stream_node):
    for q in QUBIT_NAMES:
        assert single_stream_node.outcomes[q] == "successful"


def test_single_stream_ds_fit_has_qubit_dim(single_stream_node):
    ds_fit = single_stream_node.results["ds_fit"]
    assert "qubit" in ds_fit.dims
    assert set(ds_fit.qubit.values) == set(QUBIT_NAMES)


# ── Tests: parity_pre_measurement=True ───────────────────────────────────────


@pytest.fixture
def joint_stream_node():
    node = _make_node(parity_pre_measurement=True, analysis_signal="E_p2_given_p1_0")
    node.results["ds_raw"] = _build_joint_stream_ds()
    return _run_analyse_data(node)


def test_joint_stream_fit_results_populated_for_all_qubits(joint_stream_node):
    assert set(joint_stream_node.results["fit_results"].keys()) == set(QUBIT_NAMES)


def test_joint_stream_fit_succeeds_for_all_qubits(joint_stream_node):
    for q in QUBIT_NAMES:
        assert joint_stream_node.results["fit_results"][q]["success"], (
            f"Fit failed for {q}: {joint_stream_node.results['fit_results'][q]}"
        )


def test_joint_stream_fit_finds_correct_peak_frequency(joint_stream_node):
    tolerance = 2 * STEP_MHZ * 1e6
    for q in QUBIT_NAMES:
        fitted_rel = joint_stream_node.results["fit_results"][q]["relative_freq"]
        expected = PEAK_CENTERS_HZ[q]
        assert abs(fitted_rel - expected) < tolerance, (
            f"{q}: expected peak at {expected/1e6:.2f} MHz, "
            f"got {fitted_rel/1e6:.2f} MHz (tol={tolerance/1e6:.2f} MHz)"
        )


def test_joint_stream_outcomes_are_successful(joint_stream_node):
    for q in QUBIT_NAMES:
        assert joint_stream_node.outcomes[q] == "successful"


# ── Cross-mode consistency check ─────────────────────────────────────────────


def test_both_modes_agree_on_peak_position():
    """parity_pre_measurement=True/False should find the same peak for the same signal."""
    node_single = _make_node(parity_pre_measurement=False)
    node_single.results["ds_raw"] = _build_single_stream_ds()
    _run_analyse_data(node_single)

    node_joint = _make_node(parity_pre_measurement=True)
    node_joint.results["ds_raw"] = _build_joint_stream_ds()
    _run_analyse_data(node_joint)

    for q in QUBIT_NAMES:
        rel_single = node_single.results["fit_results"][q]["relative_freq"]
        rel_joint  = node_joint.results["fit_results"][q]["relative_freq"]
        assert abs(rel_single - rel_joint) < STEP_MHZ * 1e6, (
            f"{q}: single-stream={rel_single/1e6:.3f} MHz, "
            f"joint-stream={rel_joint/1e6:.3f} MHz"
        )

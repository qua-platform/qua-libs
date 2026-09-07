"""Integration tests for the 08b analyse-data path after parity removal."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import xarray as xr

from calibration_utils.qubit_spectroscopy.analysis import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
)

# ── Constants matching quam_machine_state/state_old.json ─────────────────────────

QUBIT_NAMES = ["q1", "q2", "q3", "q4"]
LARMOR_FREQ_HZ = 5_250_000_000.0

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
    """Stub with the attributes read by fit_raw_data."""
    xy = SimpleNamespace(RF_frequency=LARMOR_FREQ_HZ)
    return SimpleNamespace(name=name, xy=xy)


def _make_node():
    qubits = [_make_qubit(n) for n in QUBIT_NAMES]
    params = SimpleNamespace(
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


def _build_state_iq_ds() -> xr.Dataset:
    state_rows = []
    i_rows = []
    q_rows = []
    for idx, qname in enumerate(QUBIT_NAMES):
        signal = _signal(qname)
        state_rows.append(signal)
        i_rows.append(0.02 * idx + 0.15 * signal)
        q_rows.append(-0.01 * idx + 0.10 * np.gradient(signal))
    return xr.Dataset(
        {
            "state": xr.DataArray(np.asarray(state_rows), dims=("qubit", "detuning")),
            "I": xr.DataArray(np.asarray(i_rows), dims=("qubit", "detuning")),
            "Q": xr.DataArray(np.asarray(q_rows), dims=("qubit", "detuning")),
        },
        coords={"qubit": QUBIT_NAMES, "detuning": _detuning_coord()},
    )


# ── Helper: run the full analyse_data trace ───────────────────────────────────
def _run_analyse_data(node):
    """Mirror the body of the node's ``analyse_data`` run_action."""
    ds_processed = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qname: ("successful" if fr["success"] else "failed")
        for qname, fr in node.results["fit_results"].items()
    }
    return node


def test_analyse_data_populates_results_for_all_qubits():
    node = _make_node()
    node.results["ds_raw"] = _build_state_iq_ds()
    analysed = _run_analyse_data(node)

    assert set(analysed.results["fit_results"].keys()) == set(QUBIT_NAMES)


def test_analyse_data_succeeds_for_all_qubits():
    node = _make_node()
    node.results["ds_raw"] = _build_state_iq_ds()
    analysed = _run_analyse_data(node)

    for q in QUBIT_NAMES:
        assert analysed.results["fit_results"][q]["success"], (
            f"Fit failed for {q}: {analysed.results['fit_results'][q]}"
        )


def test_analyse_data_finds_correct_peak_frequency():
    node = _make_node()
    node.results["ds_raw"] = _build_state_iq_ds()
    analysed = _run_analyse_data(node)

    tolerance = 2 * STEP_MHZ * 1e6
    for q in QUBIT_NAMES:
        fitted_rel = analysed.results["fit_results"][q]["relative_freq"]
        expected = PEAK_CENTERS_HZ[q]
        assert abs(fitted_rel - expected) < tolerance, (
            f"{q}: expected peak at {expected/1e6:.2f} MHz, "
            f"got {fitted_rel/1e6:.2f} MHz (tol={tolerance/1e6:.2f} MHz)"
        )


def test_analyse_data_marks_outcomes_successful():
    node = _make_node()
    node.results["ds_raw"] = _build_state_iq_ds()
    analysed = _run_analyse_data(node)

    for q in QUBIT_NAMES:
        assert analysed.outcomes[q] == "successful"


def test_analyse_data_preserves_dataset_shape_for_plotting():
    node = _make_node()
    node.results["ds_raw"] = _build_state_iq_ds()
    analysed = _run_analyse_data(node)

    ds_fit = analysed.results["ds_fit"]
    assert "qubit" in ds_fit.dims
    assert set(ds_fit.qubit.values) == set(QUBIT_NAMES)
    assert {"state", "I", "Q", "fit_curve"}.issubset(set(ds_fit.data_vars))

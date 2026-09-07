"""Regression tests for the 11x state-stream analysis and plotting paths."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr

from calibration_utils.T1.analysis import analyse_raw_data as analyse_t1
from calibration_utils.T1.plotting import plot_all as plot_t1
from calibration_utils.ramsey.analysis import analyse_raw_data as analyse_ramsey
from calibration_utils.ramsey.plotting import plot_all as plot_ramsey
from calibration_utils.ramsey_detuning.analysis import analyse_raw_data as analyse_ramsey_detuning
from calibration_utils.ramsey_detuning.plotting import plot_all as plot_ramsey_detuning
from calibration_utils.ramsey_chevron.analysis import analyse_raw_data as analyse_ramsey_chevron
from calibration_utils.ramsey_chevron.plotting import plot_all as plot_ramsey_chevron

DATASET_QUBIT_ORDER = ["q2", "q1"]
NODE_QUBIT_ORDER = ["q1", "q2"]
RF_FREQS = {"q1": 5.25e9, "q2": 5.55e9}


def _make_qubit(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, xy=SimpleNamespace(RF_frequency=RF_FREQS[name]))


def _make_node(**params) -> SimpleNamespace:
    return SimpleNamespace(
        parameters=SimpleNamespace(**params),
        namespace={"qubits": [_make_qubit(name) for name in NODE_QUBIT_ORDER]},
    )


def _ramsey_trace(
    tau_ns: np.ndarray,
    freq_hz: float,
    *,
    offset: float = 0.5,
    amplitude: float = 0.35,
    gamma: float = 0.0015,
    phase: float = 0.2,
) -> np.ndarray:
    t = np.asarray(tau_ns, dtype=float)
    return offset + amplitude * np.exp(-gamma * t) * np.cos(2.0 * np.pi * freq_hz * 1e-9 * t + phase)


def _ramsey_detuning_trace(
    detuning_hz: np.ndarray,
    delta0_hz: float,
    *,
    osc_freq_hz_inv: float,
    offset: float = 0.5,
    amplitude: float = 0.35,
) -> np.ndarray:
    delta = np.asarray(detuning_hz, dtype=float)
    return offset + amplitude * np.cos(2.0 * np.pi * osc_freq_hz_inv * (delta - delta0_hz))


def _ramsey_chevron_state(
    detuning_hz: np.ndarray,
    tau_ns: np.ndarray,
    *,
    delta0_hz: float,
    amplitude: float = 0.35,
    gamma: float = 0.002,
    sigma_g: float = 0.0008,
    offset: float = 0.5,
) -> np.ndarray:
    delta = np.asarray(detuning_hz, dtype=float)[:, None]
    tau = np.asarray(tau_ns, dtype=float)[None, :]
    envelope = np.exp(-gamma * tau - (sigma_g * tau) ** 2)
    phase = 2.0 * np.pi * (delta - delta0_hz) * tau * 1e-9
    return offset + amplitude * envelope * np.cos(phase)


def test_t1_analysis_and_plotting_follow_dataset_qubit_order():
    tau_ns = np.arange(40.0, 1240.0, 40.0)
    t1_values = {"q1": 260.0, "q2": 520.0}
    offsets = {"q1": 0.08, "q2": 0.12}
    amplitudes = {"q1": 0.75, "q2": 0.65}

    state_rows = [
        offsets[q] + amplitudes[q] * np.exp(-tau_ns / t1_values[q])
        for q in DATASET_QUBIT_ORDER
    ]
    ds = xr.Dataset(
        {"state": xr.DataArray(np.asarray(state_rows), dims=("qubit", "tau"))},
        coords={"qubit": DATASET_QUBIT_ORDER, "tau": tau_ns},
    )
    node = _make_node()

    ds_fit, fit_results, fit_results_full = analyse_t1(ds, node)
    figures = plot_t1(ds, node.namespace["qubits"], ds_fit=ds_fit, fit_results=fit_results_full, show=False)

    assert abs(fit_results["q1"]["T1"] - t1_values["q1"]) < 80.0
    assert abs(fit_results["q2"]["T1"] - t1_values["q2"]) < 120.0
    axes = figures["raw_data_with_fit"].axes
    assert axes[0].get_title().startswith("q2")
    assert axes[1].get_title().startswith("q1")


def test_ramsey_pm_delta_analysis_and_plotting_keep_qubits_unmixed():
    tau_ns = np.arange(40.0, 520.0, 40.0)
    detuning_values = np.array([4.0e6, -4.0e6])
    freq_offsets = {"q1": 0.8e6, "q2": -1.2e6}

    state_rows = []
    for qname in DATASET_QUBIT_ORDER:
        delta = freq_offsets[qname]
        f_plus = abs(delta - detuning_values[0])
        f_minus = abs(delta - detuning_values[1])
        traces = np.stack(
            [
                _ramsey_trace(tau_ns, f_plus, phase=0.15),
                _ramsey_trace(tau_ns, f_minus, phase=-0.2),
            ]
        )
        state_rows.append(traces)

    ds = xr.Dataset(
        {"state": xr.DataArray(np.asarray(state_rows), dims=("qubit", "detuning", "tau"))},
        coords={"qubit": DATASET_QUBIT_ORDER, "detuning": detuning_values, "tau": tau_ns},
    )
    node = _make_node()

    ds_fit, fit_results, fit_results_full = analyse_ramsey(ds, node)
    figures = plot_ramsey(ds, node.namespace["qubits"], ds_fit=ds_fit, fit_results=fit_results_full, show=False)

    assert abs(fit_results["q1"]["freq_offset"] - freq_offsets["q1"]) < 0.6e6
    assert abs(fit_results["q2"]["freq_offset"] - freq_offsets["q2"]) < 0.6e6
    titled_axes = [ax for ax in figures["raw_data_with_fit"].axes if ax.get_title()]
    assert titled_axes[0].get_title().startswith("q2")
    assert titled_axes[2].get_title().startswith("q1")


def test_ramsey_detuning_analysis_and_plotting_keep_qubits_unmixed():
    tau_ns = np.array([40.0, 160.0])
    detuning_hz = np.arange(-8.0e6, 8.0e6, 0.25e6)
    offsets = {"q1": 0.9e6, "q2": -1.4e6}
    osc_freq = 1.0 / 12.0e6

    state_rows = []
    for qname in DATASET_QUBIT_ORDER:
        delta0 = offsets[qname]
        traces = np.stack(
            [
                _ramsey_detuning_trace(detuning_hz, delta0, osc_freq_hz_inv=osc_freq, amplitude=0.42),
                _ramsey_detuning_trace(detuning_hz, delta0, osc_freq_hz_inv=osc_freq, amplitude=0.20),
            ]
        )
        state_rows.append(traces.T)

    ds = xr.Dataset(
        {"state": xr.DataArray(np.asarray(state_rows), dims=("qubit", "detuning", "tau"))},
        coords={"qubit": DATASET_QUBIT_ORDER, "detuning": detuning_hz, "tau": tau_ns},
    )
    node = _make_node()

    ds_fit, fit_results, fit_results_full = analyse_ramsey_detuning(ds, node)
    figures = plot_ramsey_detuning(
        ds,
        node.namespace["qubits"],
        ds_fit=ds_fit,
        fit_results=fit_results_full,
        show=False,
    )

    assert abs(fit_results["q1"]["freq_offset"] - offsets["q1"]) < 0.5e6
    assert abs(fit_results["q2"]["freq_offset"] - offsets["q2"]) < 0.5e6
    titled_axes = [ax for ax in figures["raw_data_with_fit"].axes if ax.get_title()]
    assert titled_axes[0].get_title().startswith("q2")
    assert titled_axes[2].get_title().startswith("q1")


def test_ramsey_chevron_analysis_and_plotting_keep_qubits_unmixed():
    tau_ns = np.arange(20.0, 220.0, 20.0)
    detuning_hz = np.arange(-8.0e6, 8.0e6, 0.5e6)
    offsets = {"q1": 0.6e6, "q2": -1.0e6}

    state_rows = [
        _ramsey_chevron_state(detuning_hz, tau_ns, delta0_hz=offsets[qname])
        for qname in DATASET_QUBIT_ORDER
    ]
    ds = xr.Dataset(
        {"state": xr.DataArray(np.asarray(state_rows), dims=("qubit", "detuning", "tau"))},
        coords={"qubit": DATASET_QUBIT_ORDER, "detuning": detuning_hz, "tau": tau_ns},
    )
    node = _make_node()

    ds_fit, fit_results, fit_results_full = analyse_ramsey_chevron(ds, node)
    figures = plot_ramsey_chevron(
        ds,
        node.namespace["qubits"],
        ds_fit=ds_fit,
        fit_results=fit_results_full,
        show=False,
    )

    assert abs(fit_results["q1"]["freq_offset"] - offsets["q1"]) < 0.75e6
    assert abs(fit_results["q2"]["freq_offset"] - offsets["q2"]) < 0.75e6
    titled_axes = [ax for ax in figures["raw_data_with_fit"].axes if ax.get_title()]
    assert titled_axes[0].get_title().startswith("q2")
    assert titled_axes[2].get_title().startswith("q1")

"""Regression tests for multi-qubit ordering in ``08b_qubit_spectroscopy``."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr

from calibration_utils.qubit_spectroscopy.analysis import (
    fit_raw_data,
    process_raw_dataset,
)
from calibration_utils.qubit_spectroscopy.plotting import plot_all

DATASET_QUBIT_ORDER = ["q4", "q1"]
NODE_QUBIT_ORDER = ["q1", "q4"]
RF_FREQS = {"q1": 5.25e9, "q4": 5.55e9}
CENTERS = {"q1": -3.0e6, "q4": 6.0e6}
DETUNINGS = np.arange(-12e6, 12e6, 0.25e6)
FWHM = 2.0e6


def _lorentzian(x: np.ndarray, center: float, fwhm: float, amplitude: float) -> np.ndarray:
    hwhm = fwhm / 2.0
    return amplitude / (1.0 + ((x - center) / hwhm) ** 2)


def _make_qubit(name: str) -> SimpleNamespace:
    xy = SimpleNamespace(RF_frequency=RF_FREQS[name])
    return SimpleNamespace(name=name, xy=xy)


def _make_node():
    qubits = [_make_qubit(name) for name in NODE_QUBIT_ORDER]
    params = SimpleNamespace(
        frequency_span_in_mhz=24.0,
    )
    return SimpleNamespace(parameters=params, namespace={"qubits": qubits})


def _build_state_iq_ds() -> xr.Dataset:
    state_rows = []
    i_rows = []
    q_rows = []
    for qname in DATASET_QUBIT_ORDER:
        state_rows.append(_lorentzian(DETUNINGS, CENTERS[qname], FWHM, 0.45 if qname == "q1" else 0.35))
        i_rows.append(_lorentzian(DETUNINGS, CENTERS[qname], FWHM, 0.12 if qname == "q1" else 0.09))
        q_rows.append(_lorentzian(DETUNINGS, CENTERS[qname], FWHM, -0.08 if qname == "q1" else -0.05))
    return xr.Dataset(
        {
            "state": xr.DataArray(state_rows, dims=("qubit", "detuning")),
            "I": xr.DataArray(i_rows, dims=("qubit", "detuning")),
            "Q": xr.DataArray(q_rows, dims=("qubit", "detuning")),
        },
        coords={
            "qubit": xr.DataArray(DATASET_QUBIT_ORDER, dims="qubit"),
            "detuning": xr.DataArray(
                DETUNINGS,
                dims="detuning",
                attrs={"long_name": "drive frequency", "units": "Hz"},
            ),
        },
    )


def test_process_raw_dataset_leaves_state_dataset_unchanged():
    node = _make_node()
    raw = _build_state_iq_ds()
    processed = process_raw_dataset(raw, node)

    assert set(processed.data_vars) == {"state", "I", "Q"}
    assert list(processed.qubit.values) == DATASET_QUBIT_ORDER


def test_plot_all_uses_dataset_qubit_order_for_raw_iq_traces():
    node = _make_node()
    ds = process_raw_dataset(_build_state_iq_ds(), node)
    ds_fit, _ = fit_raw_data(ds, node)
    figures = plot_all(ds_fit, node.namespace["qubits"], ds_fit, show=False)

    raw_iq_axes = [
        ax for ax in figures["iq_scatter"].axes
        if ax.get_title().startswith("IQ vs frequency")
    ]
    assert len(raw_iq_axes) == len(DATASET_QUBIT_ORDER)
    assert raw_iq_axes[0].get_title().endswith("q4")
    assert raw_iq_axes[1].get_title().endswith("q1")

    q4_i = raw_iq_axes[0].lines[0].get_ydata()
    q1_i = raw_iq_axes[1].lines[0].get_ydata()

    assert int(np.argmax(q1_i)) != int(np.argmax(q4_i))
    assert DETUNINGS[int(np.argmax(q1_i))] < 0
    assert DETUNINGS[int(np.argmax(q4_i))] > 0


def test_fit_results_follow_the_explicit_qubit_names():
    node = _make_node()
    ds_processed = process_raw_dataset(_build_state_iq_ds(), node)

    ds_fit, fit_results = fit_raw_data(ds_processed, node)

    figures = plot_all(ds_fit, node.namespace["qubits"], ds_fit, show=False)

    assert abs(fit_results["q1"].relative_freq - CENTERS["q1"]) < 0.5e6
    assert abs(fit_results["q4"].relative_freq - CENTERS["q4"]) < 0.5e6
    assert abs(fit_results["q1"].frequency - (RF_FREQS["q1"] + CENTERS["q1"])) < 0.5e6
    assert abs(fit_results["q4"].frequency - (RF_FREQS["q4"] + CENTERS["q4"])) < 0.5e6
    np.testing.assert_allclose(ds_fit.full_freq.sel(qubit="q1").values, DETUNINGS + RF_FREQS["q1"])
    np.testing.assert_allclose(ds_fit.full_freq.sel(qubit="q4").values, DETUNINGS + RF_FREQS["q4"])
    assert set(figures) == {"qubit_spectroscopy", "iq_scatter"}

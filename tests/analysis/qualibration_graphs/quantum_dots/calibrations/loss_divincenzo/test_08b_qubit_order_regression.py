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

QUBIT_ORDER = ["q4", "q1"]
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
    qubits = [_make_qubit(name) for name in QUBIT_ORDER]
    params = SimpleNamespace(
        parity_measurement=False,
        analysis_signal="E_p1_given_p0_0",
        frequency_span_in_mhz=24.0,
    )
    return SimpleNamespace(parameters=params, namespace={"qubits": qubits})


def _build_explicitly_named_ds() -> xr.Dataset:
    return xr.Dataset(
        {
            "p_q1_parity_diff": xr.DataArray(
                _lorentzian(DETUNINGS, CENTERS["q1"], FWHM, 0.45),
                dims="detuning",
            ),
            "p_q4_parity_diff": xr.DataArray(
                _lorentzian(DETUNINGS, CENTERS["q4"], FWHM, 0.35),
                dims="detuning",
            ),
            "I_q1_raw": xr.DataArray(
                _lorentzian(DETUNINGS, CENTERS["q1"], FWHM, 0.12),
                dims="detuning",
            ),
            "I_q4_raw": xr.DataArray(
                _lorentzian(DETUNINGS, CENTERS["q4"], FWHM, 0.09),
                dims="detuning",
            ),
            "Q_q1_raw": xr.DataArray(
                _lorentzian(DETUNINGS, CENTERS["q1"], FWHM, -0.08),
                dims="detuning",
            ),
            "Q_q4_raw": xr.DataArray(
                _lorentzian(DETUNINGS, CENTERS["q4"], FWHM, -0.05),
                dims="detuning",
            ),
        },
        coords={
            "detuning": xr.DataArray(
                DETUNINGS,
                dims="detuning",
                attrs={"long_name": "drive frequency", "units": "Hz"},
            ),
        },
    )


def test_process_raw_dataset_builds_canonical_analysis_variables():
    node = _make_node()
    processed = process_raw_dataset(_build_explicitly_named_ds(), node)

    assert "E_p1_given_p0_0_q1" in processed.data_vars
    assert "E_p1_given_p0_0_q4" in processed.data_vars


def test_plot_all_uses_the_explicit_raw_iq_stream_names():
    node = _make_node()
    ds = process_raw_dataset(_build_explicitly_named_ds(), node)
    ds_fit, _ = fit_raw_data(ds, node)
    figures = plot_all(ds, node.namespace["qubits"], ds_fit, analysis_signal=node.parameters.analysis_signal)

    raw_iq_axes = [
        ax for ax in figures["iq_scatter"].axes
        if ax.get_title().startswith("IQ vs frequency")
    ]
    assert len(raw_iq_axes) == len(QUBIT_ORDER)

    q4_i = raw_iq_axes[0].lines[0].get_ydata()
    q1_i = raw_iq_axes[1].lines[0].get_ydata()

    assert int(np.argmax(q1_i)) != int(np.argmax(q4_i))
    assert DETUNINGS[int(np.argmax(q1_i))] < 0
    assert DETUNINGS[int(np.argmax(q4_i))] > 0


def test_fit_results_follow_the_explicit_qubit_names():
    node = _make_node()
    ds_processed = process_raw_dataset(_build_explicitly_named_ds(), node)

    ds_fit, fit_results = fit_raw_data(ds_processed, node)

    figures = plot_all(ds_processed, node.namespace["qubits"], ds_fit, analysis_signal=node.parameters.analysis_signal)

    assert abs(fit_results["q1"].relative_freq - CENTERS["q1"]) < 0.5e6
    assert abs(fit_results["q4"].relative_freq - CENTERS["q4"]) < 0.5e6
    assert abs(fit_results["q1"].frequency - (RF_FREQS["q1"] + CENTERS["q1"])) < 0.5e6
    assert abs(fit_results["q4"].frequency - (RF_FREQS["q4"] + CENTERS["q4"])) < 0.5e6
    assert set(figures) == {"qubit_spectroscopy", "iq_scatter"}

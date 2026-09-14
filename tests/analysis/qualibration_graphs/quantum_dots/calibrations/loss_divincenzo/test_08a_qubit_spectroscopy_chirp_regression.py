"""Regression tests for chirped qubit spectroscopy after parity removal."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr

from calibration_utils.qubit_spectroscopy_chirp.analysis import (
    analyse_raw_data,
    fit_raw_data,
    find_frequency_by_threshold,
    process_raw_dataset,
)
from calibration_utils.qubit_spectroscopy_chirp.plotting import plot_all

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


def _make_node() -> SimpleNamespace:
    qubits = [_make_qubit(name) for name in NODE_QUBIT_ORDER]
    params = SimpleNamespace(
        signal_threshold=0.2,
        frequency_span_in_mhz=24.0,
        fit_peak=True,
    )
    return SimpleNamespace(parameters=params, namespace={"qubits": qubits})


def _build_state_ds() -> xr.Dataset:
    state_rows = []
    for qname in DATASET_QUBIT_ORDER:
        state_rows.append(0.05 + _lorentzian(DETUNINGS, CENTERS[qname], FWHM, 0.55))
    return xr.Dataset(
        {
            "state": xr.DataArray(
                np.asarray(state_rows),
                dims=("qubit", "detuning"),
            )
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


def test_process_raw_dataset_is_identity_for_state_data():
    node = _make_node()
    raw = _build_state_ds()
    processed = process_raw_dataset(raw, node)

    assert set(processed.data_vars) == {"state"}
    assert list(processed.qubit.values) == DATASET_QUBIT_ORDER


def test_threshold_and_peak_fit_follow_qubit_names():
    node = _make_node()
    ds = process_raw_dataset(_build_state_ds(), node)

    threshold_results = find_frequency_by_threshold(ds, node)
    ds_fit, peak_results = fit_raw_data(ds, node)

    assert abs(threshold_results["q1"].relative_freq - CENTERS["q1"]) < 0.75e6
    assert abs(threshold_results["q4"].relative_freq - CENTERS["q4"]) < 0.75e6
    assert abs(peak_results["q1"].frequency - (RF_FREQS["q1"] + CENTERS["q1"])) < 0.5e6
    assert abs(peak_results["q4"].frequency - (RF_FREQS["q4"] + CENTERS["q4"])) < 0.5e6
    np.testing.assert_allclose(ds_fit.full_freq.sel(qubit="q1").values, DETUNINGS + RF_FREQS["q1"])
    np.testing.assert_allclose(ds_fit.full_freq.sel(qubit="q4").values, DETUNINGS + RF_FREQS["q4"])


def test_analyse_raw_data_and_plotting_keep_dataset_order_stable():
    node = _make_node()
    ds = process_raw_dataset(_build_state_ds(), node)

    ds_fit, fit_results, peak_fit_results, outcomes = analyse_raw_data(ds, node)
    figures = plot_all(
        ds,
        node.namespace["qubits"],
        fits=ds_fit,
        threshold_results=fit_results,
        signal_threshold=node.parameters.signal_threshold,
        show=False,
    )

    assert outcomes == {"q4": "successful", "q1": "successful"}
    assert peak_fit_results is not None
    assert set(figures) == {"qubit_spectroscopy_chirp"}
    axes = figures["qubit_spectroscopy_chirp"].axes
    titled_axes = [ax for ax in axes if ax.get_title().startswith("qubit=")]
    assert titled_axes[0].get_title().endswith("q4")
    assert titled_axes[1].get_title().endswith("q1")

"""Unit tests for ``fit_raw_data`` in ``08b_qubit_spectroscopy``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from calibration_utils.qubit_spectroscopy.analysis import fit_raw_data

# ── Constants ─────────────────────────────────────────────────────────────────

RF_FREQ_Q1 = 10.0e9
RF_FREQ_Q2 = 10.5e9
DETUNINGS = np.arange(-10e6, 10e6, 0.25e6)
PEAK_FWHM = 2e6
PEAK_AMP = 0.4
NOISE_STD = 0.01
RNG = np.random.default_rng(42)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _lorentzian(x: np.ndarray, center: float, fwhm: float, amp: float) -> np.ndarray:
    hwhm = fwhm / 2.0
    return amp / (1.0 + ((x - center) / hwhm) ** 2)


def _make_qubit(name: str, rf_freq: float) -> SimpleNamespace:
    xy = SimpleNamespace(RF_frequency=rf_freq)
    return SimpleNamespace(name=name, xy=xy)


def _make_node() -> SimpleNamespace:
    params = SimpleNamespace(
        frequency_span_in_mhz=20.0,
    )
    qubits = [_make_qubit("q1", RF_FREQ_Q1), _make_qubit("q2", RF_FREQ_Q2)]
    return SimpleNamespace(parameters=params, namespace={"qubits": qubits})


def _detuning_coord() -> xr.DataArray:
    return xr.DataArray(
        DETUNINGS,
        dims="detuning",
        attrs={"long_name": "drive frequency", "units": "Hz"},
    )


def _make_ds_single_peak(
    centers=None,
    amp=PEAK_AMP,
) -> xr.Dataset:
    """Single Lorentzian peak per qubit with realistic noise."""
    if centers is None:
        centers = {"q1": 3e6, "q2": -2e6}
    n = len(DETUNINGS)
    return xr.Dataset(
        {
            "state": xr.DataArray(
                np.stack(
                    [
                        _lorentzian(DETUNINGS, centers["q1"], PEAK_FWHM, amp) + RNG.normal(0, NOISE_STD, n),
                        _lorentzian(DETUNINGS, centers["q2"], PEAK_FWHM, amp) + RNG.normal(0, NOISE_STD, n),
                    ]
                ),
                dims=("qubit", "detuning"),
            ),
        },
        coords={"qubit": ["q1", "q2"], "detuning": _detuning_coord()},
    )


def _make_ds_single_dip(
    centers=None,
    amp=PEAK_AMP,
) -> xr.Dataset:
    """Single Lorentzian dip per qubit (baseline at 0.5) with realistic noise."""
    if centers is None:
        centers = {"q1": 3e6, "q2": -2e6}
    baseline = 0.5
    n = len(DETUNINGS)
    return xr.Dataset(
        {
            "state": xr.DataArray(
                np.stack(
                    [
                        baseline
                        - _lorentzian(DETUNINGS, centers["q1"], PEAK_FWHM, amp)
                        + RNG.normal(0, NOISE_STD, n),
                        baseline
                        - _lorentzian(DETUNINGS, centers["q2"], PEAK_FWHM, amp)
                        + RNG.normal(0, NOISE_STD, n),
                    ]
                ),
                dims=("qubit", "detuning"),
            ),
        },
        coords={"qubit": ["q1", "q2"], "detuning": _detuning_coord()},
    )


def _make_ds_two_peaks() -> xr.Dataset:
    """Two well-separated Lorentzian peaks of equal amplitude with noise.

    Uses a dedicated RNG so the result is deterministic regardless of
    which other tests have run.
    """
    rng = np.random.default_rng(99)
    narrow_fwhm = 1e6
    n = len(DETUNINGS)
    return xr.Dataset(
        {
            "state": xr.DataArray(
                np.stack(
                    [
                        _lorentzian(DETUNINGS, 1e6, narrow_fwhm, PEAK_AMP)
                        + _lorentzian(DETUNINGS, 8e6, narrow_fwhm, PEAK_AMP)
                        + rng.normal(0, NOISE_STD, n),
                        _lorentzian(DETUNINGS, -1e6, narrow_fwhm, PEAK_AMP)
                        + _lorentzian(DETUNINGS, -8e6, narrow_fwhm, PEAK_AMP)
                        + rng.normal(0, NOISE_STD, n),
                    ]
                ),
                dims=("qubit", "detuning"),
            ),
        },
        coords={"qubit": ["q1", "q2"], "detuning": _detuning_coord()},
    )


# ── Tests: dataset structure ─────────────────────────────────────────────────


def test_fit_raw_data_preserves_state_with_correct_dims():
    node = _make_node()
    ds = _make_ds_single_peak()
    ds_fit, _ = fit_raw_data(ds, node)

    assert "state" in ds_fit.data_vars
    assert set(ds_fit.state.dims) == {"qubit", "detuning"}


def test_fit_raw_data_state_has_correct_qubit_coords():
    node = _make_node()
    ds = _make_ds_single_peak()
    ds_fit, _ = fit_raw_data(ds, node)

    assert list(ds_fit.state.qubit.values) == ["q1", "q2"]


def test_fit_raw_data_adds_full_freq_coord():
    node = _make_node()
    ds = _make_ds_single_peak()
    ds_fit, _ = fit_raw_data(ds, node)

    assert "full_freq" in ds_fit.coords
    assert set(ds_fit.full_freq.dims) == {"qubit", "detuning"}


def test_fit_raw_data_full_freq_values():
    node = _make_node()
    ds = _make_ds_single_peak()
    ds_fit, _ = fit_raw_data(ds, node)

    for qname, rf in zip(("q1", "q2"), (RF_FREQ_Q1, RF_FREQ_Q2)):
        expected = DETUNINGS + rf
        np.testing.assert_allclose(
            ds_fit.full_freq.sel(qubit=qname).values, expected
        )


def test_fit_raw_data_adds_fit_curve():
    node = _make_node()
    ds = _make_ds_single_peak()
    ds_fit, _ = fit_raw_data(ds, node)

    assert "fit_curve" in ds_fit.data_vars
    assert set(ds_fit.fit_curve.dims) == {"qubit", "detuning"}


# ── Tests: single peak fitting ───────────────────────────────────────────────


def test_single_peak_frequency_accuracy():
    """Fitted frequency should be within 0.5 MHz of the true peak centre."""
    centers = {"q1": 3e6, "q2": -2e6}
    node = _make_node()
    ds = _make_ds_single_peak(centers=centers)
    _, fit_results = fit_raw_data(ds, node)

    for qname, rf, true_det in [("q1", RF_FREQ_Q1, 3e6), ("q2", RF_FREQ_Q2, -2e6)]:
        r = fit_results[qname]
        assert r.num_peaks == 1
        assert abs(r.relative_freq - true_det) < 0.5e6
        assert abs(r.frequency - (rf + true_det)) < 0.5e6
        assert r.readout_qubit_frequency is None
        assert r.success


def test_single_peak_fwhm_accuracy():
    """Fitted FWHM should be within 50% of the true value."""
    node = _make_node()
    ds = _make_ds_single_peak()
    _, fit_results = fit_raw_data(ds, node)

    for qname in ("q1", "q2"):
        assert abs(fit_results[qname].fwhm - PEAK_FWHM) / PEAK_FWHM < 0.5


# ── Tests: single dip (trough) fitting ──────────────────────────────────────


def test_single_dip_frequency_accuracy():
    """Fit should correctly locate dips (negative Lorentzians)."""
    centers = {"q1": 3e6, "q2": -2e6}
    node = _make_node()
    ds = _make_ds_single_dip(centers=centers)
    _, fit_results = fit_raw_data(ds, node)

    for qname, true_det in [("q1", 3e6), ("q2", -2e6)]:
        r = fit_results[qname]
        assert abs(r.relative_freq - true_det) < 0.5e6
        assert r.success


# ── Tests: two-peak model selection ──────────────────────────────────────────


def test_two_peaks_detected_by_bic():
    """BIC should prefer a 2-peak model when two clear peaks are present."""
    node = _make_node()
    ds = _make_ds_two_peaks()
    _, fit_results = fit_raw_data(ds, node)

    for qname in ("q1", "q2"):
        assert fit_results[qname].num_peaks == 2


def test_two_peaks_closest_to_centre_is_primary():
    """The peak closest to detuning=0 should be the qubit frequency."""
    node = _make_node()
    ds = _make_ds_two_peaks()
    _, fit_results = fit_raw_data(ds, node)

    r_q1 = fit_results["q1"]
    assert abs(r_q1.relative_freq) < abs(r_q1.readout_qubit_relative_freq)


def test_two_peaks_readout_frequency_populated():
    """When 2 peaks are found, readout_qubit_frequency should be set."""
    node = _make_node()
    ds = _make_ds_two_peaks()
    _, fit_results = fit_raw_data(ds, node)

    for qname in ("q1", "q2"):
        assert fit_results[qname].readout_qubit_frequency is not None
        assert fit_results[qname].readout_qubit_relative_freq is not None


# ── Tests: error handling ─────────────────────────────────────────────────────


def test_fit_raw_data_raises_on_missing_signal_var():
    """fit_raw_data should raise KeyError if the state variable is missing."""
    node = _make_node()
    ds = xr.Dataset(
        {"random_var": xr.DataArray(np.zeros(len(DETUNINGS)), dims="detuning")},
        coords={"detuning": _detuning_coord()},
    )
    with pytest.raises(KeyError, match="Expected variable 'state' not found in dataset."):
        fit_raw_data(ds, node)

"""Tests for shared parity stream helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr

from calibration_utils.common_utils.experiment import AnalysisSignal
from calibration_utils.common_utils.parity_streams import (
    process_streams,
    resolve_analysis_signal,
)


def _joint_ds() -> xr.Dataset:
    amps = np.array([0.1, 0.2, 0.3])
    coords = {"amp_prefactor": xr.DataArray(amps, dims="amp_prefactor")}
    dims = ("amp_prefactor",)
    return xr.Dataset(
        {
            "p0_p0_q1": xr.DataArray(np.array([0.25, 0.20, 0.10]), dims=dims),
            "p0_p1_q1": xr.DataArray(np.array([0.25, 0.30, 0.40]), dims=dims),
            "p1_p0_q1": xr.DataArray(np.array([0.20, 0.30, 0.40]), dims=dims),
            "p1_p1_q1": xr.DataArray(np.array([0.30, 0.20, 0.10]), dims=dims),
        },
        coords=coords,
    )


def test_resolve_analysis_signal_uses_configured_signal_when_parity_enabled():
    params = SimpleNamespace(
        use_parity_pre_measurement=True,
        analysis_signal=AnalysisSignal.E_P2_GIVEN_P1_1,
    )

    assert resolve_analysis_signal(params) == "E_p2_given_p1_1"
    assert resolve_analysis_signal("E_p2_given_p1_0", True) == "E_p2_given_p1_0"


def test_resolve_analysis_signal_uses_p_when_parity_disabled():
    params = SimpleNamespace(
        use_parity_pre_measurement=False,
        analysis_signal=AnalysisSignal.E_P2_GIVEN_P1_1,
    )

    assert resolve_analysis_signal(params) == "p"
    assert resolve_analysis_signal("E_p2_given_p1_0", False) == "p"


def test_process_streams_adds_conditional_expectations_when_parity_enabled():
    ds = process_streams(_joint_ds(), ["q1"], use_parity_pre_measurement=True)

    assert "E_p2_given_p1_0_q1" in ds
    assert "E_p2_given_p1_1_q1" in ds
    np.testing.assert_allclose(ds["E_p2_given_p1_0_q1"], [0.5, 0.6, 0.8])
    np.testing.assert_allclose(ds["E_p2_given_p1_1_q1"], [0.6, 0.4, 0.2])


def test_process_streams_preserves_direct_streams_when_parity_disabled():
    amps = np.array([0.1, 0.2, 0.3])
    ds = xr.Dataset(
        {"p_q1": xr.DataArray(np.array([0.2, 0.4, 0.6]), dims=("amp_prefactor",))},
        coords={"amp_prefactor": xr.DataArray(amps, dims="amp_prefactor")},
    )

    processed = process_streams(ds, ["q1"], use_parity_pre_measurement=False)

    xr.testing.assert_identical(processed, ds)

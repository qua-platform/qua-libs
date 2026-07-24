from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr

from calibration_utils.common_utils.parity_streams import process_parity_streams
from calibration_utils.power_rabi.analysis import fit_raw_data


def _node_for_qubits(qubit_names: list[str]):
    return SimpleNamespace(
        namespace={
            "qubits": [SimpleNamespace(name=name) for name in qubit_names],
        },
        parameters=SimpleNamespace(analysis_signal="E_p2_given_p1_0"),
    )


def test_09a_process_raw_data_normalizes_single_measurement_stacked_stream():
    amps = np.linspace(0.001, 1.99, 64)
    q1_signal = 0.5 + 0.25 * np.cos(2 * np.pi * amps)
    q2_signal = 0.5 + 0.20 * np.sin(2 * np.pi * amps)
    ds_raw = xr.Dataset(
        {
            "p_q": xr.DataArray(
                np.stack([q1_signal, q2_signal]),
                dims=("qubit", "amp_prefactor"),
                coords={"qubit": ["q1", "q2"], "amp_prefactor": amps},
            )
        }
    )

    ds_processed = process_parity_streams(
        ds_raw,
        ["q1", "q2"],
        parity_pre_measurement=False,
        sweep_dims=("amp_prefactor",),
    )

    assert ds_processed["E_p2_given_p1_0_q1"].dims == ("amp_prefactor",)
    assert ds_processed["E_p2_given_p1_1_q1"].dims == ("amp_prefactor",)
    np.testing.assert_allclose(ds_processed["E_p2_given_p1_0_q1"], q1_signal)
    np.testing.assert_allclose(ds_processed["E_p2_given_p1_1_q2"], q2_signal)

    _, fit_results = fit_raw_data(ds_processed, _node_for_qubits(["q1", "q2"]))
    assert set(fit_results) == {"q1", "q2"}


def test_09a_process_raw_data_normalizes_joint_streams_to_sweep_dims():
    amps = np.linspace(0.001, 1.99, 64)
    q1_given_0 = 0.5 + 0.25 * np.cos(2 * np.pi * amps)
    q1_given_1 = 0.5 + 0.20 * np.sin(2 * np.pi * amps)
    ds_raw = xr.Dataset(
        {
            "p0_p0_q1": xr.DataArray(
                1.0 - q1_given_0,
                dims=("amp_prefactor",),
                coords={"amp_prefactor": amps},
            ),
            "p0_p1_q1": xr.DataArray(
                q1_given_0, dims=("amp_prefactor",), coords={"amp_prefactor": amps}
            ),
            "p1_p0_q1": xr.DataArray(
                1.0 - q1_given_1,
                dims=("amp_prefactor",),
                coords={"amp_prefactor": amps},
            ),
            "p1_p1_q1": xr.DataArray(
                q1_given_1, dims=("amp_prefactor",), coords={"amp_prefactor": amps}
            ),
        }
    )

    ds_processed = process_parity_streams(
        ds_raw,
        ["q1"],
        parity_pre_measurement=True,
        sweep_dims=("amp_prefactor",),
    )

    assert ds_processed["E_p2_given_p1_0_q1"].dims == ("amp_prefactor",)
    assert ds_processed["E_p2_given_p1_1_q1"].dims == ("amp_prefactor",)
    np.testing.assert_allclose(ds_processed["E_p2_given_p1_0_q1"], q1_given_0)
    np.testing.assert_allclose(ds_processed["E_p2_given_p1_1_q1"], q1_given_1)

    _, fit_results = fit_raw_data(ds_processed, _node_for_qubits(["q1"]))
    assert set(fit_results) == {"q1"}


def test_process_parity_streams_normalizes_false_mode_multidim_pair_stream():
    detuning = np.array([-1.0, 0.0, 1.0])
    tau = np.array([8.0, 16.0, 24.0, 32.0])
    pair_names = ["q1_q2", "q3_q4"]
    q12_signal = np.arange(len(tau) * len(detuning), dtype=float).reshape(
        len(tau), len(detuning)
    )
    q34_signal = q12_signal + 100.0
    ds_raw = xr.Dataset(
        {
            "p": xr.DataArray(
                np.stack([q12_signal, q34_signal]),
                dims=("qubit_pair", "tau", "detuning"),
                coords={
                    "qubit_pair": pair_names,
                    "tau": tau,
                    "detuning": detuning,
                },
            )
        }
    )

    ds_processed = process_parity_streams(
        ds_raw,
        pair_names,
        parity_pre_measurement=False,
        item_dim="qubit_pair",
        sweep_dims=("detuning", "tau"),
    )

    assert ds_processed["E_p2_given_p1_0_q1_q2"].dims == ("detuning", "tau")
    assert ds_processed["E_p2_given_p1_1_q3_q4"].dims == ("detuning", "tau")
    np.testing.assert_allclose(
        ds_processed["E_p2_given_p1_0_q1_q2"].values,
        q12_signal.T,
    )
    np.testing.assert_allclose(
        ds_processed["E_p2_given_p1_1_q3_q4"].values,
        q34_signal.T,
    )

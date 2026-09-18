"""Regression test for node 14 stacked state-stream analysis."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure

from calibration_utils.single_qubit_randomized_benchmarking.parameters import Parameters

NODE_NAME = "14_single_qubit_randomized_benchmarking"


def _make_rb_state(alpha: float, depths: np.ndarray, num_circuits: int, num_shots: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    survival = 0.45 * alpha**depths + 0.5
    return np.vstack(
        [
            rng.binomial(num_shots, np.clip(survival, 0.0, 1.0)) / num_shots
            for _ in range(num_circuits)
        ]
    )


@pytest.mark.analysis
def test_14_single_qubit_rb_state_stream_contract(analysis_runner, minimal_quam_factory):
    machine = minimal_quam_factory()
    qubit_name_1 = machine.qubits["q1"].name
    qubit_name_2 = machine.qubits["q2"].name

    params = Parameters(
        max_circuit_depth=64,
        log_scale=True,
        num_circuits_per_length=24,
        num_shots=300,
    )
    depths = params.get_depths()

    ds_raw = xr.Dataset(
        {
            "state": xr.DataArray(
                np.stack(
                    [
                        _make_rb_state(0.995, depths, 24, 300, seed=7),
                        _make_rb_state(0.965, depths, 24, 300, seed=8),
                    ],
                    axis=0,
                ),
                dims=("qubit", "circuit", "depth"),
                coords={
                    "qubit": [qubit_name_1, qubit_name_2],
                    "circuit": np.arange(24),
                    "depth": depths,
                },
            )
        }
    )

    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        analyse_qubits=["q1", "q2"],
        param_overrides={
            "max_circuit_depth": 64,
            "log_scale": True,
            "num_circuits_per_length": 24,
            "num_shots": 300,
        },
    )

    assert "ds_fit" in node.results
    assert "fit_results" in node.results
    assert "survival_probability" in node.results["ds_fit"].data_vars
    assert "state_fit" in node.results["ds_fit"].data_vars

    fit_1 = node.results["fit_results"][qubit_name_1]
    fit_2 = node.results["fit_results"][qubit_name_2]
    assert fit_1["success"]
    assert fit_2["success"]
    assert fit_1["alpha"] > fit_2["alpha"]

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert "raw_data_with_fit" in figures
    assert isinstance(figures["raw_data_with_fit"], Figure)

"""Execute test for 14_single_qubit_randomized_benchmarking."""

from __future__ import annotations

import pytest

NODE_NAME = "14_single_qubit_randomized_benchmarking"


@pytest.mark.execute
def test_single_qubit_rb_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for single-qubit RB."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "qubits": ["q1", "q2"],
            "num_circuits_per_length": 1,
            "max_circuit_depth": 100,
            "delta_clifford": 1,
            "log_scale": False,
            "simulation_duration_ns": 60_000,
            "timeout": 180,
        },
    )

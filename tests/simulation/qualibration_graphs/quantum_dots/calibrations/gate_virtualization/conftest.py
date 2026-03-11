"""Fixtures for gate-virtualization simulation tests (local-only).

Uses the unified wiring-based QuAM factory (``create_qd_quam``) and
shared test helpers from ``shared_fixtures``.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest


def pytest_configure(config):
    """Suppress known QuAM port-reference warnings (int-key vs string-key mismatch)."""
    warnings.filterwarnings(
        "ignore",
        message=r"Could not get reference.*#/ports/analog_outputs.*",
        category=UserWarning,
    )


CURRENT_DIR = Path(__file__).resolve().parent
SIMULATION_ROOT = CURRENT_DIR.parents[3]

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))

# ── Shared helpers (absolute sys.path import) ──────────────────────────
_SHARED_DIR = Path(__file__).resolve().parents[5] / "qualibration_graphs" / "quantum_dots" / "calibrations"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from shared_fixtures import (  # noqa: E402
    REPO_ROOT,
    apply_param_overrides,
    configure_machine_network,
    ensure_quam_config_stub,
    find_repo_root,
    get_parameters_dict,
    is_qm_connectivity_error,
    load_library_node,
    make_markdown_generator_sim,
    make_save_simulation_plot,
    patch_qualibrate_logger,
    setup_test_cache,
)
from quam_factory import create_qd_quam  # noqa: E402

# ── Cache setup ────────────────────────────────────────────────────────
_cache_base = Path(os.environ.get("SIM_TEST_CACHE_DIR", "/tmp/qua_simulation_pytest_cache"))
setup_test_cache(_cache_base)
patch_qualibrate_logger(_cache_base)

# ── Paths and defaults ─────────────────────────────────────────────────

CALIBRATION_LIBRARY_ROOT = REPO_ROOT / "qualibration_graphs" / "quantum_dots" / "calibrations" / "gate_virtualization"
ARTIFACTS_BASE = SIMULATION_ROOT / "artifacts"

DEFAULT_SMALL_SWEEP_PARAMS: Dict[str, Any] = {
    "num_shots": 1,
    "sensor_gate_span": 0.02,
    "sensor_gate_points": 21,
    "device_gate_span": 0.02,
    "device_gate_points": 21,
    "simulation_duration_ns": 20_000,
    "timeout": 30,
    "sensor_device_mapping": {"virtual_sensor_1": ["virtual_dot_1"]},
    "virtual_gate_set_id": "main_qpu",
}


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture returning a Stage-1 ``BaseQuamQD``."""

    def _factory():
        return create_qd_quam()

    return _factory


@pytest.fixture
def markdown_generator():
    return make_markdown_generator_sim()


@pytest.fixture
def save_simulation_plot():
    return make_save_simulation_plot()


@pytest.fixture
def simulation_runner(minimal_quam_factory, save_simulation_plot, markdown_generator):
    """Run a local simulation test by node name with optional overrides."""

    def _run(
        node_name: str,
        param_overrides: Optional[Dict[str, Any]] = None,
        artifacts_subdir: Optional[str] = None,
        apply_small_sweep: bool = True,
        library_root: Optional[Path] = None,
    ) -> None:
        machine = minimal_quam_factory()
        if not configure_machine_network(machine):
            pytest.skip("Missing QM host configuration for local simulation.")

        ensure_quam_config_stub(machine)

        library_root = library_root or CALIBRATION_LIBRARY_ROOT
        node = load_library_node(node_name, library_root)
        node.machine = machine

        if apply_small_sweep:
            apply_param_overrides(node, DEFAULT_SMALL_SWEEP_PARAMS)
        apply_param_overrides(node, param_overrides)

        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)
        from quam_config import Quam

        run_params: Dict[str, Any] = {}
        params_obj = getattr(node, "parameters", None)
        if params_obj is not None and hasattr(params_obj, "model_dump"):
            run_params = params_obj.model_dump()
        run_params["simulate"] = True

        with patch.object(Quam, "load", return_value=machine):
            try:
                result = node.run(**run_params)
            except Exception as exc:
                if is_qm_connectivity_error(exc):
                    pytest.fail(f"QM cluster connectivity failure: {exc}")
                raise

        sim_result = getattr(node, "results", {}).get("simulation") if hasattr(node, "results") else None
        job = sim_result or result
        if job is None and hasattr(node, "namespace"):
            job = node.namespace.get("job")
        if job is None:
            pytest.skip("Simulation job not available from node execution.")

        save_simulation_plot(job, artifacts_dir, title="Simulated Samples")
        markdown_generator(node, get_parameters_dict(node), artifacts_dir)

        assert (artifacts_dir / "simulation.png").exists(), "simulation.png not created"
        assert (artifacts_dir / "README.md").exists(), "README.md not created"

    return _run


# ── Pytest hooks ───────────────────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "simulation: mark test as a simulation test (requires RUN_SIM_TESTS=1)",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/simulation/" in str(item.fspath) and not item.get_closest_marker("simulation"):
            item.add_marker(pytest.mark.simulation)

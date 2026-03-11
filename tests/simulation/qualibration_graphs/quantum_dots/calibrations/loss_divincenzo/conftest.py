"""Fixtures for Loss-DiVincenzo simulation tests (local-only).

Uses the unified wiring-based QuAM factory (``create_ld_quam``) and
shared test helpers from ``shared_fixtures``.
"""

from __future__ import annotations

import os
import sys
import types
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
    warnings.filterwarnings(
        "ignore",
        message=r"Skipping sticky-voltage tracking for macro.*",
        category=UserWarning,
    )


CURRENT_DIR = Path(__file__).resolve().parent
SIMULATION_ROOT = CURRENT_DIR.parents[3]

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))

# ── Shared helpers ─────────────────────────────────────────────────────
_SHARED_DIR = Path(__file__).resolve().parents[5] / "qualibration_graphs" / "quantum_dots" / "calibrations"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from shared_fixtures import (  # noqa: E402
    REPO_ROOT,
    apply_param_overrides,
    configure_machine_network,
    ensure_quam_config_stub,
    get_parameters_dict,
    load_library_node,
    make_markdown_generator_sim,
    make_save_simulation_plot,
    patch_qualibrate_logger,
    setup_test_cache,
)
from quam_factory import create_ld_quam  # noqa: E402

# ── Cache setup ────────────────────────────────────────────────────────
_cache_base = SIMULATION_ROOT / ".pytest_cache"
setup_test_cache(_cache_base)
patch_qualibrate_logger(_cache_base)

# ── Paths and defaults ─────────────────────────────────────────────────

CALIBRATION_LIBRARY_ROOT = REPO_ROOT / "qualibration_graphs" / "quantum_dots" / "calibrations" / "loss_divincenzo"
ARTIFACTS_BASE = SIMULATION_ROOT / "artifacts"

DEFAULT_SMALL_SWEEP_PARAMS: Dict[str, Any] = {
    "qubits": ["q1"],
    "num_shots": 1,
    "min_wait_time_in_ns": 16,
    "max_wait_time_in_ns": 1_024,
    "time_step_in_ns": 500,
    "frequency_span_in_mhz": 4,
    "frequency_step_in_mhz": 2,
    "gap_wait_time_in_ns": 1056,
    "simulation_duration_ns": 40_000,
    "timeout": 120,
}


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture returning a ``LossDiVincenzoQuam`` with default macros."""

    def _factory():
        return create_ld_quam()

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
        from quam_config import Quam

        library_root = library_root or CALIBRATION_LIBRARY_ROOT
        node = load_library_node(node_name, library_root)
        node.machine = machine

        if apply_small_sweep:
            apply_param_overrides(node, DEFAULT_SMALL_SWEEP_PARAMS)
        apply_param_overrides(node, param_overrides)

        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)

        with patch.object(Quam, "load", return_value=machine):
            result = node.run(simulate=True)

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

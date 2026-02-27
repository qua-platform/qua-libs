"""Fixtures and helpers for Loss-DiVincenzo QUA program simulation tests (local-only)."""

from __future__ import annotations

import json
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

CURRENT_DIR = Path(__file__).resolve().parent
SIMULATION_ROOT = CURRENT_DIR.parents[3]

# Ensure local modules (quam_factory, macros, etc.) are importable regardless
# of the working directory from which pytest is invoked.
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))

# Ensure matplotlib/qualibrate can write caches/logs under repo.
_cache_base = SIMULATION_ROOT / ".pytest_cache"
_cache_base.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_base / "matplotlib"))
os.environ.setdefault("QUALIBRATE_LOG_DIR", str(_cache_base / "qualibrate"))

import matplotlib  # type: ignore[import-not-found]  # noqa: E402

matplotlib.use("Agg")  # Headless backend for CI/local runs
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # type: ignore[import-not-found]  # noqa: E402

from qualibrate.qualibration_library import QualibrationLibrary  # type: ignore[import-not-found]  # noqa: E402
from quam_builder.architecture.quantum_dots.qpu import (  # type: ignore[import-not-found]  # noqa: E402
    LossDiVincenzoQuam,
)

try:
    from .....path_utils import find_repo_root  # noqa: E402
except ImportError:
    from path_utils import find_repo_root  # type: ignore[import-not-found]  # noqa: E402
from quam.components import pulses as quam_pulses  # type: ignore[import-not-found]  # noqa: E402

try:
    from .quam_factory import create_minimal_quam  # noqa: E402
except ImportError:
    from quam_factory import create_minimal_quam  # type: ignore[import-not-found]  # noqa: E402

# pylint: enable=wrong-import-position

# =============================================================================
# Paths and defaults
# =============================================================================

TEST_ROOT = SIMULATION_ROOT
REPO_ROOT = find_repo_root(CURRENT_DIR)
CALIBRATION_LIBRARY_ROOT = REPO_ROOT / "qualibration_graphs" / "quantum_dots" / "calibrations" / "loss_divincenzo"
ARTIFACTS_BASE = TEST_ROOT / "artifacts"
CLUSTER_CONFIG_PATH = REPO_ROOT / "tests" / ".qm_cluster_config.json"
DEFAULT_SMALL_SWEEP_PARAMS: Dict[str, Any] = {
    "num_shots": 1,
    "min_wait_time_in_ns": 16,
    "max_wait_time_in_ns": 1_024,
    "time_step_in_ns": 500,
    "frequency_span_in_mhz": 4,
    "frequency_step_in_mhz": 2,
    "gap_wait_time_in_ns": 1056,
    "simulation_duration_ns": 20_000,
}


# =============================================================================
# QuAM factory
# =============================================================================


def _add_native_gate_operations(machine: LossDiVincenzoQuam) -> None:
    """Register the lowercase ``x180`` operation alias on each qubit's XY channel.

    All RB gates derive from this single calibrated pulse via
    ``amplitude_scale`` and ``frame_rotation_2pi`` at play time:

      - X rotations: amplitude_scale = theta / 180
      - Y rotations: +90° frame shift, play X equivalent, -90° frame shift
      - Z rotations: pure frame rotation (virtual, zero duration)
    """
    for qubit in machine.qubits.values():  # pylint: disable=no-member
        xy = qubit.xy
        ref_pulse = xy.operations.get("X180")
        if ref_pulse is None:
            continue

        if "x180" not in xy.operations:
            xy.operations["x180"] = quam_pulses.GaussianPulse(
                length=ref_pulse.length,
                amplitude=ref_pulse.amplitude,
                sigma=ref_pulse.sigma,
            )


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture that creates a minimal LossDiVincenzoQuam with 4 qubits.

    The returned machine includes native-gate operations (x90, x180, -x90,
    y90, y180, -y90) on every XY channel, scaled from the calibrated X180
    pulse.
    """

    def _factory() -> LossDiVincenzoQuam:
        machine = create_minimal_quam()
        _add_native_gate_operations(machine)
        return machine

    return _factory


# =============================================================================
# Markdown generator
# =============================================================================


@pytest.fixture
def markdown_generator():
    """Fixture that generates README.md documentation for a node."""

    def _generate(
        node: Any,
        parameters_dict: Dict[str, Any],
        artifacts_dir: Path,
    ) -> Path:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        params_table = ["| Parameter | Value | Description |", "|-----------|-------|-------------|"]
        for name, value in parameters_dict.items():
            doc = ""
            node_parameters = getattr(node, "parameters", None)
            if node_parameters is not None and hasattr(node_parameters, "__class__"):
                for cls in node_parameters.__class__.__mro__:
                    if hasattr(cls, "__annotations__") and name in cls.__annotations__:
                        if hasattr(cls, "__pydantic_fields__"):
                            field_info = cls.__pydantic_fields__.get(name)
                            if field_info and field_info.description:
                                doc = field_info.description
                                break
            params_table.append(f"| `{name}` | `{value}` | {doc} |")

        params_section = "\n".join(params_table)

        content = f"""# {getattr(node, 'name', 'Unknown Node')}

## Description

{getattr(node, 'description', 'No description available')}

## Parameters

{params_section}

## Simulation Output

![Simulation](simulation.png)

---
*Generated by simulation test infrastructure*
"""

        output_path = artifacts_dir / "README.md"
        output_path.write_text(content, encoding="utf-8")
        return output_path

    return _generate


# =============================================================================
# Simulation plot saver
# =============================================================================


@pytest.fixture
def save_simulation_plot():
    """Fixture that saves simulated samples to a PNG file."""

    def _save(simulated_output, artifacts_dir: Path, title: str = "Simulated Samples") -> Path:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Prefer the node's provided simulation figure (simulate_and_plot output).
        if isinstance(simulated_output, dict):
            fig = simulated_output.get("figure")
            if fig is not None and hasattr(fig, "savefig"):
                output_path = artifacts_dir / "simulation.png"
                fig.savefig(output_path, dpi=200)
                plt.close(fig)
                return output_path
            simulated_output = simulated_output.get("samples", simulated_output)

        simulated_samples = simulated_output
        if hasattr(simulated_output, "get_simulated_samples"):
            simulated_samples = simulated_output.get_simulated_samples()

        con_names = sorted(name for name in dir(simulated_samples) if name.startswith("con"))
        if not con_names:
            pytest.skip("No simulated analog connections found to plot.")

        con = getattr(simulated_samples, con_names[0])
        if not hasattr(con, "plot"):
            pytest.skip("Simulated connection does not support plotting.")
        con.plot()
        plt.title(title)
        plt.tight_layout()

        output_path = artifacts_dir / "simulation.png"
        plt.savefig(output_path, dpi=200)
        plt.close()

        return output_path

    return _save


# =============================================================================
# Simulation runner (local-only)
# =============================================================================


def _apply_param_overrides(node: Any, overrides: Optional[Dict[str, Any]]) -> None:
    if not overrides:
        return
    params = getattr(node, "parameters", None)
    if params is None:
        return
    for key, value in overrides.items():
        if hasattr(params, key):
            setattr(params, key, value)


def _configure_qualibrate(library_root: Path) -> None:
    """Best-effort configuration of Qualibrate runner settings."""
    try:
        from qualibrate.config import config as qualibrate_config  # type: ignore[import-not-found]
    except Exception:
        return

    try:
        qualibrate_config.set(
            "runner-calibration-library-folder",
            str(library_root),
        )
        qualibrate_config.set(
            "runner-calibration-library-resolver",
            "qualibrate.QualibrationLibrary",
        )
    except Exception:
        return


def _configure_machine_network(machine: LossDiVincenzoQuam) -> bool:
    """Populate machine.network with a host/cluster_name for local simulation."""
    network = getattr(machine, "network", None)
    if network is None:
        return False

    host = os.environ.get("QM_HOST")
    cluster_name = os.environ.get("QM_CLUSTER_NAME")

    if not host and CLUSTER_CONFIG_PATH.exists():
        try:
            config = CLUSTER_CONFIG_PATH.read_text(encoding="utf-8")
            data = json.loads(config)
            if isinstance(data, dict):
                host = host or data.get("host")
                cluster_name = cluster_name or data.get("cluster_name")
        except Exception:
            pass

    if not host:
        return False

    try:
        network["host"] = host
        if cluster_name:
            network["cluster_name"] = cluster_name
    except Exception:
        return False
    return True


def _load_library_node(node_name: str, library_root: Path) -> Any:
    if not library_root.exists():
        warnings.warn(f"Simulation skip: calibration library not found at {library_root}")
        pytest.skip("Calibration library not found.")

    _configure_qualibrate(library_root)
    library = QualibrationLibrary(library_folder=library_root)
    if node_name not in library.nodes:
        warnings.warn(f"Simulation skip: node '{node_name}' not found under {library_root}")
        pytest.skip("Node not found in calibration library.")

    return library.nodes[node_name]


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
        # 1) Build a minimal QuAM and make sure it can connect to the cluster.
        machine = minimal_quam_factory()
        if not _configure_machine_network(machine):
            pytest.skip("Missing QM host configuration for local simulation.")
        library_root = library_root or CALIBRATION_LIBRARY_ROOT
        node = _load_library_node(node_name, library_root)
        node.machine = machine

        # 2) Apply overrides (small sweep first, then test-specific overrides).
        if apply_small_sweep:
            _apply_param_overrides(node, DEFAULT_SMALL_SWEEP_PARAMS)
        _apply_param_overrides(node, param_overrides)

        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)

        try:
            from quam_config import Quam  # type: ignore[import-not-found]
            from unittest.mock import patch
        except Exception:
            stub = types.ModuleType("quam_config")

            class QuamStub:  # type: ignore[unused-ignore]
                @staticmethod
                def load():
                    return machine

            stub.Quam = QuamStub
            sys.modules["quam_config"] = stub
            from quam_config import Quam  # type: ignore[import-not-found,attr-defined]
            from unittest.mock import patch

        # 3) Force simulate=True and run the node with the programmatic QuAM.
        with patch.object(Quam, "load", return_value=machine):
            result = node.run(simulate=True)

        # 4) Collect simulation output and write artifacts.
        sim_result = getattr(node, "results", {}).get("simulation") if hasattr(node, "results") else None
        job = sim_result or result
        if job is None and hasattr(node, "namespace"):
            job = node.namespace.get("job")
        if job is None:
            pytest.skip("Simulation job not available from node execution.")

        save_simulation_plot(job, artifacts_dir, title="Simulated Samples")
        markdown_generator(node, _get_parameters_dict(node), artifacts_dir)

        assert (artifacts_dir / "simulation.png").exists(), "simulation.png not created"
        assert (artifacts_dir / "README.md").exists(), "README.md not created"

    return _run


def _get_parameters_dict(node: Any) -> Dict[str, Any]:
    params_obj = getattr(node, "parameters", None)
    params: Dict[str, Any] = {}
    if params_obj is None:
        return params
    for name in dir(params_obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(params_obj, name)
            if not callable(val):
                params[name] = val
        except Exception:  # pylint: disable=broad-except
            pass
    return params


# =============================================================================
# SECTION 6: Pytest Hooks
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "simulation: mark test as a simulation test (requires RUN_SIM_TESTS=1)",
    )


def pytest_collection_modifyitems(config, items):
    """Ensure tests under tests/simulation are marked as simulation."""
    for item in items:
        if "tests/simulation/" in str(item.fspath) and not item.get_closest_marker("simulation"):
            item.add_marker(pytest.mark.simulation)

"""Fixtures for CMA-ES simulation tests.

Mirrors the Loss-DiVincenzo simulation conftest but points the calibration
library root at the ``cmaes`` node directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CURRENT_DIR = Path(__file__).resolve().parent
SIMULATION_ROOT = CURRENT_DIR.parents[3]

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))

# ── Shared helpers ─────────────────────────────────────────────────────
_SHARED_DIR = (
    Path(__file__).resolve().parents[5]
    / "qualibration_graphs"
    / "quantum_dots"
    / "calibrations"
)
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from shared_fixtures import (  # noqa: E402
    REPO_ROOT,
    apply_param_overrides,
    cloud_simulator_qmm,
    configure_machine_network,
    ensure_quam_config_stub,
    find_saas_credentials,
    get_parameters_dict,
    load_library_node,
    make_markdown_generator_sim,
    make_save_simulation_plot,
    patch_qualibrate_logger,
    setup_test_cache,
)
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam

from tests.quam_test_machine import regenerate_state_directory  # noqa: E402

# ── Cache setup ────────────────────────────────────────────────────────
_cache_base = SIMULATION_ROOT / ".pytest_cache"
setup_test_cache(_cache_base)
patch_qualibrate_logger(_cache_base)

# ── Paths and defaults ─────────────────────────────────────────────────

CALIBRATION_LIBRARY_ROOT = (
    REPO_ROOT
    / "qualibration_graphs"
    / "quantum_dots"
    / "calibrations"
    / "cmaes"
)
ARTIFACTS_BASE = SIMULATION_ROOT / "artifacts"


def compute_area_under_curve(samples) -> Dict[str, Dict[str, float]]:
    """Compute mean voltage for each analog output channel."""
    import numpy as np

    areas: Dict[str, Dict[str, float]] = {}
    for con_name in sorted(samples.keys()):
        con = samples[con_name]
        port_areas: Dict[str, float] = {}
        for port_name in sorted(con.analog.keys()):
            waveform = np.asarray(con.analog[port_name])
            if np.iscomplexobj(waveform):
                port_areas[port_name] = float(np.mean(waveform.real))
            else:
                port_areas[port_name] = float(np.mean(waveform))
        areas[con_name] = port_areas
    return areas


def append_area_to_readme(artifacts_dir, areas: Dict[str, Dict[str, float]]) -> None:
    """Append an area-under-curve section to the existing README."""
    readme_path = Path(artifacts_dir) / "README.md"
    lines = [
        "",
        "## Area Under Curve (Mean Voltage per Channel)",
        "",
        "| Controller | Port | Mean Voltage (V) |",
        "|------------|------|------------------|",
    ]
    for con_name, ports in sorted(areas.items()):
        for port_name, area in sorted(ports.items()):
            lines.append(f"| {con_name} | {port_name} | {area:.6e} |")
    lines.append("")

    with open(readme_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def assert_balanced_analog_means_if_strict(
    areas: Dict[str, Dict[str, float]],
    *,
    margin_v: float = 0.001,
) -> None:
    """Require each simulated channel mean to sit within ``margin_v`` of 0 V, if opted in."""
    val = os.environ.get("QUAM_SIM_STRICT_ANALOG_MEAN", "")
    if val.lower() not in ("1", "true", "yes"):
        return
    for con_name, ports in areas.items():
        for port_name, mean_voltage in ports.items():
            assert abs(mean_voltage) < margin_v, (
                f"Channel {con_name}/{port_name} has non-zero mean voltage: "
                f"{mean_voltage:.6e} V (expected < {margin_v} V)"
            )


def _regenerate_quam_machine() -> LossDiVincenzoQuam:
    """Rebuild QUAM JSON, apply ``update_machine``, save, reload from disk."""
    loaded, _cfg = regenerate_state_directory()
    return loaded


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture returning a fresh QUAM machine from disk."""

    def _factory():
        return _regenerate_quam_machine()

    return _factory


@pytest.fixture
def markdown_generator():
    return make_markdown_generator_sim()


@pytest.fixture
def save_simulation_plot():
    return make_save_simulation_plot()


_last_simulated_samples = None


def _make_simulate_and_plot_fn(override_qmm=None):
    """Return a simulate_and_plot patch function."""

    def _patched(qmm, config, program, node_parameters):
        global _last_simulated_samples
        import matplotlib.pyplot as plt
        from qm.simulate import SimulationConfig

        active_qmm = override_qmm if override_qmm is not None else qmm
        simulation_config = SimulationConfig(
            duration=node_parameters.simulation_duration_ns // 4
        )
        job = active_qmm.simulate(config, program, simulation_config)
        job.wait_until("Done", 240)

        samples = job.get_simulated_samples()
        _last_simulated_samples = samples

        fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
        for i, con in enumerate(samples.keys()):
            plt.subplot(len(samples.keys()), 1, i + 1)
            samples[con].plot()
            plt.title(con)
        plt.tight_layout()

        wf_report = None
        if node_parameters.use_waveform_report:
            wf_report = job.get_simulated_waveform_report()
            wf_report.create_plot(samples, plot=True, save_path=None)

        return samples, fig, wf_report

    return _patched


@pytest.fixture
def simulation_runner(minimal_quam_factory, save_simulation_plot, markdown_generator):
    """Run a simulation test by node name with optional overrides."""

    def _run(
        node_name: str,
        param_overrides: Optional[Dict[str, Any]] = None,
        artifacts_subdir: Optional[str] = None,
        apply_small_sweep: bool = True,
        library_root: Optional[Path] = None,
        use_cloud: bool = True,
    ):
        global _last_simulated_samples
        _last_simulated_samples = None

        machine = minimal_quam_factory()
        ensure_quam_config_stub(machine)
        from quam_config import Quam

        library_root = library_root or CALIBRATION_LIBRARY_ROOT
        node = load_library_node(node_name, library_root)
        node.machine = machine

        apply_param_overrides(node, param_overrides)

        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)

        use_cloud = use_cloud and find_saas_credentials() is not None

        if use_cloud:
            with cloud_simulator_qmm() as cloud_qmm:
                simulate_fn = _make_simulate_and_plot_fn(override_qmm=cloud_qmm)
                with (
                    patch.object(Quam, "load", return_value=machine),
                    patch.object(machine, "connect", return_value=cloud_qmm),
                    patch(
                        "qualibration_libs.runtime.simulate_and_plot",
                        simulate_fn,
                    ),
                    patch(
                        "qualibration_libs.runtime.simulate.simulate_and_plot",
                        simulate_fn,
                    ),
                ):
                    result = node.run(simulate=True)
        else:
            if not configure_machine_network(machine):
                pytest.skip("Missing QM host configuration for OPX simulation.")
            simulate_fn = _make_simulate_and_plot_fn()
            with (
                patch.object(Quam, "load", return_value=machine),
                patch(
                    "qualibration_libs.runtime.simulate_and_plot",
                    simulate_fn,
                ),
                patch(
                    "qualibration_libs.runtime.simulate.simulate_and_plot",
                    simulate_fn,
                ),
            ):
                result = node.run(simulate=True)

        sim_result = (
            getattr(node, "results", {}).get("simulation")
            if hasattr(node, "results")
            else None
        )
        job = sim_result or result
        if job is None and hasattr(node, "namespace"):
            job = node.namespace.get("job")
        if job is None:
            pytest.skip("Simulation job not available from node execution.")

        save_simulation_plot(job, artifacts_dir, title="Simulated Samples")
        markdown_generator(node, get_parameters_dict(node), artifacts_dir)

        assert (artifacts_dir / "simulation.png").exists(), "simulation.png not created"
        assert (artifacts_dir / "README.md").exists(), "README.md not created"

        return _last_simulated_samples, artifacts_dir

    return _run


# ── Pytest hooks ───────────────────────────────────────────────────────


def _remove_deprecated_version_from_qua_config_template() -> None:
    try:
        from quam.core.qua_config_template import qua_config_template

        qua_config_template.pop("version", None)
    except Exception:
        pass


def pytest_configure(config):
    _remove_deprecated_version_from_qua_config_template()
    config.addinivalue_line(
        "markers",
        "simulation: mark test as a simulation test (requires RUN_SIM_TESTS=1)",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/simulation/" in str(item.fspath) and not item.get_closest_marker(
            "simulation"
        ):
            item.add_marker(pytest.mark.simulation)

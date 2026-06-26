"""Shared helpers and fixtures for quantum-dot calibration tests.

This module consolidates the utilities that were previously duplicated
across the four per-suite ``conftest.py`` files:

* Qualibrate configuration and logger patching
* Calibration-library node loading
* ActionManager register-only patching (analysis tests)
* Parameter-override helpers
* ``qua_dashboards`` import stub (gate-virtualization tests)
* Machine network configuration (simulation tests)
* Markdown / artifact generators
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import types
import warnings
from functools import wraps
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, Optional
from unittest.mock import patch

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ── Repo root discovery ────────────────────────────────────────────────


def find_repo_root(start: Path) -> Path:
    """Find the repository root by locating ``tests/`` and ``qualibration_graphs/``."""
    current = start
    while current != current.parent:
        if (current / "tests").is_dir() and (current / "qualibration_graphs").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not locate repo root containing tests/ and qualibration_graphs/."
    )


SHARED_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(SHARED_DIR)
CLUSTER_CONFIG_PATH = REPO_ROOT / "tests" / ".qm_cluster_config.json"

_LOCAL_QD_ROOT = str(REPO_ROOT / "qualibration_graphs" / "quantum_dots")
if _LOCAL_QD_ROOT not in sys.path:
    sys.path.insert(0, _LOCAL_QD_ROOT)
_local_cu_path = str(
    REPO_ROOT / "qualibration_graphs" / "quantum_dots" / "calibration_utils"
)
_cu_mod = sys.modules.get("calibration_utils")
if (
    _cu_mod is not None
    and hasattr(_cu_mod, "__path__")
    and _local_cu_path not in list(_cu_mod.__path__)
):
    _cu_mod.__path__.insert(0, _local_cu_path)


# ── Cache / logger helpers ─────────────────────────────────────────────


def setup_test_cache(cache_root: Path) -> None:
    """Create cache directories and set env vars for matplotlib/qualibrate."""
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("QUALIBRATE_LOG_DIR", str(cache_root / "qualibrate"))


def patch_qualibrate_logger(cache_root: Path) -> None:
    """Force qualibrate file logging into the repo-local pytest cache."""
    try:
        import qualibrate.utils.logger_m as logger_m
    except Exception:
        try:
            import qualibrate.core.utils.logger_m as logger_m
        except Exception:
            return

    log_dir = cache_root / "qualibrate" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def _local_log_filepath() -> Path:
        return log_dir / "qualibrate.log"

    logger_m.LazyInitLogger.get_log_filepath = staticmethod(_local_log_filepath)


# ── Qualibrate configuration ──────────────────────────────────────────


def configure_qualibrate(library_root: Path) -> None:
    """Best-effort configuration of Qualibrate runner settings."""
    try:
        from qualibrate.config import config as qualibrate_config
    except Exception:
        return
    try:
        qualibrate_config.set("runner-calibration-library-folder", str(library_root))
        qualibrate_config.set(
            "runner-calibration-library-resolver",
            "qualibrate.QualibrationLibrary",
        )
    except Exception:
        return


# ── Node loading ───────────────────────────────────────────────────────


def load_library_node(node_name: str, library_root: Path) -> Any:
    """Load a calibration node by directly importing its module file.

    Uses register-only patching so that ``@node.run_action`` decorators
    only *register* the action without eagerly executing it (qualibrate
    >= 1.1 executes actions at decoration time by default).

    Raises on any error so that test failures are visible rather than
    silently skipped.
    """
    if not library_root.exists():
        pytest.fail(f"Calibration library not found at {library_root}")

    node_file = library_root / f"{node_name}.py"
    if not node_file.exists():
        pytest.fail(f"Node file '{node_name}.py' does not exist in {library_root}")

    parent_dir = str(library_root.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    cal_utils_dir = str(library_root.parents[1])
    if cal_utils_dir not in sys.path:
        sys.path.insert(0, cal_utils_dir)

    mod_name = f"_test_node_{node_name}"
    spec = spec_from_file_location(mod_name, node_file)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not create import spec for {node_file}")
    mod = module_from_spec(spec)

    with patch_action_manager_register_only():
        spec.loader.exec_module(mod)

    node = getattr(mod, "node", None)
    if node is None:
        pytest.fail(f"Module {node_file} loaded but has no 'node' attribute")
    if getattr(node, "filepath", None) is None:
        node.filepath = node_file
    return node


def reimport_node_to_register_actions(node_name: str, library_root: Path) -> Any | None:
    """Re-import the node module to get run_action handlers registered.

    Library scanning uses inspection mode and stops before decorators run,
    so the scanned node has no registered actions.
    """
    node_file = library_root / f"{node_name}.py"
    if not node_file.exists():
        return None
    mod_name = f"_analysis_node_{node_name}"
    spec = spec_from_file_location(mod_name, node_file)
    if spec is None or spec.loader is None:
        return None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "node", None)


# ── ActionManager register-only patching ───────────────────────────────


def patch_action_manager_register_only():
    """Patch ``ActionManager.register_action`` to only register, not execute.

    Returns a context manager that patches all discoverable ActionManager
    classes (qualibrate exposes the class under both a legacy path and a
    ``core`` path; they can be distinct classes, so we patch both).
    """
    am_classes = []
    Action = None
    try:
        from qualibrate.core.runnables.run_action.action import Action as _A
        from qualibrate.core.runnables.run_action.action_manager import (
            ActionManager as _AM,
        )

        Action = _A
        am_classes.append(_AM)
    except ImportError:
        pass
    try:
        from qualibrate.runnables.run_action.action import Action as _A2
        from qualibrate.runnables.run_action.action_manager import ActionManager as _AM2

        if Action is None:
            Action = _A2
        if _AM2 not in am_classes:
            am_classes.append(_AM2)
    except ImportError:
        pass

    if not am_classes or Action is None:
        raise ImportError("Cannot import ActionManager from qualibrate")

    def _register_only(self, node, func=None, *, skip_if=False):
        def decorator(f):
            action = Action(f, self)
            self.actions[f.__name__] = action

            @wraps(f)
            def wrapper(*args, **kwargs):
                return self.run_action(f.__name__, node, *args, **kwargs)

            return wrapper

        return decorator if func is None else decorator(func)

    from contextlib import ExitStack

    class _MultiPatch:
        def __init__(self):
            self._stack = ExitStack()

        def __enter__(self):
            for cls in am_classes:
                self._stack.enter_context(
                    patch.object(cls, "register_action", _register_only)
                )
            return self

        def __exit__(self, *exc):
            return self._stack.__exit__(*exc)

    return _MultiPatch()


def call_node_action(node: Any, action_name: str) -> None:
    """Call a registered run_action on *node*."""
    action_manager = getattr(node, "_action_manager", None)
    if action_manager is not None:
        actions = getattr(action_manager, "actions", {})
        action = actions.get(action_name)
        if action is not None:
            try:
                action.execute_run_action(node)
            except Exception as exc:
                warnings.warn(f"Action '{action_name}' raised: {exc}")
            return
    pytest.fail(
        f"Node {getattr(node, 'name', '?')} has no registered "
        f"run_action '{action_name}'."
    )


# ── Parameter / metadata helpers ───────────────────────────────────────


def apply_param_overrides(node: Any, overrides: Optional[Dict[str, Any]]) -> None:
    """Apply a dict of parameter overrides to a calibration node."""
    if not overrides:
        return
    params = getattr(node, "parameters", None)
    if params is None:
        return
    for key, value in overrides.items():
        if hasattr(params, key):
            setattr(params, key, value)


def get_parameters_dict(node: Any) -> Dict[str, Any]:
    """Extract a flat ``{name: value}`` dict from a node's parameters."""
    params_obj = getattr(node, "parameters", None)
    params: Dict[str, Any] = {}
    if params_obj is None:
        return params
    if hasattr(type(params_obj), "model_fields"):
        for name in type(params_obj).model_fields:
            try:
                params[name] = getattr(params_obj, name)
            except Exception:
                pass
    else:
        for name in dir(params_obj):
            if name.startswith("_"):
                continue
            try:
                val = getattr(params_obj, name)
                if not callable(val):
                    params[name] = val
            except Exception:
                pass
    return params


# ── Machine network configuration (simulation tests) ──────────────────


def configure_machine_network(machine) -> bool:
    """Populate ``machine.network`` from env vars or cluster config file."""
    network = getattr(machine, "network", None)
    if network is None:
        return False

    host = os.environ.get("QM_HOST")
    cluster_name = os.environ.get("QM_CLUSTER_NAME")

    if not host and CLUSTER_CONFIG_PATH.exists():
        try:
            data = json.loads(CLUSTER_CONFIG_PATH.read_text(encoding="utf-8"))
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


# ── Cloud (SaaS) simulator helpers ────────────────────────────────────

_SAAS_CREDENTIALS_CANDIDATES = [
    REPO_ROOT / ".qm_saas_credentials.json",
    REPO_ROOT / "quam-builder" / ".qm_saas_credentials.json",
    Path.home() / ".qm_saas_credentials.json",
]


def find_saas_credentials() -> Optional[Path]:
    """Return the first existing QM SaaS credentials file, or None."""
    for path in _SAAS_CREDENTIALS_CANDIDATES:
        if path.is_file():
            return path
    return None


def load_saas_credentials(path: Optional[Path] = None) -> Optional[Dict[str, str]]:
    """Load SaaS credentials from *path* or the first discovered candidate.

    Returns a dict with at least ``email`` and ``password`` keys, or None.
    """
    creds_path = path or find_saas_credentials()
    if creds_path is None:
        return None
    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "email" in data and "password" in data:
            return data
    except Exception:
        pass
    return None


@contextlib.contextmanager
def cloud_simulator_qmm(credentials: Optional[Dict[str, str]] = None) -> Iterator:
    """Context manager that yields a QMM connected to a QM SaaS cloud simulator.

    Usage::

        with cloud_simulator_qmm() as qmm:
            job = qmm.simulate(config, program, SimulationConfig(duration=1000))
            job.wait_until("Done", 300)
            samples = job.get_simulated_samples()

    Raises ``ImportError`` if ``qm_saas`` is not installed, and ``RuntimeError``
    if no credentials file can be found.
    """
    try:
        import qm_saas  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Cloud simulation requires the `qm_saas` package. "
            "Install it or pass use_cloud=False to fall back to OPX simulation."
        ) from exc

    from qm import QuantumMachinesManager

    creds = credentials or load_saas_credentials()
    if creds is None:
        raise RuntimeError(
            "No QM SaaS credentials found. Add a .qm_saas_credentials.json file "
            "to the repo root or pass use_cloud=False to use OPX simulation."
        )

    email = creds["email"]
    password = creds["password"]
    host = creds.get("host", "qm-saas.dev.quantum-machines.co")

    client = qm_saas.QmSaas(email=email, password=password, host=host)
    client.close_all()
    with client.simulator(client.latest_version()) as instance:
        qmm = QuantumMachinesManager(
            host=instance.host,
            port=instance.port,
            connection_headers=instance.default_connection_headers,
        )
        yield qmm


# ── quam_config stub ───────────────────────────────────────────────────


def ensure_quam_config_stub(machine) -> None:
    """Ensure ``quam_config.Quam.load()`` returns *machine*.

    Injects (or updates) a stub ``quam_config`` module in ``sys.modules``.
    Can be called multiple times; the most recent *machine* wins.
    """

    class QuamStub:
        @staticmethod
        def load(*args, **kwargs):
            return machine

    existing = sys.modules.get("quam_config")
    if existing is not None:
        existing.Quam = QuamStub
        if not hasattr(existing, "QubitQuam"):
            existing.QubitQuam = QuamStub
        return

    stub = types.ModuleType("quam_config")
    stub.Quam = QuamStub
    stub.QubitQuam = QuamStub
    sys.modules["quam_config"] = stub


# ── qua_dashboards import stub ─────────────────────────────────────────


def ensure_qua_dashboards_stub() -> None:
    """Inject a minimal ``qua_dashboards`` stub when optional UI deps are missing."""
    if "qua_dashboards" in sys.modules:
        return

    qua_dashboards = ModuleType("qua_dashboards")
    video_mode = ModuleType("qua_dashboards.video_mode")
    voltage_control = ModuleType("qua_dashboards.voltage_control")
    core = ModuleType("qua_dashboards.core")
    virtual_gates = ModuleType("qua_dashboards.virtual_gates")

    class _Dummy:
        def __init__(self, *a, **kw):
            pass

    class _ScanModes:
        class SwitchRasterScan:
            pass

        class RasterScan:
            pass

        class SpiralScan:
            pass

    video_mode.VideoModeComponent = _Dummy
    video_mode.OPXDataAcquirer = _Dummy
    video_mode.scan_modes = _ScanModes
    voltage_control.VoltageControlComponent = _Dummy
    core.build_dashboard = lambda *a, **kw: type("_App", (), {"server": None})()
    virtual_gates.VirtualLayerEditor = _Dummy
    virtual_gates.ui_update = lambda *a, **kw: None

    qua_dashboards.video_mode = video_mode
    qua_dashboards.voltage_control = voltage_control
    qua_dashboards.core = core
    qua_dashboards.virtual_gates = virtual_gates

    for name, mod in [
        ("qua_dashboards", qua_dashboards),
        ("qua_dashboards.video_mode", video_mode),
        ("qua_dashboards.voltage_control", voltage_control),
        ("qua_dashboards.core", core),
        ("qua_dashboards.virtual_gates", virtual_gates),
    ]:
        sys.modules[name] = mod


# ── QM connectivity error detection ───────────────────────────────────


def is_qm_connectivity_error(exc: Exception) -> bool:
    """Return True when the error looks like an unreachable QM cluster."""
    msg = str(exc)
    markers = (
        "QmServerDetectionError",
        "Failed to detect to QuantumMachines server",
        "All connection attempts failed",
        "Connection refused",
        "timed out",
        "Name or service not known",
    )
    return any(m in msg for m in markers)


# ── Fixture helpers (not actual pytest fixtures — used by conftest) ────


def make_markdown_generator_sim():
    """Return a callable that generates README.md for a simulation node."""

    def _generate(node, parameters_dict, artifacts_dir):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        params_table = [
            "| Parameter | Value | Description |",
            "|-----------|-------|-------------|",
        ]
        for name, value in parameters_dict.items():
            doc = ""
            node_params = getattr(node, "parameters", None)
            if node_params is not None and hasattr(node_params, "__class__"):
                for cls in node_params.__class__.__mro__:
                    if hasattr(cls, "__annotations__") and name in cls.__annotations__:
                        if hasattr(cls, "__pydantic_fields__"):
                            fi = cls.__pydantic_fields__.get(name)
                            if fi and fi.description:
                                doc = fi.description
                                break
            params_table.append(f"| `{name}` | `{value}` | {doc} |")

        content = (
            f"# {getattr(node, 'name', 'Unknown Node')}\n\n"
            f"## Description\n\n"
            f"{getattr(node, 'description', 'No description available')}\n\n"
            f"## Parameters\n\n"
            + "\n".join(params_table)
            + "\n\n## Simulation Output\n\n"
            "![Simulation](simulation.png)\n\n"
            "---\n*Generated by simulation test infrastructure*\n"
        )
        output_path = artifacts_dir / "README.md"
        output_path.write_text(content, encoding="utf-8")
        return output_path

    return _generate


def make_save_simulation_plot():
    """Return a callable that saves simulated samples to PNG."""

    def _save(simulated_output, artifacts_dir, title="Simulated Samples"):
        artifacts_dir.mkdir(parents=True, exist_ok=True)

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
            simulated_output.wait_until("Done", 480)
            simulated_samples = simulated_output.get_simulated_samples()

        con_names = sorted(
            name for name in dir(simulated_samples) if name.startswith("con")
        )
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


def make_save_analysis_plot():
    """Return a callable that saves a matplotlib figure to the artifacts dir."""

    def _save(fig, artifacts_dir, filename="simulation.png"):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifacts_dir / filename
        fig.savefig(output_path, dpi=200)
        fig.savefig(output_path.with_suffix(".svg"))
        plt.close(fig)
        return output_path

    return _save


# ── Execute-test helpers ───────────────────────────────────────────────


def _diff_dicts(
    before: Dict[str, Any], after: Dict[str, Any], prefix: str = ""
) -> list:
    """Recursively compare two nested dicts, returning changed leaf entries.

    Returns a list of ``(dotted_key, before_value, after_value)`` tuples.
    Keys starting with ``_`` or equal to ``__class__`` are skipped.
    """
    changes: list = []
    all_keys = set(before) | set(after)
    for key in sorted(all_keys):
        if isinstance(key, str) and (key.startswith("_") or key == "__class__"):
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        b_val = before.get(key)
        a_val = after.get(key)
        if isinstance(b_val, dict) and isinstance(a_val, dict):
            changes.extend(_diff_dicts(b_val, a_val, path))
        elif b_val != a_val:
            changes.append((path, b_val, a_val))
    return changes


def save_execute_figures(node, artifacts_dir: Path) -> list:
    """Extract and save all figures from a node's results after execution.

    Returns a list of saved filenames (relative to *artifacts_dir*).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    saved: list = []
    results = getattr(node, "results", None) or {}

    snapshot_idx = getattr(node, "snapshot_idx", None)
    prefix = f"snapshot{snapshot_idx}_" if snapshot_idx is not None else ""

    single_fig = results.get("figure")
    if single_fig is not None and hasattr(single_fig, "savefig"):
        fname = f"{prefix}figure.png"
        single_fig.savefig(artifacts_dir / fname, dpi=200)
        plt.close(single_fig)
        saved.append(fname)

    figures = results.get("figures")
    if isinstance(figures, dict):
        for name, fig in figures.items():
            if fig is not None and hasattr(fig, "savefig"):
                safe_name = str(name).replace("/", "_").replace(" ", "_")
                fname = f"{prefix}{safe_name}.png"
                fig.savefig(artifacts_dir / fname, dpi=200)
                plt.close(fig)
                saved.append(fname)

    return saved


def make_markdown_generator_exec():
    """Return a callable that generates README.md for an execute-test node."""

    def _generate(
        node,
        parameters_dict: Dict[str, Any],
        artifacts_dir: Path,
        figures_saved: list,
        fit_results: Dict[str, Any],
        state_diff: list,
        metadata: Dict[str, Any],
    ) -> Path:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # -- Parameters table (skip Pydantic internals) --
        _SKIP_PARAMS = {
            "model_fields",
            "model_config",
            "model_computed_fields",
            "model_extra",
            "model_fields_set",
            "targets",
            "targets_name",
        }
        params_table = [
            "| Parameter | Value | Description |",
            "|-----------|-------|-------------|",
        ]
        node_params = getattr(node, "parameters", None)
        for name, value in parameters_dict.items():
            if name in _SKIP_PARAMS:
                continue
            doc = ""
            if node_params is not None and hasattr(node_params, "__class__"):
                for cls in node_params.__class__.__mro__:
                    if hasattr(cls, "__annotations__") and name in cls.__annotations__:
                        if hasattr(cls, "__pydantic_fields__"):
                            fi = cls.__pydantic_fields__.get(name)
                            if fi and fi.description:
                                doc = fi.description
                                break
            params_table.append(f"| `{name}` | `{value}` | {doc} |")

        sections = [
            f"# {getattr(node, 'name', 'Unknown Node')}",
            "",
            "## Description",
            "",
            getattr(node, "description", "No description available"),
            "",
            "## Parameters",
            "",
            *params_table,
        ]

        # -- Figures --
        if figures_saved:
            sections += ["", "## Execution Output", ""]
            for fname in figures_saved:
                label = fname.rsplit(".", 1)[0].replace("_", " ").title()
                sections.append(f"![{label}]({fname})")
            sections.append("")

        # -- Fit results --
        if fit_results:
            sections += ["", "## Fit Results", ""]
            for target_name, params in fit_results.items():
                sections.append(f"### {target_name}")
                sections += [
                    "| Parameter | Value |",
                    "|-----------|-------|",
                ]
                if isinstance(params, dict):
                    for k, v in params.items():
                        sections.append(f"| `{k}` | `{v}` |")
                sections.append("")

        # -- State updates --
        if state_diff:
            sections += ["", "## State Updates", ""]
            sections += [
                "| Parameter | Before | After |",
                "|-----------|--------|-------|",
            ]
            for key, before, after in state_diff:
                sections.append(f"| `{key}` | `{before}` | `{after}` |")
            sections.append("")

        # -- Metadata --
        if metadata:
            sections += ["", "## Metadata", ""]
            sections += [
                "| Key | Value |",
                "|-----|-------|",
            ]
            for k, v in metadata.items():
                sections.append(f"| {k} | {v} |")
            sections.append("")

        sections += [
            "---",
            "*Generated by execute test infrastructure*",
            "",
        ]

        content = "\n".join(sections)
        output_path = artifacts_dir / "README.md"
        output_path.write_text(content, encoding="utf-8")
        return output_path

    return _generate

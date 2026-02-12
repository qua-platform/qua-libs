"""Fixtures and helpers for Loss-DiVincenzo analysis tests using virtual_qpu.

These tests generate synthetic ``ds_raw`` datasets via physics simulation
(virtual_qpu) and then run the node's ``analyse_data``, ``plot_data``, and
``update_state`` actions against that data -- without requiring a real QOP
connection.
"""

from __future__ import annotations

import os
import sys
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

CURRENT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = CURRENT_DIR.parents[3]  # tests/analysis/

# Ensure matplotlib/qualibrate can write caches/logs under repo.
_cache_base = ANALYSIS_ROOT / ".pytest_cache"
_cache_base.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_base / "matplotlib"))
os.environ.setdefault("QUALIBRATE_LOG_DIR", str(_cache_base / "qualibrate"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # Headless backend for CI/local runs
import matplotlib.pyplot as plt  # noqa: E402

from qualibrate.qualibration_library import QualibrationLibrary  # noqa: E402
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam  # noqa: E402
from .....path_utils import find_repo_root  # noqa: E402
from .quam_factory import create_minimal_quam  # noqa: E402

# =============================================================================
# Paths and defaults
# =============================================================================

REPO_ROOT = find_repo_root(CURRENT_DIR)
CALIBRATION_LIBRARY_ROOT = REPO_ROOT / "qualibration_graphs" / "quantum_dots" / "calibrations" / "loss_divincenzo"
ARTIFACTS_BASE = ANALYSIS_ROOT / "artifacts"

# ── virtual_qpu path setup ──────────────────────────────────────────────────
# If virtual_qpu is not pip-installed, look for it as a sibling repo.
VIRTUAL_QPU_ROOT = REPO_ROOT.parent / "virtual_qpu"
_vqpu_src = str(VIRTUAL_QPU_ROOT / "src")
_vqpu_platforms = str(VIRTUAL_QPU_ROOT / "platforms")
if _vqpu_src not in sys.path:
    sys.path.insert(0, _vqpu_src)
if _vqpu_platforms not in sys.path:
    sys.path.insert(0, _vqpu_platforms)


# =============================================================================
# QuAM factory fixture
# =============================================================================


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture that creates a minimal LossDiVincenzoQuam with 4 qubits."""

    def _factory() -> LossDiVincenzoQuam:
        return create_minimal_quam()

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

        fit_results = node.results.get("fit_results", {})
        fit_section = ""
        if fit_results:
            fit_rows = [
                "| Qubit | f_res (GHz) | t_π (ns) | Ω_R (rad/ns) | success |",
                "|-------|-------------|----------|--------------|--------|",
            ]
            for qname, r in sorted(fit_results.items()):
                f_ghz = r.get("optimal_frequency", 0) * 1e-9
                t_pi = r.get("optimal_duration", float("nan"))
                omega = r.get("rabi_frequency", float("nan"))
                succ = r.get("success", False)
                fit_rows.append(f"| {qname} | {f_ghz:.4f} | {t_pi:.1f} | {omega:.6f} | {succ} |")
            fit_section = "\n## Fit Results\n\n" + "\n".join(fit_rows)

        state_section = ""
        if fit_results:
            op_name = getattr(getattr(node, "parameters", None), "operation", "x180")
            state_rows = [
                "| Qubit | intermediate_frequency (Hz) | xy.operations." + op_name + ".length (ns) |",
                "|-------|-----------------------------|-----------------------------------------|",
            ]
            for qname, r in sorted(fit_results.items()):
                if not r.get("success", False):
                    continue
                f_hz = r.get("optimal_frequency", 0)
                t_pi = r.get("optimal_duration", float("nan"))
                state_rows.append(f"| {qname} | {f_hz:.0f} | {t_pi:.1f} |")
            if len(state_rows) > 2:
                state_section = "\n## Updated State\n\n" + "\n".join(state_rows)

        content = f"""# {getattr(node, 'name', 'Unknown Node')}

## Description

{getattr(node, 'description', 'No description available')}

## Parameters

{params_section}
{fit_section}
{state_section}

## Analysis Output

![Analysis simulation](simulation.png)

---
*Generated by analysis test infrastructure (virtual_qpu)*
"""

        output_path = artifacts_dir / "README.md"
        output_path.write_text(content, encoding="utf-8")
        return output_path

    return _generate


# =============================================================================
# Analysis plot saver
# =============================================================================


@pytest.fixture
def save_analysis_plot():
    """Fixture that saves a matplotlib figure to the artifacts directory."""

    def _save(fig: Any, artifacts_dir: Path, filename: str = "simulation.png") -> Path:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifacts_dir / filename
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    return _save


# =============================================================================
# Node loading helpers
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
        from qualibrate.config import config as qualibrate_config
    except Exception:
        return
    try:
        qualibrate_config.set("runner-calibration-library-folder", str(library_root))
        qualibrate_config.set("runner-calibration-library-resolver", "qualibrate.QualibrationLibrary")
    except Exception:
        return


def _load_library_node(node_name: str, library_root: Path) -> Any:
    """Load a node from QualibrationLibrary, or skip if not found."""
    if not library_root.exists():
        warnings.warn(f"Analysis skip: calibration library not found at {library_root}")
        pytest.skip("Calibration library not found.")

    _configure_qualibrate(library_root)
    library = QualibrationLibrary(library_folder=library_root)
    if node_name not in library.nodes:
        warnings.warn(f"Analysis skip: node '{node_name}' not found under {library_root}")
        pytest.skip("Node not found in calibration library.")

    return library.nodes[node_name]


def _reimport_node_to_register_actions(node_name: str, library_root: Path) -> Any | None:
    """Re-import the node module and return the node with registered actions.

    Library scanning uses inspection mode and stops before decorators run, so
    the scanned node has no registered actions. Re-importing with ActionManager
    patched to register-only produces a node with analyse_data, plot_data, etc.
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


def _patch_action_manager_register_only():
    """Patch ActionManager.register_action to only register, not execute at import.

    The default decorator runs the action immediately when skip_if is False, which
    would execute create_qua_program etc. during module load. For analysis tests
    we only want to register actions so we can call them explicitly later.
    """
    from functools import wraps

    from qualibrate.runnables.run_action.action import Action
    from qualibrate.runnables.run_action.action_manager import ActionManager

    _original_register = ActionManager.register_action

    def _register_only(
        self,
        node: Any,
        func: Any = None,
        *,
        skip_if: bool = False,
    ) -> Any:
        def decorator(f: Any) -> Any:
            action = Action(f, self)
            action_name = f.__name__
            # Register on self (the ActionManager) - it's the same as node._action_manager
            self.actions[action_name] = action

            @wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.run_action(action_name, node, *args, **kwargs)

            return wrapper  # Do not call wrapper() at import time

        return decorator if func is None else decorator(func)

    return patch.object(ActionManager, "register_action", _register_only)


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
# Analysis runner fixture
# =============================================================================


@pytest.fixture
def analysis_runner(minimal_quam_factory, save_analysis_plot, markdown_generator):
    """Run an analysis test: inject synthetic ds_raw, execute analyse/plot/update actions.

    Parameters
    ----------
    node_name : str
        Calibration node name (e.g. ``"09b_time_rabi_chevron_parity_diff"``).
    ds_raw : xarray.Dataset
        Synthetic raw dataset matching the format produced by ``execute_qua_program``.
    fig : matplotlib Figure, optional
        If provided, saved as the artifact PNG.
    param_overrides : dict, optional
        Override node parameters before running analysis.
    artifacts_subdir : str, optional
        Override the artifact sub-directory name.
    library_root : Path, optional
        Override the calibration library root.
    analyse_qubits : list of str, optional
        Restrict analysis to these qubits only (e.g. ``["Q1"]``). Drops other
        qubits from ds_raw before injection so only they are fitted/plotted.

    Returns
    -------
    node
        The node after running the analysis pipeline.
    """

    def _run(
        node_name: str,
        ds_raw: Any,
        fig: Any = None,
        param_overrides: Optional[Dict[str, Any]] = None,
        artifacts_subdir: Optional[str] = None,
        library_root: Optional[Path] = None,
        analyse_qubits: Optional[list[str]] = None,
    ) -> Any:
        # 1) Build a minimal QuAM.
        machine = minimal_quam_factory()
        library_root = library_root or CALIBRATION_LIBRARY_ROOT

        # 2) Re-import the node module to get a node with registered run_action handlers.
        #    Library scanning uses inspection mode and stops before decorators run.
        #    Patch Quam.load() and ActionManager.register_action (register-only, no execute).
        from quam_config import Quam

        with patch.object(Quam, "load", return_value=machine), _patch_action_manager_register_only():
            node = _reimport_node_to_register_actions(node_name, library_root)
            if node is None:
                node = _load_library_node(node_name, library_root)
        node.machine = machine

        # 4) Apply parameter overrides.
        # simulate=False so analyse_data, plot_data, update_state run.
        overrides = dict(param_overrides) if param_overrides else {}
        overrides["simulate"] = False  # Analysis tests inject ds_raw; skip QUA sim/execute.
        _apply_param_overrides(node, overrides)

        # 5) Populate the namespace with qubits.
        try:
            from calibration_utils.common_utils.experiment import get_qubits

            node.namespace["qubits"] = get_qubits(node)
        except Exception:
            if hasattr(machine, "qubits"):
                qubits = machine.qubits
                if isinstance(qubits, dict):
                    node.namespace["qubits"] = list(qubits.values())
                else:
                    node.namespace["qubits"] = list(qubits)

        # 6) Optionally restrict to a subset of qubits for analysis.
        if analyse_qubits:
            keep_vars = []
            for q in analyse_qubits:
                for prefix in ("p1_", "p2_", "pdiff_"):
                    v = f"{prefix}{q}"
                    if v in ds_raw.data_vars:
                        keep_vars.append(v)
            if keep_vars:
                ds_raw = ds_raw[[v for v in ds_raw.data_vars if v in keep_vars]]

        # 7) Inject synthetic ds_raw.
        node.results["ds_raw"] = ds_raw

        # 8) Run analysis actions via the node's run_action handlers.
        _call_node_action(node, "analyse_data")
        _call_node_action(node, "plot_data")

        if "fit_results" in node.results:
            _call_node_action(node, "update_state")

        # 9) Save artifacts.
        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)
        # Prefer analysis figure (from plot_data) over raw simulation figure
        fig_to_save = node.results.get("figure") or fig
        if fig_to_save is not None:
            save_analysis_plot(fig_to_save, artifacts_dir, "simulation.png")

        markdown_generator(node, _get_parameters_dict(node), artifacts_dir)

        if fig_to_save is not None:
            assert (artifacts_dir / "simulation.png").exists(), "simulation.png not created"
        assert (artifacts_dir / "README.md").exists(), "README.md not created"

        return node

    return _run


def _call_node_action(node: Any, action_name: str) -> None:
    """Call a node's registered run_action by function name.

    Analysis, plotting, and state update run via the node's run_action handlers.
    """
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

    # No registered action — tests should fail if the node does not define it.
    pytest.fail(f"Node {getattr(node, 'name', '?')} has no registered run_action '{action_name}'.")


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "analysis: mark test as an analysis test using virtual_qpu",
    )


def pytest_collection_modifyitems(config, items):
    """Ensure tests under tests/analysis are marked as analysis."""
    for item in items:
        if "tests/analysis/" in str(item.fspath) and not item.get_closest_marker("analysis"):
            item.add_marker(pytest.mark.analysis)

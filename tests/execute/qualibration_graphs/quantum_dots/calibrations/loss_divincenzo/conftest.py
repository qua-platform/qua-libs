"""Fixtures for Loss-DiVincenzo execute tests (real QM hardware).

Runs each node with ``simulate=False`` on a real QM cluster, then collects
figures, fit results, and state updates into a comprehensive README report.

Uses on-disk QUAM state from :mod:`tests.quam_test_machine`. Each run calls
:func:`tests.quam_test_machine.regenerate_state_directory`, which rebuilds from
:func:`~tests.quam_test_machine.build_machine`, applies
:func:`~tests.quam_test_machine.update_machine`, saves (overwriting existing JSON),
then reloads from disk for the execute pipeline. Shared helpers come from
``shared_fixtures``.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CURRENT_DIR = Path(__file__).resolve().parent
EXECUTE_ROOT = CURRENT_DIR.parents[3]

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(EXECUTE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXECUTE_ROOT))

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
    _diff_dicts,
    apply_param_overrides,
    configure_machine_network,
    ensure_quam_config_stub,
    get_parameters_dict,
    load_library_node,
    make_markdown_generator_exec,
    patch_qualibrate_logger,
    save_execute_figures,
    setup_test_cache,
)
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam

from tests.quam_test_machine import regenerate_state_directory  # noqa: E402

# ── Patch qm_session to close other QMs on open ──────────────────────
# In test environments we are the sole user, so always grab the hardware
# immediately instead of polling/waiting for other sessions to release.
import logging
from contextlib import contextmanager

import qualang_tools.multi_user as _mu_pkg
import qualang_tools.multi_user.multi_user_tools as _mu_mod


@contextmanager
def _qm_session_close_others(qmm, config, timeout=100):
    """Like ``qm_session`` but with ``close_other_machines=True``."""
    qm_log = logging.getLogger("qm.api.frontend_api")
    qm_log.info("Opening QM (close_other_machines=True)")
    qm = qmm.open_qm(config, close_other_machines=True)
    try:
        yield qm
    except KeyboardInterrupt:
        pass
    finally:
        qm_log.info("Closing QM")
        qm.close()


# Patch both the implementation module and the public package so that
# ``from qualang_tools.multi_user import qm_session`` in node files
# picks up the patched version at import time.
_mu_mod.qm_session = _qm_session_close_others
_mu_pkg.qm_session = _qm_session_close_others

# ── Cache setup ────────────────────────────────────────────────────────
_cache_base = EXECUTE_ROOT / ".pytest_cache"
setup_test_cache(_cache_base)
patch_qualibrate_logger(_cache_base)

# ── Paths and defaults ─────────────────────────────────────────────────

CALIBRATION_LIBRARY_ROOT = (
    REPO_ROOT
    / "qualibration_graphs"
    / "quantum_dots"
    / "calibrations"
    / "loss_divincenzo"
)
ARTIFACTS_BASE = EXECUTE_ROOT / "artifacts"

DEFAULT_SMALL_SWEEP_PARAMS: Dict[str, Any] = {
    "qubits": ["q1"],
    "num_shots": 1,
    "min_wait_time_in_ns": 16,
    "max_wait_time_in_ns": 1_024,
    "time_step_in_ns": 500,
    "frequency_span_in_mhz": 40,
    "frequency_step_in_mhz": 5,
    "simulation_duration_ns": 40_000,
    "timeout": 500,
}


def _regenerate_quam_machine() -> LossDiVincenzoQuam:
    """Rebuild QUAM JSON, ``update_machine``, save (overwrite), load from disk."""
    loaded, _cfg = regenerate_state_directory()
    return loaded


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture returning a fresh ``LossDiVincenzoQuam`` from disk."""

    def _factory():
        return _regenerate_quam_machine()

    return _factory


@pytest.fixture
def markdown_generator():
    return make_markdown_generator_exec()


@pytest.fixture
def execute_runner(minimal_quam_factory, markdown_generator):
    """Run a full execute test by node name with optional overrides.

    Builds the machine, loads the node, runs ``node.run(simulate=False)``
    on real hardware, then collects figures, fit results, and state updates
    into a comprehensive README report.
    """

    def _run(
        node_name: str,
        param_overrides: Optional[Dict[str, Any]] = None,
        artifacts_subdir: Optional[str] = None,
        apply_small_sweep: bool = True,
        library_root: Optional[Path] = None,
    ) -> None:
        machine = minimal_quam_factory()
        if not configure_machine_network(machine):
            pytest.skip("Missing QM host configuration for execute test.")

        ensure_quam_config_stub(machine)
        from quam_config import Quam

        library_root = library_root or CALIBRATION_LIBRARY_ROOT
        node = load_library_node(node_name, library_root)
        node.machine = machine

        if apply_small_sweep:
            apply_param_overrides(node, DEFAULT_SMALL_SWEEP_PARAMS)
        apply_param_overrides(node, param_overrides)

        node.parameters.simulate = False

        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)

        # Snapshot machine state before execution
        machine_before: Optional[Dict[str, Any]] = None
        try:
            machine_before = machine.to_dict(include_defaults=False)
        except Exception:
            pass

        # Execute the full pipeline with save() patched to no-op.
        # Catch errors so we can still collect partial results and generate
        # the README — the report documents which actions succeeded/failed.
        run_error: Optional[Exception] = None
        start_time = time.monotonic()
        with patch.object(Quam, "load", return_value=machine):
            with patch.object(node, "save", new=lambda *a, **kw: None):
                try:
                    node.run(simulate=False)
                except Exception as exc:
                    run_error = exc
        elapsed = time.monotonic() - start_time

        # Snapshot machine state after execution
        machine_after: Optional[Dict[str, Any]] = None
        try:
            machine_after = machine.to_dict(include_defaults=False)
        except Exception:
            pass

        # Compute state diff
        state_diff: list = []
        if machine_before is not None and machine_after is not None:
            try:
                state_diff = _diff_dicts(machine_before, machine_after)
            except Exception:
                pass

        # Extract and save figures
        figures_saved = save_execute_figures(node, artifacts_dir)

        # Extract fit results
        results = getattr(node, "results", None) or {}
        fit_results = results.get("fit_results", {})
        if not isinstance(fit_results, dict):
            fit_results = {}

        # Collect action completion info from qualibrate's run error
        run_error_info = getattr(node, "error", None)
        completed_actions = []
        failed_action = None
        if run_error_info is not None:
            details = getattr(run_error_info, "details", "") or ""
            for line in details.splitlines():
                stripped = line.strip().lstrip("- ")
                if stripped and not stripped.startswith(
                    ("Completed", "Skipped", "Failed", "Source")
                ):
                    if "Completed actions" in details and line.strip().startswith("- "):
                        completed_actions.append(stripped)
            failed_action = getattr(run_error_info, "details_headline", "")

        # Build metadata
        metadata = {
            "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC"),
            "Node": node_name,
            "Duration": f"{elapsed:.1f}s",
            "Status": "completed" if run_error is None else "completed with errors",
        }
        if run_error is not None:
            metadata["Error"] = f"`{type(run_error).__name__}: {run_error}`"
            if failed_action:
                metadata["Failed Action"] = failed_action

        # Generate README
        markdown_generator(
            node=node,
            parameters_dict=get_parameters_dict(node),
            artifacts_dir=artifacts_dir,
            figures_saved=figures_saved,
            fit_results=fit_results,
            state_diff=state_diff,
            metadata=metadata,
        )

        assert (artifacts_dir / "README.md").exists(), "README.md not created"

        if run_error is not None:
            raise run_error

    return _run


# ── Pytest hooks ───────────────────────────────────────────────────────


def _remove_deprecated_version_from_qua_config_template() -> None:
    """Remove the deprecated ``version`` key from QuAM's QUA config template.

    ``qm-qua >= 1.2.2`` warns when ``version`` is present in the config.
    ``quam`` 0.4.x still includes it in the template; removing it here avoids
    the warning until quam drops it upstream.
    """
    try:
        from quam.core.qua_config_template import qua_config_template

        qua_config_template.pop("version", None)
    except Exception:
        pass


def pytest_configure(config):
    _remove_deprecated_version_from_qua_config_template()
    config.addinivalue_line(
        "markers",
        "execute: mark test as a full-execution test (requires QM host)",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/execute/" in str(item.fspath) and not item.get_closest_marker(
            "execute"
        ):
            item.add_marker(pytest.mark.execute)

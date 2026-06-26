"""Fixtures for CMA-ES execute tests (real QM hardware).

Mirrors the Loss-DiVincenzo execute conftest but points the calibration
library root at the ``cmaes`` node directory.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
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
    / "cmaes"
)
ARTIFACTS_BASE = EXECUTE_ROOT / "artifacts"


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
    return make_markdown_generator_exec()


@pytest.fixture
def execute_runner(minimal_quam_factory, markdown_generator):
    """Run a full execute test by node name with optional overrides."""

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

        apply_param_overrides(node, param_overrides)
        node.parameters.simulate = False

        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)

        machine_before: Optional[Dict[str, Any]] = None
        try:
            machine_before = machine.to_dict(include_defaults=False)
        except Exception:
            pass

        run_error: Optional[Exception] = None
        start_time = time.monotonic()
        with patch.object(Quam, "load", return_value=machine):
            with patch.object(node, "save", new=lambda *a, **kw: None):
                try:
                    node.run(simulate=False)
                except Exception as exc:
                    run_error = exc
        elapsed = time.monotonic() - start_time

        machine_after: Optional[Dict[str, Any]] = None
        try:
            machine_after = machine.to_dict(include_defaults=False)
        except Exception:
            pass

        state_diff: list = []
        if machine_before is not None and machine_after is not None:
            try:
                state_diff = _diff_dicts(machine_before, machine_after)
            except Exception:
                pass

        figures_saved = save_execute_figures(node, artifacts_dir)

        results = getattr(node, "results", None) or {}
        fit_results = results.get("fit_results", {})
        if not isinstance(fit_results, dict):
            fit_results = {}

        metadata = {
            "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC"),
            "Node": node_name,
            "Duration": f"{elapsed:.1f}s",
            "Status": "completed" if run_error is None else "completed with errors",
        }
        if run_error is not None:
            metadata["Error"] = f"`{type(run_error).__name__}: {run_error}`"

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

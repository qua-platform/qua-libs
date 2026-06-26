"""Fixtures for CMA-ES analysis tests.

Mirrors the Loss-DiVincenzo analysis conftest but points the calibration
library root at the ``cmaes`` node directory and provides a bespoke
analysis runner for the CMA-ES optimisation node.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import matplotlib
import numpy as np
import pytest
import xarray as xr

CURRENT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = CURRENT_DIR.parents[3]  # tests/analysis/

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
    call_node_action,
    ensure_quam_config_stub,
    get_parameters_dict,
    load_library_node,
    make_save_analysis_plot,
    patch_action_manager_register_only,
    patch_qualibrate_logger,
    reimport_node_to_register_actions,
    setup_test_cache,
)
from quam_factory import create_ld_quam  # noqa: E402

# ── Cache setup ────────────────────────────────────────────────────────
_cache_base = ANALYSIS_ROOT / ".pytest_cache"
setup_test_cache(_cache_base)
patch_qualibrate_logger(_cache_base)

matplotlib.use("Agg")

# ── Paths and defaults ─────────────────────────────────────────────────

CALIBRATION_LIBRARY_ROOT = (
    REPO_ROOT
    / "qualibration_graphs"
    / "quantum_dots"
    / "calibrations"
    / "cmaes"
)
ARTIFACTS_BASE = ANALYSIS_ROOT / "artifacts"

QUBIT_PAIR_NAMES: list[str] = ["q1_q2"]


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture returning a ``LossDiVincenzoQuam`` with default macros."""

    def _factory():
        return create_ld_quam()

    return _factory


@pytest.fixture
def save_analysis_plot():
    return make_save_analysis_plot()


# ── Pytest hooks ───────────────────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "analysis: mark test as an analysis test",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/analysis/" in str(item.fspath) and not item.get_closest_marker(
            "analysis"
        ):
            item.add_marker(pytest.mark.analysis)

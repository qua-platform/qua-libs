"""Runtime compatibility aliases for qualibrate module path changes.

Loaded automatically by Python at startup when present on ``sys.path``.
This keeps legacy imports used in this repo working with newer qualibrate
package layouts during test runs.
"""

from __future__ import annotations

import importlib
import sys


def _alias_module(old_name: str, new_name: str) -> None:
    """Alias ``old_name`` to ``new_name`` if the old module is unavailable."""
    try:
        importlib.import_module(old_name)
        return
    except Exception:
        pass

    try:
        sys.modules[old_name] = importlib.import_module(new_name)
    except Exception:
        return


_alias_module("qualibrate.parameters", "qualibrate.core.parameters")
_alias_module("qualibrate.qualibration_library", "qualibrate.core.qualibration_library")
_alias_module("qualibrate.qualibration_node", "qualibrate.core.qualibration_node")
_alias_module("qualibrate.qualibration_graph", "qualibrate.core.qualibration_graph")
_alias_module("qualibrate.runnables", "qualibrate.core.runnables")
_alias_module("qualibrate.utils.logger_m", "qualibrate.core.utils.logger_m")

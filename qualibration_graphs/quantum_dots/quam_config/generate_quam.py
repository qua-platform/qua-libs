"""
General purpose script to generate the wiring and the QUAM that corresponds to your experiment for the first time.
The workflow is as follows:
    - Copy the content of the wiring example corresponding to your architecture and paste it here.
    - Modify the statis parameters to match your network configuration.
    - Update the instrument setup section with the available hardware.
    - Define which qubit ids are present in the system.
    - Define any custom/hardcoded channel addresses.
    - Allocate the wiring to the connectivity object based on the available instruments.
    - Visualize and validate the resulting connectivity.
    - Build the wiring and QUAM.
    - Populate the generated quam with initial values by modifying and running populate_quam_xxx.py
"""


from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring, visualize

from quam_builder.architecture.quantum_dots.operations.macro_catalog import (
    MacroCatalog,
    VoltageBalancedMacroCatalog,
)
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD, LossDiVincenzoQuam
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.quantum_dots import build_quam

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DIR = Path(__file__).resolve().parent

CLUSTER_CONFIG_PATH = DIR / ".qm_cluster_config.json"

DEFAULT_QUAM_STATE_DIR = DIR / "quam_state"
"""Directory for ``state_old.json`` / ``wiring_old.json`` ."""

# ---------------------------------------------------------------------------
# Macro catalog selection
# ---------------------------------------------------------------------------

MacroCatalogName = Literal["default", "voltage_balanced"]

MACRO_CATALOG: MacroCatalogName = "voltage_balanced"
"""Which macro catalog to wire onto the test machine.

- ``"default"``:           Built-in :class:`DefaultMacroCatalog` (priority 100).
- ``"voltage_balanced"``:  Adds :class:`VoltageBalancedMacroCatalog` (priority 200),
                           overriding default state/drive/gate macros with
                           DC-balanced implementations.

Change this value to switch every execute and simulation test between catalogs.
"""


def _resolve_macro_catalogs(
    name: MacroCatalogName = MACRO_CATALOG,
) -> Sequence[MacroCatalog] | None:
    """Return the extra catalog list to pass to ``build_quam(catalogs=...)``."""
    if name == "default":
        return None
    if name == "voltage_balanced":
        return [VoltageBalancedMacroCatalog()]
    raise ValueError(f"Unknown macro catalog name: {name!r}")


def _load_cluster_config() -> tuple[str, str]:
    if not CLUSTER_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CLUSTER_CONFIG_PATH}. "
            "Copy tests/.qm_cluster_config.json.example and set host/cluster_name."
        )
    raw: dict[str, Any] = json.loads(CLUSTER_CONFIG_PATH.read_text(encoding="utf-8"))
    return str(raw["host"]), str(raw["cluster_name"])


def build_machine(
    path: Path | None = None,
    *,
    save: bool = True,
    macro_catalog: MacroCatalogName = MACRO_CATALOG,
    plot=True
) -> LossDiVincenzoQuam:
    """Build a ``LossDiVincenzoQuam`` using the same wiring recipe as ``qm_example``.

    Topology: 4 plunger dots, 3 pairs, 2 sensors, MW/LF FEMs, shared MW line,
    reservoir barriers ``rb``, and the same ``qubit_pair_sensor_map`` as the example.

    Args:
        path: Directory passed to ``build_quam_wiring`` / ``build_quam`` saves.
              Defaults to :data:`DEFAULT_QUAM_STATE_DIR`.
        save: If True, persists after ``build_quam`` (honours ``path``).
        macro_catalog: Which macro catalog to use (see :data:`MACRO_CATALOG`).

    Returns:
        Fully built machine before :func:`update_machine` runs (call that separately).
    """
    connectivity = Connectivity()
    connectivity.add_quantum_dots(quantum_dots=[1, 2, 3, 4])
    connectivity.add_quantum_dot_drive_lines(
        quantum_dots=[1, 2, 3, 4], shared_line=True, use_mw_fem=True
    )
    connectivity.add_sensor_dots(sensor_dots=[1, 2], shared_resonator_line=False)

    connectivity.add_quantum_dot_pairs(quantum_dot_pairs=[(1, 2), (2, 3), (3, 4)])

    instruments = Instruments()
    instruments.add_mw_fem(controller=1, slots=[2])
    instruments.add_lf_fem(controller=1, slots=[3, 5]) # 5, 6 for cs4

    allocate_wiring(connectivity, instruments)

    host, cluster_name = _load_cluster_config()
    dest = path if path is not None else DEFAULT_QUAM_STATE_DIR

    machine = build_quam_wiring(
        connectivity,
        host,
        cluster_name,
        BaseQuamQD(),
        path=dest,
    )

    machine = build_quam(
        machine,
        qubit_pair_sensor_map={
            "q1_q2": ["sensor_1"],
            "q2_q3": ["sensor_1"],
            "q3_q4": ["sensor_2"],
        },
        catalogs=_resolve_macro_catalogs(macro_catalog),
        save=save,
        path=dest,
    )

    # Optional: Visualize Wiring
    if plot:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt  # noqa: E402

        visualize(
            connectivity.elements,
            available_channels=instruments.available_channels,
            use_matplotlib=True,
        )
        plt.show()

    return machine

build_machine(save=False)
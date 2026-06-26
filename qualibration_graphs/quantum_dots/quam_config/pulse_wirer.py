
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
from typing import Any, Literal, List

import numpy as np
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring, visualize

from quam_builder.architecture.quantum_dots.operations.macro_catalog import (
    MacroCatalog,
    VoltageBalancedMacroCatalog,
)
from quam_builder.architecture.quantum_dots.operations.names import (
    DrivePulseName,
    SingleQubitMacroName,
    VoltagePointName,
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

# Align LF vs MW output timing in the QM pulse config (matches ``quam_factory``).
LF_FEM_DELAY_NS: int = 161
MW_FEM_DELAY_NS: int = 0

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

DEFAULT_QUAM_STATE_DIR = DIR / "quam_state"
"""Directory for ``state_old.json`` / ``wiring_old.json`` ."""

dest = DEFAULT_QUAM_STATE_DIR

loaded = LossDiVincenzoQuam.load(dest)
loaded.generate_config()

config = loaded.generate_config()
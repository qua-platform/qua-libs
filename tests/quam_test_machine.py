"""Build-on-disk QUAM state matching ``qm_example``, then reload for tests.

Workflow
--------
1. :func:`build_machine` mirrors ``quam_builder/.../examples/qm_example.py`` —
   Connectivity → ``build_quam_wiring`` → ``build_quam`` — and writes split JSON under
   ``DEFAULT_QUAM_STATE_DIR`` when ``save`` is True.

2. :func:`update_machine` registers placeholder tuning points and LF/MW FEM port delays.

3. Quantum-dot detuning axes come from ``BaseQuamQD.register_quantum_dot_pair``
   (see quam-builder); they are serialized in ``state_old.json`` when you save.

4. Callers load with ``LossDiVincenzoQuam.load(...)`` and ``machine.generate_config()``.

This module does **not** patch ``get_quam_config`` (unlike ``qualibration_graphs/.../quam_factory.py``).
Passes explicit ``path=`` everywhere so saves do not rely on ~/.quam.

Typical usage::

    from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam
    from tests.quam_test_machine import DEFAULT_QUAM_STATE_DIR

    machine = LossDiVincenzoQuam.load(DEFAULT_QUAM_STATE_DIR)
    qm_config = machine.generate_config()

Populate ``tests/quam_machine_state/`` once (or from CI) with::

    python tests/quam_test_machine.py

Loss-DiVincenzo execute/simulation pytest fixtures call
:func:`regenerate_state_directory` on each run (overwriting JSON), so manual
pre-generation is optional when using those tests.

Requires ``tests/.qm_cluster_config.json`` (copy from ``.example``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring

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

TESTS_DIR = Path(__file__).resolve().parent

CLUSTER_CONFIG_PATH = TESTS_DIR / ".qm_cluster_config.json"

DEFAULT_QUAM_STATE_DIR = TESTS_DIR / "quam_machine_state"
"""Directory for ``state_old.json`` / ``wiring_old.json`` (qm_example-aligned layout)."""

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
    connectivity.add_quantum_dots(quantum_dots=[1, 2])
    connectivity.add_quantum_dot_drive_lines(
        quantum_dots=[1, 2], shared_line=True, use_mw_fem=True
    )
    connectivity.add_sensor_dots(sensor_dots=[1], shared_resonator_line=False)

    connectivity.add_quantum_dot_pairs(quantum_dot_pairs=[(1, 2)])

    instruments = Instruments()
    instruments.add_mw_fem(controller=1, slots=[1])
    instruments.add_lf_fem(controller=1, slots=[5])

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
            "q1_q2": ["sensor_1"]
        },
        catalogs=_resolve_macro_catalogs(macro_catalog),
        save=save,
        path=dest,
    )
    return machine


def _apply_fem_output_port_delays(machine: LossDiVincenzoQuam) -> None:
    """Set per-FEM analog output delays (LF path skew vs MW)."""
    for controller_ports in machine.ports.analog_outputs.values():
        for fem_ports in controller_ports.values():
            for port in fem_ports.values():
                port.delay = LF_FEM_DELAY_NS

    for controller_ports in machine.ports.mw_outputs.values():
        for fem_ports in controller_ports.values():
            for port in fem_ports.values():
                port.delay = MW_FEM_DELAY_NS


def _register_placeholder_voltage_points(machine: LossDiVincenzoQuam) -> None:
    """Placeholder tuning-point voltages consumed by wired macros (nodes override at runtime)."""
    for qubit in machine.qubits.values():
        dot_id = qubit.quantum_dot.id
        qubit.add_point(VoltagePointName.INITIALIZE, {dot_id: 0.075}, duration=248)
        qubit.add_point(VoltagePointName.MEASURE, {dot_id: -0.05}, duration=248)
        qubit.add_point(VoltagePointName.EMPTY, {dot_id: -0.1}, duration=524)
        qubit.add_point(VoltagePointName.EXCHANGE, {dot_id: 0.025}, duration=248)

        qubit.x.update(pi_amplitude=0.5 * qubit.x.pi_pulse.amplitude)
        qubit.x.update(frequency=5e9 - 50)
        qubit.x.update(duration=248)

    for qdp in machine.quantum_dot_pairs.values():
        dot_ids = [d.id for d in qdp.quantum_dots]
        barrier_id = qdp.barrier_gate.id
        qdp.add_point(
            VoltagePointName.INITIALIZE,
            {**{did: 0.075 for did in dot_ids}, barrier_id: 0.0},
            duration=248,
        )
        qdp.add_point(
            VoltagePointName.MEASURE,
            {**{did: -0.05 for did in dot_ids}, barrier_id: 0.0},
            duration=248,
        )
        qdp.add_point(
            VoltagePointName.EMPTY,
            {**{did: -0.1 for did in dot_ids}, barrier_id: 0.0},
            duration=524,
        )
        qdp.add_point(
            VoltagePointName.EXCHANGE,
            {**{did: 0.025 for did in dot_ids}, barrier_id: 0.05},
            duration=248,
        )

    for qp in machine.qubit_pairs.values():
        qdp = qp.quantum_dot_pair
        dot_ids = [d.id for d in qdp.quantum_dots]
        barrier_id = qdp.barrier_gate.id
        qp.add_point(
            VoltagePointName.INITIALIZE,
            {**{did: 0.075 for did in dot_ids}, barrier_id: 0.0},
            duration=248,
        )
        qp.add_point(
            VoltagePointName.MEASURE,
            {**{did: -0.05 for did in dot_ids}, barrier_id: 0.0},
            duration=248,
        )
        qp.add_point(
            VoltagePointName.EMPTY,
            {**{did: -0.1 for did in dot_ids}, barrier_id: 0.0},
            duration=524,
        )
        qp.add_point(
            VoltagePointName.EXCHANGE,
            {**{did: 0.025 for did in dot_ids}, barrier_id: 0.0},
            duration=248,
        )
        qp.add_point(
            "CZ",
            {**{did: 0.075 for did in dot_ids}, barrier_id: 0.2},
            duration=248,
        )
        qp.macros["cz"].wait_duration = 200

    for qdp_name, qdp in machine.quantum_dot_pairs.items():
        for sd in qdp.sensor_dots:
            sd._add_readout_params(qdp_name, threshold=0.0)

    for qubit in machine.qubits.values():
        qubit.macros[VoltagePointName.MEASURE].hold_duration = 248
        qubit.macros[SingleQubitMacroName.XY_DRIVE].pulse_family = (
            DrivePulseName.GAUSSIAN.value
        )


def update_machine(machine: LossDiVincenzoQuam) -> LossDiVincenzoQuam:
    """Placeholder tuning points + LF/MW FEM output delays (``quam_factory``-aligned)."""
    _register_placeholder_voltage_points(machine)
    _apply_fem_output_port_delays(machine)
    return machine


def regenerate_state_directory(
    path: Path | None = None,
) -> tuple[LossDiVincenzoQuam, dict[str, Any]]:
    """Build :func:`build_machine`, apply :func:`update_machine`, persist, reload.

    Always writes fresh JSON under ``dest`` (overwrites ``state_old.json`` / siblings if
    present). :func:`update_machine` runs on the built machine **before** the final
    ``machine.save(...)``, then the machine is loaded again from disk so tests
    exercise the saved snapshot.

    Returns the loaded machine and its QM config. The reload step mirrors how tests
    hydrate from disk before ``generate_config``.
    """
    dest = path if path is not None else DEFAULT_QUAM_STATE_DIR
    machine = update_machine(build_machine(dest, save=True))
    machine.save(dest)
    loaded = LossDiVincenzoQuam.load(dest)
    return loaded, loaded.generate_config()


if __name__ == "__main__":
    m, cfg = regenerate_state_directory()
    print(f"Wrote QUAM state under {DEFAULT_QUAM_STATE_DIR}")
    print(f"QUA/QM config keys (top-level): {sorted(cfg.keys())[:12]} …")

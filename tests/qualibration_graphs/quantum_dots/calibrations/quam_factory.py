"""Unified wiring-based QuAM factory for quantum dot calibration tests.

Builds machines using the ``quam_builder`` wiring tools and the
``release/nightly`` default macro engine.  Two entry points are provided:

* :func:`create_qd_quam` -- Stage 1 dot-layer ``BaseQuamQD``
  (quantum dots, sensor dots, virtual gate set, dot pairs).
  Used by gate-virtualization / virtual-gate-subgraph tests.

* :func:`create_ld_quam` -- Stage 2 full ``LossDiVincenzoQuam``
  (adds qubits, XY drives, default pulses, and default macros).
  Used by loss-DiVincenzo calibration tests.

Hardware configuration
----------------------
All FEM slot numbers, controller identifiers, and topology constants live
as module-level variables at the top of this file.  Change them here to
reconfigure for a different instrument rack -- nothing is hard-coded deeper
in the factory functions.
"""

from __future__ import annotations

from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring

from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.architecture.quantum_dots.operations.default_macros.single_qubit_macros import (
    Measure1QMacro,
)
from quam_builder.architecture.quantum_dots.operations.names import VoltagePointName
from quam_builder.builder.quantum_dots import (
    build_base_quam,
    build_loss_divincenzo_quam,
)

# ── Hardware Configuration ──────────────────────────────────────────────
# Change these constants to match your instrument rack.
# All FEM slot and controller IDs are defined here; nothing is hardcoded
# deeper in the factory functions.

HOST_IP: str = "172.16.33.115"
"""QOP IP address used for ``machine.network``."""

CLUSTER_NAME: str = "CS_3"
"""OPX cluster name."""

CONTROLLER_ID: int = 1
"""Controller number passed to ``Instruments.add_*_fem(controller=...)``."""

MW_FEM_SLOT: int = 1
"""MW-FEM slot for qubit XY drive lines (Stage 2 only)."""

LF_FEM_SLOT_1: int = 3
"""LF-FEM slot for dot pair 1 (plungers 1-2, sensor 1, resonator 1)."""

LF_FEM_SLOT_2: int = 5
"""LF-FEM slot for dot pair 2 (plungers 3-4, sensor 2, resonator 2)."""

# ── Quantum-dot topology ───────────────────────────────────────────────

SENSOR_DOTS: list[int] = [1, 2]
"""Sensor dot indices passed to ``Connectivity.add_sensor_dots``."""

QUANTUM_DOTS: list[int] = [1, 2, 3, 4]
"""Quantum dot (plunger) indices passed to ``Connectivity.add_quantum_dots``."""

QUANTUM_DOT_PAIRS: list[tuple[int, int]] = [(1, 2), (3, 4)]
"""Quantum dot pair tuples passed to ``Connectivity.add_quantum_dot_pairs``."""

QUBIT_PAIR_SENSOR_MAP: dict[str, list[str]] = {
    "q1_q2": ["sensor_1"],
    "q3_q4": ["sensor_2"],
}
"""Maps qubit-pair IDs to their readout sensor(s) for Stage 2."""


# ── Factory functions ──────────────────────────────────────────────────


def create_qd_quam() -> BaseQuamQD:
    """Build a Stage-1 ``BaseQuamQD`` with the dot layer only.

    Creates quantum dots, sensor dots with readout resonators, a virtual
    gate set with an identity compensation matrix, and quantum-dot pairs.
    No qubits, XY drives, or macros are added.

    The returned machine is suitable for gate-virtualization and
    virtual-gate-subgraph calibration tests that operate on the dot layer.
    """
    connectivity = Connectivity()
    connectivity.add_sensor_dots(
        sensor_dots=SENSOR_DOTS,
        shared_resonator_line=False,
        use_mw_fem=False,
    )
    connectivity.add_quantum_dots(
        quantum_dots=QUANTUM_DOTS,
        add_drive_lines=False,
    )
    connectivity.add_quantum_dot_pairs(quantum_dot_pairs=QUANTUM_DOT_PAIRS)

    instruments = Instruments()
    instruments.add_lf_fem(
        controller=CONTROLLER_ID,
        slots=[LF_FEM_SLOT_1, LF_FEM_SLOT_2],
    )

    allocate_wiring(connectivity, instruments)

    machine = BaseQuamQD()
    machine = build_quam_wiring(connectivity, HOST_IP, CLUSTER_NAME, machine)
    machine = build_base_quam(machine, connect_qdac=False, save=False)
    return machine


def create_ld_quam():
    """Build a Stage-2 ``LossDiVincenzoQuam`` with qubits and default macros.

    Internally calls :func:`create_qd_quam` for the dot layer, then adds
    qubit XY drive lines, registers qubits (q1-q4), wires default pulses
    (x180, x90, y180, y90, ...) and default macros (initialize, measure,
    empty, xy_drive, x, y, z, x180, x90, ...) via
    ``wire_machine_macros()``.

    Returns a fully configured ``LossDiVincenzoQuam`` ready for
    loss-DiVincenzo calibration tests.
    """
    base_machine = create_qd_quam()

    connectivity = Connectivity()
    connectivity.add_sensor_dots(
        sensor_dots=SENSOR_DOTS,
        shared_resonator_line=False,
        use_mw_fem=False,
    )
    connectivity.add_quantum_dots(
        quantum_dots=QUANTUM_DOTS,
        add_drive_lines=True,
        use_mw_fem=False,
        shared_drive_line=True,
    )
    connectivity.add_quantum_dot_pairs(quantum_dot_pairs=QUANTUM_DOT_PAIRS)

    instruments = Instruments()
    instruments.add_lf_fem(
        controller=CONTROLLER_ID,
        slots=[LF_FEM_SLOT_1, LF_FEM_SLOT_2],
    )

    allocate_wiring(connectivity, instruments)

    machine = build_quam_wiring(connectivity, HOST_IP, CLUSTER_NAME, base_machine)
    machine = build_loss_divincenzo_quam(
        machine,
        qubit_pair_sensor_map=QUBIT_PAIR_SENSOR_MAP,
        implicit_mapping=True,
        save=False,
    )

    # The builder sets qubit.id to the quantum-dot name (e.g. "virtual_dot_1")
    # but downstream code expects qubit.name to equal the dict key ("q1").
    for key, qubit in machine.qubits.items():
        qubit.id = key

    _add_default_voltage_points(machine)
    return machine


def _add_default_voltage_points(machine) -> None:
    """Register canonical voltage tuning points consumed by state macros.

    The default ``Initialize1QMacro``, ``Measure1QMacro``, and
    ``Empty1QMacro`` from quam-builder invoke ``step_to_point`` /
    ``ramp_to_point`` on the owning qubit.  Those calls require named
    voltage points to be registered beforehand.

    The actual voltage values here are nominal placeholders; calibration
    nodes override them at run time.
    """
    for qubit in machine.qubits.values():
        dot_id = qubit.quantum_dot.id
        qubit.with_step_point(VoltagePointName.INITIALIZE.value, {dot_id: 0.10}, duration=200)
        qubit.with_step_point(VoltagePointName.EMPTY.value, {dot_id: 0.00}, duration=180)
        # For "measure", register the voltage point but install Measure1QMacro
        # (delegates to quantum_dot_pair) instead of the generic StepPointMacro.
        qubit.add_point(VoltagePointName.MEASURE.value, {dot_id: 0.15}, duration=220)
        qubit.macros[VoltagePointName.MEASURE.value] = Measure1QMacro()

    # Register voltage points on quantum_dot_pairs so the MeasureStateMacro
    # can call step_to_point("measure") during the delegation chain.
    for qdp in machine.quantum_dot_pairs.values():
        dot_ids = [d.id for d in qdp.quantum_dots]
        voltages = {did: 0.15 for did in dot_ids}
        qdp.add_point(VoltagePointName.MEASURE.value, voltages, duration=220)

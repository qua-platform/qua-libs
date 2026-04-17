"""Unified wiring-based QuAM factory for quantum dot calibration tests.

Builds machines using the ``quam_builder`` wiring tools and the
``release/nightly`` default macro engine.  Two entry points are provided:

* :func:`create_qd_quam` -- Stage 1 dot-layer ``BaseQuamQD``
  (quantum dots 1-2, sensor dot 1, virtual gate set, dot pair 1-2).
  Used by gate-virtualization / virtual-gate-subgraph tests.

* :func:`create_ld_quam` -- Stage 2 full ``LossDiVincenzoQuam``
  (adds qubits 1-2, MW-FEM XY drives, default pulses, and default macros).
  Used by loss-DiVincenzo calibration tests.

Hardware configuration
----------------------
Cluster connection (host IP, cluster name) is loaded from
``tests/.qm_cluster_config.json`` -- copy the ``.example`` file and
fill in your values.  FEM slot numbers and topology constants live
as module-level variables at the top of this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring

from qualang_tools.wirer.wirer.wirer import ChannelSpec
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.architecture.quantum_dots.operations.names import (
    DrivePulseName,
    SingleQubitMacroName,
    VoltagePointName,
)
from quam_builder.builder.quantum_dots import (
    build_base_quam,
    build_loss_divincenzo_quam,
)

# ── Hardware Configuration ──────────────────────────────────────────────
# Cluster connection details are loaded at import time from
# ``tests/.qm_cluster_config.json`` (not tracked by git).
# Copy ``.qm_cluster_config.json.example`` and fill in your values.


def _find_repo_root(start: Path) -> Path:
    current = start
    while current != current.parent:
        if (current / "tests").is_dir() and (current / "qualibration_graphs").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate repo root")


def _load_cluster_config() -> tuple[str, str]:
    config_path = _find_repo_root(Path(__file__).resolve().parent) / "tests" / ".qm_cluster_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Cluster config not found at {config_path}. "
            "Copy tests/.qm_cluster_config.json.example and fill in your values."
        )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return data["host"], data["cluster_name"]


HOST_IP, CLUSTER_NAME = _load_cluster_config()

CONTROLLER_ID: int = 1
"""Controller number passed to ``Instruments.add_*_fem(controller=...)``."""

MW_FEM_SLOT: int = 4
"""MW-FEM slot for qubit XY drive lines (Stage 2 only)."""

LF_FEM_SLOT: int = 1
"""LF-FEM slot for plungers 1-2, sensor 1, and resonator 1."""

LF_FEM_DELAY_NS: int = 155
"""Delay (ns) applied to all LF-FEM analog output ports to compensate for MW-FEM path skew."""

MW_FEM_DELAY_NS: int = 0
"""Delay (ns) applied to all MW-FEM output ports."""

DEFAULT_LARMOR_FREQUENCY: float = 5.1e9
"""Nominal qubit Larmor (RF) frequency in Hz.

The MW-FEM upconverter is at 5 GHz, so this gives IF = 100 MHz —
well within the MW-FEM NCO limit of ±500 MHz.
"""

# ── Quantum-dot topology ───────────────────────────────────────────────

SENSOR_DOTS: list[int] = [1]
"""Sensor dot indices passed to ``Connectivity.add_sensor_dots``."""

QUANTUM_DOTS: list[int] = [1, 2]
"""Quantum dot (plunger) indices passed to ``Connectivity.add_quantum_dots``."""

QUANTUM_DOT_PAIRS: list[tuple[int, int]] = [(1, 2)]
"""Quantum dot pair tuples passed to ``Connectivity.add_quantum_dot_pairs``."""

QUBIT_PAIR_SENSOR_MAP: dict[str, list[str]] = {
    "q1_q2": ["sensor_1"],
}
"""Maps qubit-pair IDs to their readout sensor(s) for Stage 2."""


# ── Factory functions ──────────────────────────────────────────────────


def create_qd_quam() -> BaseQuamQD:
    """Build a Stage-1 ``BaseQuamQD`` with the dot layer only.

    Creates quantum dots 1-2, sensor dot 1 with readout resonator, a virtual
    gate set with an identity compensation matrix, and dot pair (1, 2).
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
        slots=[LF_FEM_SLOT],
    )

    allocate_wiring(connectivity, instruments)

    machine = BaseQuamQD()
    machine = build_quam_wiring(connectivity, HOST_IP, CLUSTER_NAME, machine)
    machine = build_base_quam(machine, connect_qdac=False, save=False)
    return machine


def create_ld_quam():
    """Build a Stage-2 ``LossDiVincenzoQuam`` with qubits and default macros.

    Internally calls :func:`create_qd_quam` for the dot layer, then adds
    MW-FEM XY drive lines, registers qubits (q1-q2), wires the default
    single-reference XY pulse and the default macros via
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
        use_mw_fem=True,
        shared_drive_line=True,
    )
    connectivity.add_quantum_dot_pairs(quantum_dot_pairs=QUANTUM_DOT_PAIRS)

    instruments = Instruments()
    instruments.add_mw_fem(controller=CONTROLLER_ID, slots=[MW_FEM_SLOT])
    instruments.add_lf_fem(
        controller=CONTROLLER_ID,
        slots=[LF_FEM_SLOT],
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

    _set_default_larmor_frequencies(machine)
    _define_default_detuning_axes(machine)
    _override_default_pulse_lengths(machine)
    _add_default_voltage_points(machine)
    _apply_port_delays(machine)
    return machine


def _set_default_larmor_frequencies(machine) -> None:
    """Set a nominal Larmor frequency on each qubit so the MW-FEM IF resolves.

    XYDriveMW.RF_frequency refs qubit.larmor_frequency; without a numeric
    value the reference string leaks into the QM config as a non-number.
    """
    for qubit in machine.qubits.values():
        if getattr(qubit, "larmor_frequency", None) is None:
            qubit.larmor_frequency = DEFAULT_LARMOR_FREQUENCY


def _apply_port_delays(machine) -> None:
    """Set per-FEM output port delays to align LF-FEM and MW-FEM paths."""
    for controller_ports in machine.ports.analog_outputs.values():
        for fem_ports in controller_ports.values():
            for port in fem_ports.values():
                port.delay = LF_FEM_DELAY_NS

    for controller_ports in machine.ports.mw_outputs.values():
        for fem_ports in controller_ports.values():
            for port in fem_ports.values():
                port.delay = MW_FEM_DELAY_NS


def _override_default_pulse_lengths(machine) -> None:
    """Override quam-builder default pulse lengths for this test configuration."""
    for qubit in machine.qubits.values():
        if hasattr(qubit, "xy") and qubit.xy is not None:
            gaussian_pulse = qubit.xy.operations.get(DrivePulseName.GAUSSIAN)
            if gaussian_pulse is not None:
                gaussian_pulse.length = 524
                gaussian_pulse.amplitude = 0.2
                if hasattr(gaussian_pulse, "sigma"):
                    gaussian_pulse.sigma = 524 / 6

    for sd in machine.sensor_dots.values():
        rr = getattr(sd, "readout_resonator", None)
        if rr is not None:
            rr.intermediate_frequency = 50e6
            if "readout" in getattr(rr, "operations", {}):
                rr.operations["readout"].length = 1000
                rr.operations["readout"].amplitude = 0.025


def _define_default_detuning_axes(machine) -> None:
    """Materialize a simple detuning axis for each registered dot pair."""
    for qdp in machine.quantum_dot_pairs.values():
        gate_set = qdp.voltage_sequence.gate_set
        if qdp.detuning_axis_name in getattr(gate_set, "valid_channel_names", []):
            continue
        qdp.define_detuning_axis(
            matrix=[[1.0, -1.0]],
            detuning_axis_name=qdp.detuning_axis_name,
            set_dc_virtual_axis=False,
        )

    for gate_set_id in machine.virtual_gate_sets:
        machine.reset_voltage_sequence(gate_set_id)


def _add_default_voltage_points(machine) -> None:
    """Register canonical voltage tuning points consumed by state macros.

    ``build_loss_divincenzo_quam()`` already wires the latest default macro
    instances. The test factory only needs to define the canonical points those
    macros consume and tune a few runtime defaults on the instantiated macros.

    The voltage values here are nominal placeholders; calibration nodes override
    them at run time.
    """
    for qubit in machine.qubits.values():
        dot_id = qubit.quantum_dot.id
        qubit.add_point(VoltagePointName.INITIALIZE, {dot_id: 0.075}, duration=248)
        qubit.add_point(VoltagePointName.MEASURE, {dot_id: 0.05}, duration=248)
        qubit.add_point(VoltagePointName.EMPTY, {dot_id: -0.05}, duration=524)
        qubit.add_point(VoltagePointName.EXCHANGE, {dot_id: 0.025}, duration=248)
        qubit.macros[VoltagePointName.INITIALIZE].ramp_duration = 16

    # Register canonical points on quantum-dot pairs so the latest pair macros
    # can dispatch by enum-backed names as well.
    for qdp in machine.quantum_dot_pairs.values():
        dot_ids = [d.id for d in qdp.quantum_dots]
        barrier_id = qdp.barrier_gate.id
        qdp.add_point(VoltagePointName.INITIALIZE, {**{did: 0.075 for did in dot_ids}, barrier_id: 0.0}, duration=248)
        qdp.add_point(VoltagePointName.MEASURE, {**{did: 0.05 for did in dot_ids}, barrier_id: 0.0}, duration=248)
        qdp.add_point(VoltagePointName.EMPTY, {**{did: -0.05 for did in dot_ids}, barrier_id: 0.0}, duration=524)
        qdp.add_point(VoltagePointName.EXCHANGE, {**{did: 0.025 for did in dot_ids}, barrier_id: 0.0}, duration=248)

    # Register the same canonical points on qubit pairs (LDQubitPair).
    # Qubit pairs have a different id (e.g. "q1_q2") than their underlying
    # quantum-dot pair ("virtual_dot_1_virtual_dot_2_pair"), so macros like
    # qubit_pair.empty() resolve to a different prefixed name ("q1_q2_empty").
    for qp in machine.qubit_pairs.values():
        qdp = qp.quantum_dot_pair
        dot_ids = [d.id for d in qdp.quantum_dots]
        barrier_id = qdp.barrier_gate.id
        qp.add_point(VoltagePointName.INITIALIZE, {**{did: 0.075 for did in dot_ids}, barrier_id: 0.0}, duration=248)
        qp.add_point(VoltagePointName.MEASURE, {**{did: 0.05 for did in dot_ids}, barrier_id: 0.0}, duration=248)
        qp.add_point(VoltagePointName.EMPTY, {**{did: -0.05 for did in dot_ids}, barrier_id: 0.0}, duration=524)
        qp.add_point(VoltagePointName.EXCHANGE, {**{did: 0.025 for did in dot_ids}, barrier_id: 0.0}, duration=248)

    # Populate default readout thresholds / projectors on sensor dots so
    # the SensorDotMeasureMacro can perform state discrimination.
    for qdp_name, qdp in machine.quantum_dot_pairs.items():
        for sd in qdp.sensor_dots:
            sd._add_readout_params(qdp_name, threshold=0.0)

    for qubit in machine.qubits.values():
        qubit.macros[VoltagePointName.MEASURE].hold_duration = 248
        qubit.macros[SingleQubitMacroName.XY_DRIVE].reference_pulse_name = DrivePulseName.GAUSSIAN

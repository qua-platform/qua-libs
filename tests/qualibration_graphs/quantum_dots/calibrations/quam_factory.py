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
Cluster connection (host IP, cluster name) is loaded from
``tests/.qm_cluster_config.json`` -- copy the ``.example`` file and
fill in your values.  FEM slot numbers and topology constants live
as module-level variables at the top of this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from contextlib import contextmanager
import warnings

from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring

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

MW_FEM_SLOT: int = 1
"""MW-FEM slot for qubit XY drive lines (Stage 2 only)."""

LF_FEM_SLOT_1: int = 5
"""LF-FEM slot for dot pair 1 (plungers 1-2, sensor 1, resonator 1)."""

LF_FEM_SLOT_2: int = 6
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


@contextmanager
def _suppress_known_quam_builder_warnings():
    """Hide known non-fatal QuAM builder warnings from the test factory."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This component is not part of any QuamRoot.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Could not get reference.*#/ports/analog_outputs.*",
            category=UserWarning,
        )
        yield


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
    with _suppress_known_quam_builder_warnings():
        machine = build_quam_wiring(connectivity, HOST_IP, CLUSTER_NAME, machine)
        machine = build_base_quam(machine, connect_qdac=False, save=False)
    return machine


def create_ld_quam():
    """Build a Stage-2 ``LossDiVincenzoQuam`` with qubits and default macros.

    Internally calls :func:`create_qd_quam` for the dot layer, then adds
    qubit XY drive lines, registers qubits (q1-q4), wires the default
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
    # TODO: To enable IQ driving on LF-FEM (XYDriveIQ), add an Octave to
    # instruments and set use_mw_fem=True.  The RF allocation path falls
    # through from MW-FEM to LF-FEM + Octave when no MW-FEM is configured.
    # With use_mw_fem=False the wirer allocates a single LF-FEM output per
    # drive line, producing XYDriveSingle (baseband EDSR).
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

    with _suppress_known_quam_builder_warnings():
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

        _override_default_pulse_lengths(machine)
        _add_default_voltage_points(machine)
    return machine


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
        qdp.add_point(VoltagePointName.INITIALIZE, {did: 0.075 for did in dot_ids}, duration=248)
        qdp.add_point(VoltagePointName.MEASURE, {did: 0.05 for did in dot_ids}, duration=248)
        qdp.add_point(VoltagePointName.EMPTY, {did: -0.05 for did in dot_ids}, duration=524)
        qdp.add_point(VoltagePointName.EXCHANGE, {did: 0.025 for did in dot_ids}, duration=248)

    # Populate default readout thresholds / projectors on sensor dots so
    # the SensorDotMeasureMacro can perform state discrimination.
    for qdp_name, qdp in machine.quantum_dot_pairs.items():
        for sd in qdp.sensor_dots:
            sd._add_readout_params(qdp_name, threshold=0.0)

    for qubit in machine.qubits.values():
        qubit.macros[VoltagePointName.MEASURE].hold_duration = 248
        qubit.macros[SingleQubitMacroName.XY_DRIVE].reference_pulse_name = DrivePulseName.GAUSSIAN

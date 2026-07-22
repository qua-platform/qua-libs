import numpy as np
from typing import List
from pathlib import Path

from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam
from quam_builder.architecture.quantum_dots.operations.names import (
    DrivePulseName,
    SingleQubitMacroName,
    VoltagePointName,
)

DIR = Path(__file__).resolve().parent

CLUSTER_CONFIG_PATH = DIR / ".qm_cluster_config.json"

DEFAULT_QUAM_STATE_DIR = DIR / "quam_state"
"""Directory for ``state_old.json`` / ``wiring_old.json`` ."""


# Align LF vs MW output timing in the QM pulse config (matches ``quam_factory``).
LF_FEM_DELAY_NS: int = 161
MW_FEM_DELAY_NS: int = 0

def populate_machine(machine: LossDiVincenzoQuam):

    #######################################
    ###### Qubits Physical Properties #####
    #######################################

    # XY / MW-FEM: QuAM uses IF = larmor_frequency - MW_upconverter (see XYDriveMW).
    # QM enforces |IF| <= 500 MHz. The old name ``LO`` here was really the *Larmor*
    # centre (~9.7 GHz), not the FEM LO; leaving upconverter at ~5 GHz made IF ~4.7 GHz.
    larmor_center_hz = 9.697371455e9
    mw_upconverter_hz = larmor_center_hz
    qubit_frequencies = [
        larmor_center_hz - 15e6,
        larmor_center_hz - 5e6,
        larmor_center_hz + 5e6,
        larmor_center_hz + 15e6,
    ]

    for i, q in enumerate(machine.qubits.values()):
        q.xy.opx_output.band = 3
        # Same params for each qubit for now. Subject to change.
        q.macros[VoltagePointName.INITIALIZE].update(ramp_duration=2000, hold_duration=200)
        q.macros[VoltagePointName.MEASURE].update(buffer_duration=240)
        q.macros[VoltagePointName.EMPTY].update(hold_duration=80)

        # MW FEM LO on this XY line (shared port → same value each iteration is fine).
        q.xy.opx_output.upconverter_frequency = mw_upconverter_hz

        # Absolute drive / Larmor frequency (RF), not the OPX IF.
        q_xy = q.macros[SingleQubitMacroName.XY_DRIVE]
        q_xy.update(frequency=qubit_frequencies[i])

        q.xy.operations[f"{DrivePulseName.GAUSSIAN}_x90"].amplitude = 0.17

        # Default values
        q.T1 = 1e-6
        q.T2ramsey = 0.5e-6
        q.T2echo = 2e-6

    #########################
    ###### State Points #####
    #########################

    for i, qdp in enumerate(machine.quantum_dot_pairs.values()):
        qdp.add_point(
            point_name=VoltagePointName.INITIALIZE,
            voltages={d.id: (i + 1) * 0.015 for d in qdp.quantum_dots},
            duration=1000,
        )
        qdp.add_point(
            point_name=VoltagePointName.EMPTY,
            voltages={d.id: (i + 1) * 0.02 for d in qdp.quantum_dots},
            duration=1500,
        )
        qdp.add_point(
            point_name=VoltagePointName.MEASURE,
            voltages={d.id: (i + 1) * 0.025 for d in qdp.quantum_dots},
            duration=1000,
        )
        qdp.add_point(
            point_name=VoltagePointName.EXCHANGE,
            voltages={d.id: (i + 1) * -0.025 for d in qdp.quantum_dots},
            duration=1000,
        )

    ##############################
    ###### Sensor Properties #####
    ##############################

    resonator_frequencies = [300.78e6, 436.542e6]
    for i, s in enumerate(machine.sensor_dots.values()):
        s.readout_resonator.intermediate_frequency = resonator_frequencies[i]
        s.readout_resonator.operations["readout"].amplitude = 0.02
        s.readout_resonator.operations["readout"].length = 50_000  # 50us

    ################################
    ###### Compensation Matrix #####
    ################################

    full_given_matrix = np.array(
        [
            [1.49696, 0.5218, 0.36891, 1.0, -0.15019, 0.11477, 0.02468],
            [-0.54456, 0.4782, 0.33809, 1.0, 0.01011, 0.04221, 0.09137],
            [-0.55239, -0.58, 0.58994, 1.0, 0.08125, -0.14962, 0.02272],
            [-0.40001, -0.42, -1.29694, 1.0, 0.05883, -0.00736, -0.13877],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    inverse_matrix = np.linalg.inv(full_given_matrix)
    barrier_orthogonalising_submatrix = -full_given_matrix[:4, 4:]

    gate_set_id = next(iter(machine.virtual_gate_sets))
    vgs = machine.virtual_gate_sets[gate_set_id]
    qds = machine.quantum_dots

    # Orthogonalise the barriers. Detuning will be another layer.
    machine.update_cross_compensation_submatrix(
        virtual_names=["virtual_barrier_1", "virtual_barrier_2", "virtual_barrier_3"],
        channels=[
            qds["virtual_dot_1"].physical_channel,
            qds["virtual_dot_2"].physical_channel,
            qds["virtual_dot_3"].physical_channel,
            qds["virtual_dot_4"].physical_channel,
        ],
        matrix=barrier_orthogonalising_submatrix,
        target="opx",
    )

    return machine


def update_machine(machine: LossDiVincenzoQuam) -> LossDiVincenzoQuam:
    """Placeholder tuning points + LF/MW FEM output delays (``quam_factory``-aligned)."""
    populate_machine(machine)
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


if __name__ == "__main__":
    m, cfg = regenerate_state_directory()
    print(f"Wrote QUAM state under {DEFAULT_QUAM_STATE_DIR}")
    print(f"QUA/QM config keys (top-level): {sorted(cfg.keys())[:12]} …")
    m.physical_channels['plunger_1'].opx_output.output_mode


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

    #################################
    ###### Define Detuning Axis #####
    #################################

    update_detuning_axis(machine, inverse_matrix)

    vgs.add_to_layer(
        source_gates=["delta_2134"],
        target_gates=[qd.id for qd in machine.quantum_dots.values()],
        layer_id="quantum_dot_pair_detuning_matrix",
        matrix=inverse_matrix[3:4, :4],
    )

    vgs.layers[-1].source_gates = ['virtual_dot_1_virtual_dot_2_pair', 'virtual_dot_2_virtual_dot_3_pair', 'virtual_dot_3_virtual_dot_4_pair', 'delta_2134']
    return machine


def update_detuning_axis(
    machine: LossDiVincenzoQuam,
    full_matrix: List[List[float]],
):
    vgs = machine.virtual_gate_sets[next(iter(machine.virtual_gate_sets))]
    target_gates = [qd.id for qd in machine.quantum_dots.values()]
    for i, qdp in enumerate(machine.quantum_dot_pairs.values()):
        source_gates = [qdp.detuning_axis_name]
        matrix = full_matrix[i : i + 1, :4]
        vgs.add_to_layer(
            source_gates=source_gates,
            target_gates=target_gates,
            matrix=matrix,
            layer_id="quantum_dot_pair_detuning_matrix",
        )

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


if __name__ == "__main__":
    m, cfg = regenerate_state_directory()
    print(f"Wrote QUAM state under {DEFAULT_QUAM_STATE_DIR}")
    print(f"QUA/QM config keys (top-level): {sorted(cfg.keys())[:12]} …")
    m.physical_channels['plunger_1'].opx_output.output_mode
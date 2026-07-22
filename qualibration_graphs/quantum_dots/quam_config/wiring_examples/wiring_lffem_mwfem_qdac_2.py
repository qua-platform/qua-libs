"""
Build the same C12 Loss-DiVincenzo QUAM state as ``quam_demo_clean.py``, using the
**qualang_tools wirer framework** for channel allocation.

The wirer handles *which OPX/QDAC ports* each element uses (``allocate_wiring``).
Component *names* (``g1``, ``dot1``, ``q1_xy``, …) are then assigned explicitly via
``register_quam_from_wiring()``, producing a QUAM state identical to the manual demo.

Workflow
--------
1. Declare connectivity + allocate channels  (wirer)
2. ``build_quam_wiring()``                   (wiring JSON → QUAM)
3. ``register_quam_from_wiring()``           (same names as quam_demo_clean.py)
4. Post-build: detuning layer, voltage points, QDAC connection
"""

import matplotlib.pyplot as plt  # only needed if visualize() is enabled below
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring, visualize
from qualang_tools.wirer.wirer.channel_specs import *
from quam.components import StickyChannelAddon, pulses, DigitalOutputChannel
from quam.components.ports import LFFEMAnalogOutputPort, LFFEMAnalogInputPort
from quam_builder.architecture.quantum_dots.components import XYDriveMW
from quam_builder.architecture.quantum_dots.components.voltage_gate import VoltageGate
from quam_builder.architecture.quantum_dots.components.dac_spec import QdacSpec
from quam_builder.architecture.quantum_dots.components.readout_resonator import ReadoutResonatorSingle
from quam_builder.architecture.quantum_dots.components.reservoir import DrainSingle, SourceSingle
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam
from quam_builder.architecture.quantum_dots.macro_engine import wire_machine_macros
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.quantum_dots.build_quam import add_ports


def qdac_config(ip: str) -> dict:
    return {
        "driver_module": "qcodes_contrib_drivers.drivers.QDevil.QDAC2",
        "driver_class": "QDac2",
        "connection": {"visalib": "@py", "address": f"TCPIP::{ip}::5025::SOCKET"},
        "channel_method": "channel",
        "accessor": "dc_constant_V",
        "is_qdac": True,
    }


def plunger_constraint(lf_fem: int, out_port: int, trigger_port: int):
    """LF-FEM analog output + digital trigger, matching quam_demo_clean gate wiring."""
    return (
        lf_fem_spec(con=1, out_slot=lf_fem, out_port=out_port)
        & lf_fem_dig_spec(con=1, slot=lf_fem, out_port=trigger_port)
    )


def register_quam_from_wiring(
    machine: LossDiVincenzoQuam,
    lf_fem: int,
    plunger_wiring: dict,
    qdac_name: str = "qdac1",
) -> tuple[dict[int, VoltageGate], SourceSingle, DrainSingle, VoltageGate]:
    """Register QUAM components using wirer port refs and manual-demo naming."""

    def plunger_path(n: int) -> str:
        return f"#/wiring/qubits/q{n}/p"

    def xy_path(n: int) -> str:
        return f"#/wiring/qubits/q{n}/xy"

    gates: dict[int, VoltageGate] = {}
    for n, spec in plunger_wiring.items():
        wiring = machine.wiring["qubits"][f"q{n}"]["p"]
        gate = VoltageGate(
            id=f"g{n}",
            opx_output=f"{plunger_path(n)}/opx_output",
            dac_spec=QdacSpec(
                output_port=spec["qdac_port"],
                dac_name=qdac_name,
                qdac_trigger_in=spec["trigger"],
            ),
            sticky=StickyChannelAddon(duration=16, digital=False),
        )
        if wiring.get("digital_output"):
            gate.digital_outputs["qdac_trig"] = DigitalOutputChannel(
                opx_output=f"{plunger_path(n)}/digital_output",
                delay=0,
                buffer=0,
            )
        gates[n] = gate

    drain = DrainSingle(
        id="drain",
        opx_output="#/wiring/globals/drain/g/opx_output",
        dac_spec=QdacSpec(output_port=11, dac_name=qdac_name),
    )
    drain_wiring = machine.wiring["globals"]["drain"]["g"]
    if drain_wiring.get("digital_output"):
        drain.digital_outputs["qdac_trig"] = DigitalOutputChannel(
            opx_output="#/wiring/globals/drain/g/digital_output",
            delay=0,
            buffer=0,
        )

    source = SourceSingle(id="source", opx_output=None, dac_spec=QdacSpec(output_port=10, dac_name=qdac_name))
    g6 = VoltageGate(id="g6", opx_output=None, dac_spec=QdacSpec(output_port=6, dac_name=qdac_name))

    resonator_1 = ReadoutResonatorSingle(
        id="rr1",
        intermediate_frequency=120e6,
        operations={"readout": pulses.SquareReadoutPulse(length=200, id="readout", amplitude=0.01)},
        opx_output=LFFEMAnalogOutputPort("con1", lf_fem, port_id=8, upsampling_mode="mw"),
        opx_input=LFFEMAnalogInputPort("con1", lf_fem, port_id=1),
    )
    resonator_2 = ReadoutResonatorSingle(
        id="rr2",
        intermediate_frequency=500e6,
        operations={"readout": pulses.SquareReadoutPulse(length=200, id="readout", amplitude=0.01)},
        opx_output=LFFEMAnalogOutputPort("con1", lf_fem, port_id=2, upsampling_mode="mw"),
        opx_input=LFFEMAnalogInputPort("con1", lf_fem, port_id=2),
    )

    xy_drives = {}
    for n, rf_freq in zip([1, 2, 3], [5.1e9, 5.2e9, 5.3e9]):
        xy_drives[n] = XYDriveMW(
            id=f"q{n}_xy",
            opx_output=f"{xy_path(n)}/opx_output",
            RF_frequency=rf_freq,
        )

    machine.create_virtual_gate_set(
        virtual_channel_mapping={f"dot{n}": gates[n] for n in plunger_wiring},
        gate_set_id="main_qpu",
    )
    machine.create_virtual_gate_set(
        virtual_channel_mapping={"source": source, "drain": drain, "g6": g6},
        gate_set_id="qdac_only_gateset",
    )

    machine.register_global_gates([source, drain])
    machine.register_quantum_dots([gates[n] for n in sorted(plunger_wiring)])

    gates[1].readout = resonator_1
    gates[4].readout = resonator_2

    for n in [1, 2, 3]:
        machine.register_qubit(f"dot{n}", f"q{n}", xy_drives[n])
        machine.qubits[f"q{n}"].xy.opx_output.upconverter_frequency = 5.0e9
        machine.qubits[f"q{n}"].xy.opx_output.band = 2

    for gate in list(gates.values()) + [drain]:
        gate.operations["trigger"] = pulses.SquarePulse(amplitude=0, length=1000, digital_marker="ON")
        gate.operations["half_max_square"] = pulses.SquarePulse(amplitude=0.25, length=16)

    wire_machine_macros(machine)
    return gates, source, drain, g6


########################################################################################################################
# %% 1. Static parameters — must match quam_demo_clean.py
########################################################################################################################
host_ip = "172.16.33.115"
cluster_name = "CS_3"
lf_fem = 5
mw_fem = 1
qdac_ip = "172.16.33.101"
qdac_name = "qdac_1"  # key in set_dac_config (dac_spec uses dac_name="qdac1")
save_path = "./c12_state_wirer"

qubit_drive_dots = [1, 2, 3]

plunger_wiring = {
    1: {"out_port": 1, "trigger": 1, "qdac_port": 1},
    2: {"out_port": 2, "trigger": 2, "qdac_port": 2},
    3: {"out_port": 3, "trigger": 1, "qdac_port": 3},
    4: {"out_port": 4, "trigger": 2, "qdac_port": 4},
    5: {"out_port": 5, "trigger": 2, "qdac_port": 5},
}

########################################################################################################################
# %% 2. Available instruments
########################################################################################################################
instruments = Instruments()
instruments.add_lf_fem(controller=1, slots=[lf_fem])
instruments.add_mw_fem(controller=1, slots=[mw_fem])

########################################################################################################################
# %% 3. Declare connectivity — logical elements and port constraints
########################################################################################################################
connectivity = Connectivity()

for dot_id, spec in plunger_wiring.items():
    connectivity.add_quantum_dot_voltage_gate_lines(
        dot_id,
        triggered=True,
        constraints=plunger_constraint(lf_fem, spec["out_port"], spec["trigger"]),
    )

connectivity.add_voltage_gate_lines(
    "drain",
    name="",
    triggered=True,
    constraints=plunger_constraint(lf_fem, out_port=8, trigger_port=1),
)

connectivity.add_quantum_dot_drive_lines(
    qubit_drive_dots,
    shared_line=True,
    use_mw_fem=True,
    constraints=mw_fem_spec(con=1, slot=mw_fem, out_port=1),
)

########################################################################################################################
# %% 4. Allocate channels on the declared instruments
########################################################################################################################
allocate_wiring(connectivity, instruments)

# Optional: visualize wiring (requires a GUI backend)
# visualize(connectivity.elements, instruments.available_channels)
# plt.show(block=False)

########################################################################################################################
# %% 5. Build wiring JSON
########################################################################################################################
machine = LossDiVincenzoQuam()

build_quam_wiring(
    connectivity,
    host_ip,
    cluster_name,
    machine,
    path=save_path,
)
add_ports(machine)

########################################################################################################################
# %% 6. Register components — same names as quam_demo_clean.py
########################################################################################################################
gates, source, drain, g6 = register_quam_from_wiring(machine, lf_fem, plunger_wiring)

########################################################################################################################
# %% 7. Post-build — experiment-specific configuration
########################################################################################################################
gate_set = machine.virtual_gate_sets["main_qpu"]
gate_set.add_layer(
    layer_id="detuning",
    source_gates=["e1", "mu1", "e2", "mu2", "t"],
    target_gates=["dot1", "dot2", "dot3", "dot4", "dot5"],
    matrix=[
        [1, -1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, -1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
    ],
)
print(gate_set.resolve_voltages({"e1": 0.05, "mu1": 0.2, "t": 0.1}))

machine.voltage_sequences["main_qpu"] = gate_set.new_sequence(track_integrated_voltage=True)
gate_set.add_point("load", {"dot1": 0.05, "dot2": 0.2}, 2000)
gate_set.add_point("operation", {"dot1": -0.05, "dot2": 0.0, "dot3": 0.05}, 100)
gate_set.add_point("readout", {"dot1": 0, "e1": 0.05, "mu1": 0.2, "t": 0.1}, 1500)

machine.quantum_dots["dot1"].add_point(
    "isolated_operation", {"dot1": 0.1, "dot2": -0.08, "dot3": -0.08}, 1000
)
machine.quantum_dots["dot2"].add_point(
    "isolated_operation", {"dot1": -0.08, "dot2": 0.1, "dot3": -0.08}, 1000
)

for point in gate_set.macros.values():
    print(point)

########################################################################################################################
# %% 8. QDAC connection
########################################################################################################################
qdac_connect = True

if qdac_connect:
    machine.set_dac_config({qdac_name: qdac_config(qdac_ip)})

    for dot_id, spec in plunger_wiring.items():
        machine.wire_voltage_gate_qdac(
            gates[dot_id],
            dac_name=qdac_name,
            qdac_output_port=spec["qdac_port"],
            with_trigger_channel=True,
            qdac_trigger_in=spec["trigger"],
        )

    for i, global_gate in enumerate([source, drain]):
        machine.wire_voltage_gate_qdac(
            global_gate,
            qdac_output_port=i + 20,
            dac_name=qdac_name,
        )

    machine.connect_to_external_source()
    machine.create_virtual_dc_set("main_qpu")

########################################################################################################################
# %% 9. Save
########################################################################################################################
machine.save(save_path)
print(f"QUAM state saved to {save_path}")

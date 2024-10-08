import json
from qm import SimulationConfig
from qm.qua import program
from quam import QuamDict
from quam.components.ports import MWFEMAnalogOutputPort, MWFEMAnalogInputPort
from quam.components.channels import InOutMWChannel, MWChannel
from quam.components.pulses import SquarePulse, SquareReadoutPulse
from quam_libs.components import QuAM

machine = QuAM()  # or, QuAM.load() if the state already exists

# vvv  delete these if using QuAM.load()
machine.network.host = "172.16.33.116"
machine.network.cluster_name = "Beta_8"
machine.wiring = QuamDict({})
# ^^^

mw_out = MWChannel(
    id="mw_out",
    operations={
        "cw": SquarePulse(amplitude=1, length=100),
        "readout": SquareReadoutPulse(amplitude=0.2, length=100),
    },
    opx_output=MWFEMAnalogOutputPort(
        controller_id="con1", fem_id=1, port_id=2, band=1, upconverter_frequency=int(3e9), full_scale_power_dbm=-14
    ),
    upconverter=1,
    intermediate_frequency=20e6,
)
mw_in = InOutMWChannel(
    id="mw_in",
    operations={
        "readout": SquareReadoutPulse(amplitude=0.1, length=100),
    },
    opx_output=MWFEMAnalogOutputPort(
        controller_id="con1", fem_id=1, port_id=1, band=1, upconverter_frequency=int(3e9), full_scale_power_dbm=-14
    ),
    opx_input=MWFEMAnalogInputPort(controller_id="con1", fem_id=1, port_id=1, band=1, downconverter_frequency=int(3e9)),
    upconverter=1,
    time_of_flight=28,
    intermediate_frequency=10e6,
)

machine.qubits["dummy_out"] = mw_out
machine.qubits["dummy_in"] = mw_in

with program() as prog:
    mw_out.play("cw")
    mw_in.align()
    mw_in.play("readout")

config = machine.generate_config()
qmm = machine.connect()

simulation_config = SimulationConfig(duration=250)  # In clock cycles = 4ns
job = qmm.simulate(config, prog, simulation_config)
job.get_simulated_samples().con1.plot()

# save machine into state.json
machine.save("dummy_state.json")

# %%
# View the corresponding "raw-QUA" config
with open("dummy_qua_config.json", "w+") as f:
    json.dump(machine.generate_config(), f, indent=4)

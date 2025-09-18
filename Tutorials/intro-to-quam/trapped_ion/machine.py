# %%
import numpy as np

from qm.qua import *
from quam.components.ports import (
    LFFEMAnalogInputPort,
    LFFEMAnalogOutputPort,
    MWFEMAnalogOutputPort,
    FEMDigitalOutputPort,
)
from quam.components.channels import (
    Channel,
    SingleChannel,
    DigitalOutputChannel,
    InOutSingleChannel,
    MWChannel,
)
from quam.components.pulses import (
    Pulse,
    SquareReadoutPulse,
    SquarePulse,
)

#############################################################
## Import custom components and macros
#############################################################
from trapped_ion.custom_components import (
    HyperfineQubit,
    GlobalOperations,
    Quam,
)
from trapped_ion.custom_macros import MeasureMacro, SingleXMacro, DoubleXMacro


#############################################################
## Generate QUAM object
#############################################################

machine = Quam()

n_qubits = 2
aom_position = np.linspace(200e6, 300e6, n_qubits)
mw_IF = 100e6
mw_LO = 3e9
mw_band = 1

# for each qubit
for i in range(n_qubits):
    qubit_id = f"q{i + 1}"
    qubit = HyperfineQubit(
        id=f"{qubit_id}",
        readout=InOutSingleChannel(
            opx_output=LFFEMAnalogOutputPort("con1", 1, 2),
            opx_input=LFFEMAnalogInputPort("con1", 1, 2),
            intermediate_frequency=aom_position[i],
        ),
        shelving=SingleChannel(
            opx_output=LFFEMAnalogOutputPort("con1", 1, 3),
            intermediate_frequency=aom_position[i],
        ),
    )

    # define pulse
    qubit.shelving.operations["const"] = SquarePulse(length=1_000, amplitude=0.1)
    qubit.readout.operations["const"] = SquareReadoutPulse(length=2_000, amplitude=0.1)

    # define macro
    qubit.macros["measure"] = MeasureMacro(threshold=10)

    # add to quam
    machine.qubits[qubit_id] = qubit

# set global properties
machine.global_op = GlobalOperations(
    global_mw=MWChannel(
        id="global_mw",
        opx_output=MWFEMAnalogOutputPort(
            "con1", 8, 1, band=mw_band, upconverter_frequency=mw_LO
        ),
        intermediate_frequency=mw_IF,
    ),
    ion_displacement=Channel(
        digital_outputs={
            "ttl": DigitalOutputChannel(
                opx_output=FEMDigitalOutputPort("con1", 8, 1), delay=136, buffer=0
            )
        },
    ),
)

# define pulse
machine.global_op.global_mw.operations["x180"] = SquarePulse(amplitude=0.2, length=1000)
machine.global_op.global_mw.operations["y180"] = SquarePulse(
    amplitude=0.2, length=1000, axis_angle=90
)
machine.global_op.ion_displacement.operations["ttl"] = Pulse(
    length=1000, digital_marker=[(1, 500), (0, 0)]
)

# operation macro
machine.global_op.macros["X"] = SingleXMacro()
machine.global_op.macros["N_XX"] = DoubleXMacro()

machine.print_summary()

machine.save("state_before.json")

# %%
from qm import generate_qua_script

n_avg = 10
optimize_qubit_idx = 2

with program() as prog:
    n = declare(int)
    state_st = declare_stream()
    qubit_idx = declare(int, optimize_qubit_idx)

    with for_(n, 0, n < n_avg, n + 1):
        machine.global_op.apply("X", qubit_idx=qubit_idx)
        for i, qubit in enumerate(machine.qubits.values()):
            state = qubit.apply("measure")
            save(state, state_st)
            align()
            wait(1_000)

    with stream_processing():
        state_st.buffer(n_qubits).average().save_all("state")


print(generate_qua_script(prog))

# %%
from qm import QuantumMachinesManager
from qm import SimulationConfig

qop_ip = "172.16.33.115"  # Write the QM router IP address
cluster_name = "CS_4"  # Write your cluster_name if version >= QOP220
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qua_config = machine.generate_config()

# Simulates the QUA program for the specified duration
simulation_config = SimulationConfig(duration=20_000)  # In clock cycles = 4ns
# Simulate blocks python until the simulation is done
job = qmm.simulate(qua_config, prog, simulation_config)
# Get the simulated samples
samples = job.get_simulated_samples()
# Plot the simulated samples
samples.con1.plot()

# %%
# Get the waveform report object
waveform_report = job.get_simulated_waveform_report()
# Cast the waveform report to a python dictionary
waveform_dict = waveform_report.to_dict()
# Visualize and save the waveform report
waveform_report.create_plot(samples, plot=True, save_path=None)

# %%
optimize_qubit_idx = 1
XX_rep = 1
n_avg = 10
amp_scan = np.linspace(0.5, 1.5, 10)

with program() as prog:
    n = declare(int)
    state_st = declare_stream()
    qubit_idx = declare(int, optimize_qubit_idx)
    amp_i = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        with for_each_(amp_i, amp_scan):
            machine.global_op.apply(
                "N_XX", qubit_idx=qubit_idx, amp_scale=amp_i, XX_rep=XX_rep
            )
            for i, qubit in enumerate(machine.qubits.values()):
                state = qubit.apply("measure")
                save(state, state_st)
                align()
                wait(1_000)

    with stream_processing():
        state_st.buffer(n_qubits).buffer(len(amp_scan)).average().save_all("state")

# %%
# Simulates the QUA program for the specified duration
simulation_config = SimulationConfig(duration=20_000)  # In clock cycles = 4ns
# Simulate blocks python until the simulation is done
job = qmm.simulate(qua_config, prog, simulation_config)
# Get the simulated samples
samples = job.get_simulated_samples()
# Plot the simulated samples
samples.con1.plot()

# %%
state = job.result_handles.get("state").fetch_all()["value"]
state_scan = state[0, :, optimize_qubit_idx - 1]
best_amp_scan = amp_scan[np.argmin(state_scan)]
original_amplitude = machine.global_op.global_mw.operations["x180"].amplitude
machine.global_op.global_mw.operations["x180"].amplitude *= best_amp_scan
machine.global_op.global_mw.operations["y180"].amplitude *= best_amp_scan

# %%
# Update the configuration
machine.save("state_after.json")

# Load the QUAM configuration
machine = Quam.load("state_after.json")

new_amplitude = machine.global_op.global_mw.operations["x180"].amplitude
print(f"{original_amplitude=}")
print(f"{new_amplitude=}")

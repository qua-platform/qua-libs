# %%
import numpy as np

from qm.qua import *

#############################################################
## Import custom components and macros
#############################################################
from trapped_ion.custom_components import (
    Quam,
)

# readily loads the configuration saved
machine = Quam.load("state_before.json")
n_qubits = len(machine.qubits)

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

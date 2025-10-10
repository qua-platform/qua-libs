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
optimize_qubit_idx = 1
XX_rep = 1
n_avg = 1
amp_scan = np.linspace(0.5, 1.5, 4)

with program() as prog:
    n = declare(int)
    state_st = declare_stream()
    qubit_idx = declare(int, optimize_qubit_idx)
    amp_i = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        with for_each_(amp_i, amp_scan):
            machine.global_op.apply("N_XX", qubit_idx=qubit_idx, amp_scale=amp_i, XX_rep=XX_rep)
            for i, qubit in enumerate(machine.qubits.values()):
                state = qubit.apply("measure")
                save(state, state_st)
                align()
                wait(1_000)

    with stream_processing():
        state_st.buffer(n_qubits).buffer(len(amp_scan)).average().save_all("state")

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
import matplotlib.pyplot as plt

state = job.result_handles.get("state").fetch_all()["value"]
state_scan = state[0, :, optimize_qubit_idx - 1]
best_amp_scan = amp_scan[np.argmin(state_scan)]
plt.plot(amp_scan, state_scan, "o-")
plt.axvline(best_amp_scan, color="red", linestyle="--")
plt.xlabel("Amplitude scale")
plt.ylabel("Average state")

# %%
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

# %%

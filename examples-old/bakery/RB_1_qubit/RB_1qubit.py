from qm import SimulationConfig
from qm.qua import *
from qm.QmJob import QmJob
from qm.QuantumMachinesManager import QuantumMachinesManager
from RB_1qb_configuration import config, pulse_len
from qualang_tools.bakery.randomized_benchmark import RBOneQubit
import numpy as np
import matplotlib.pyplot as plt

d_max = 100  # Maximum RB sequence length
K = 4  # Number of RB sequences

RB = RBOneQubit(config, d_max, K, "qe1")
RB_sequences = RB.sequences
RB_baked_sequences = RB.baked_sequences
duration_trackers = RB.duration_trackers
inverse_ops = RB.inverse_ops  # Array of inverse indices
played_Cliffords = [RB_sequences[k].operations_list for k in range(K)]  # List of random ops (strings)
played_inverse_Ops = [RB.sequences[k].inverse_op_string for k in range(K)]  # List of inverse sequences (strings)

with program() as RB_prog:
    truncate = declare(int)
    inverse_op = declare(int)

    I = declare(fixed)
    th = declare(fixed, value=0.0)
    state = declare(bool, value=False)

    out_str = declare_stream()

    for k in range(K):
        truncate_array = declare(int, value=[x * pulse_len // 4 for x in duration_trackers[k]])

        inverse_ops_QUA = declare(int, value=inverse_ops[k])
        with for_each_((truncate, inverse_op), (truncate_array, inverse_ops_QUA)):
            align("qe1", "rr")
            wait(30, "qe1")
            # Active reset
            assign(state, I > th)
            with if_(state):
                play("X", "qe1")
            RB_baked_sequences[k].run(trunc_array=[("qe1", truncate)])
            RB_sequences[k].play_revert_op2(inverse_op)

            align("qe1", "rr")
            # Measurement
            measure("readout", "rr", None, integration.full("integW1", I))

            save(state, out_str)
            save(inverse_op, "inv")
            save(truncate, "truncate")

    with stream_processing():
        out_str.boolean_to_int().buffer(K, d_max).average().save("out_stream")

qmm = QuantumMachinesManager()
qmm.close_all_quantum_machines()
qm = qmm.open_qm(config)
job: QmJob = qm.simulate(RB_prog, SimulationConfig(2000))
results = job.result_handles

inv = results.inv.fetch_all()["value"]
truncate = results.truncate.fetch_all()["value"]

# Plot simulated samples
samples = job.get_simulated_samples()
samples.con1.plot()
plt.show()

print("Inversion operations:", inv)
print("Truncations indices:", truncate)
print(played_Cliffords)
print(played_inverse_Ops)

# Plotting first baked RB sequence
baked_pulse_I = config["waveforms"]["qe1_baked_wf_I_0"]["samples"]
baked_pulse_Q = config["waveforms"]["qe1_baked_wf_Q_0"]["samples"]
plt.figure()
t = np.arange(0, len(baked_pulse_I), 1)
plt.plot(t, baked_pulse_I)
plt.plot(t, baked_pulse_Q)
plt.show()

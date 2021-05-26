from qm import SimulationConfig
from qm.qua import *
from qm.QmJob import QmJob
from qm.QuantumMachinesManager import QuantumMachinesManager
from RB_1qb_configuration import config, pulse_len, np
from RB_utils import c1_table, c1_ops, RB_one_qubit
import matplotlib.pyplot as plt


d_max = 20  # Maximum RB sequence length
K = 1  # Number of RB sequences

RB = RB_one_qubit(config, d_max, K, "qe1", c1_ops, c1_table)
duration_tracker = RB.duration_tracker
inverse_ops = RB.revert_ops
RB_seq = RB.sequences


with program() as RB_prog:
    truncate = declare(int)
    truncate2 = declare(int)
    truncate_array = declare(int, value=duration_tracker)

    inverse_ops_QUA = declare(int, value=inverse_ops)
    inverse_op = declare(int)

    I = declare(fixed)
    th = declare(fixed, value=0.)
    state = declare(bool)

    out_str = declare_stream()

    for k in range(K):
        with for_each_((truncate, inverse_op), (truncate_array, inverse_ops_QUA)):

            assign(truncate2, truncate * pulse_len)

            align("qe1", "rr")
            play(RB_seq[k].operations["qe1"], 'qe1', truncate=truncate2)  # Truncate for RB seq of smaller lengths
            RB.play_revert_op(inverse_op)

            align("qe1", "rr")
            # Measurement
            measure("readout", "rr", None, integration.full("integW1", I))
            # Active reset
            with if_(state):
                play("X", "qe1")

            assign(state, I > th)

            save(state, out_str)
            save(inverse_op, 'inv')
            save(truncate, "truncate")

    with stream_processing():
        out_str.boolean_to_int().buffer(K, d_max).average().save("out_stream")


qmm = QuantumMachinesManager()
job: QmJob = qmm.simulate(config, RB_prog, SimulationConfig(20000))
results = job.result_handles

inv = results.inv.fetch_all()["value"]
truncate = results.truncate.fetch_all()["value"]

# Plot simulated samples
samples = job.get_simulated_samples()
samples.con1.plot()

print('Inversion operations:', inv)
print('Truncations indices:', truncate)

# Plotting baked RB sequence
baked_pulse_I = config["waveforms"]["qe1_baked_wf_I_0"]["samples"]
baked_pulse_Q = config["waveforms"]["qe1_baked_wf_Q_0"]["samples"]
t = np.arange(0, len(baked_pulse_I), 1)
plt.plot(t, baked_pulse_I)
plt.plot(t, baked_pulse_Q)

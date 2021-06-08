from qm import SimulationConfig
from qm.qua import *
from qm.QmJob import QmJob
from qm.QuantumMachinesManager import QuantumMachinesManager
from rb_1qb_configuration import config, pulse_len
from rb_utils import RBOneQubit
import matplotlib.pyplot as plt


d_max = 300  # Maximum RB sequence length
K = 1  # Number of RB sequences

RB = RBOneQubit(config, d_max, K, "qe1")
RB_sequences = RB.sequences
RB_baked_sequences = RB.baked_sequences
duration_trackers = RB.duration_trackers
inverse_ops = RB.inverse_ops


with program() as RB_prog:
    truncate = declare(int)
    inverse_op = declare(int)

    I = declare(fixed)
    th = declare(fixed, value=0.)
    state = declare(bool)

    out_str = declare_stream()

    for k in range(K):
        truncate_array = declare(int, value=[x * pulse_len // 4 for x in duration_trackers[k]])

        inverse_ops_QUA = declare(int, value=inverse_ops[k])
        with for_each_((truncate, inverse_op), (truncate_array, inverse_ops_QUA)):

            align("qe1", "rr")
            wait(30, "qe1")
            play(RB_baked_sequences[k].operations["qe1"], 'qe1', truncate=truncate)  # Truncate for RB seq of smaller lengths
            RB_sequences[k].play_revert_op2(inverse_op)

            align("qe1", "rr")
            # Measurement
            measure("readout", "rr", None, integration.full("integW1", I))
            # Active reset
            with if_(state):
                play("X", "qe1")

            assign(state, I > th)

            save(state, out_str)
            save(inverse_op, 'inv')
            save(truncate, 'truncate')

    with stream_processing():
        out_str.boolean_to_int().buffer(K, d_max).average().save('out_stream')


qmm = QuantumMachinesManager()
job: QmJob = qmm.simulate(config, RB_prog, SimulationConfig(20000))
results = job.result_handles

inv = results.inv.fetch_all()["value"]
truncate = results.truncate.fetch_all()["value"]

# # Plot simulated samples
# samples = job.get_simulated_samples()
# samples.con1.plot()
#
# print('Inversion operations:', inv)
# print('Truncations indices:', truncate)
#
# # Plotting baked RB sequence
# baked_pulse_I = config["waveforms"]["qe1_baked_wf_I_0"]["samples"]
# baked_pulse_Q = config["waveforms"]["qe1_baked_wf_Q_0"]["samples"]
# t = np.arange(0, len(baked_pulse_I), 1)
# plt.plot(t, baked_pulse_I)
# plt.plot(t, baked_pulse_Q)

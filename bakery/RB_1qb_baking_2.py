from qm import SimulationConfig
from qm.QmJob import QmJob
from qm.QuantumMachinesManager import QuantumMachinesManager
from RB_1qb_configuration import config, pulse_len
from rb_1_with_table_guide import *
import matplotlib.pyplot as plt

d_max = 20  # Maximum RB sequence length
K = 1  # Number of RB sequences

state_tracker = [int] * d_max  # Keeps track of all transformations done on qubit state
state_init = 0
revert_op = [int] * d_max  # Keeps track of inverse op index associated to each sequence
duration_tracker = [0] * d_max  # Keeps track of each Clifford's duration
K_list = [Baking] * K

for k in range(K):  # Generate K RB sequences of length d_max
    with baking(config) as b:
        for d in range(d_max):
            i = np.random.randint(0, len(c1_ops))
            duration_tracker[d] = d + 1  # Set the duration to the value of the sequence step

            # Play the random Clifford
            random_clifford = c1_ops[i]
            for op in random_clifford:
                b.play(op, "qe1")
                duration_tracker[d] += 1  # Add additional duration for each pulse played to build Clifford

            if d == 0:  # Handle the case for qubit set to original/ground state
                state_tracker[d] = c1_table[state_init][i]
            else:  # Get the newly transformed state within th Cayley table based on previous step
                state_tracker[d] = c1_table[state_tracker[d-1]][i]
            revert_op[d] = find_revert_op(state_tracker[d])
    K_list[k] = b  # Stores all the RB sequences


#  Store here all Cliffords gates through baked waveforms
baked_cliffords = []
for i in range(len(c1_ops)):
    with baking(config) as b2:
        for op in c1_ops[i]:
            b2.play(op, "qe1")
    baked_cliffords.append(b2)


with program() as RB:
    truncate = declare(int)
    truncate2 = declare(int)
    truncate_array = declare(int, value=duration_tracker)

    revert_op_QUA = declare(int, value=revert_op)
    inverse_op = declare(int)

    I = declare(fixed)
    state = declare(bool)
    out_str = declare_stream()
    for k in range(K):
        with for_each_((truncate, inverse_op), (truncate_array, revert_op_QUA)):
            assign(truncate2, truncate * pulse_len)
            play(K_list[k].operations["qe1"], 'qe1', truncate=truncate2)  # Truncate for RB seq of smaller lengths
            play_revert_op(inverse_op, baked_cliffords)
            save(inverse_op, 'inv')
            save(truncate, "truncate")
            measure_state(state, I)
            save(state, out_str)
            active_reset(state)

    with stream_processing():
        out_str.boolean_to_int().buffer(K, d_max).average().save("out_stream")


qmm = QuantumMachinesManager()
job: QmJob = qmm.simulate(config, RB, SimulationConfig(12000))
results = job.result_handles
inv = results.inv.fetch_all()["value"]
truncate = results.truncate.fetch_all()["value"]
samples = job.get_simulated_samples()
samples.con1.plot()

# print(inv)
# print(truncate)

# Plotting baked sequence
# baked_pulse_I = config["waveforms"]["qe1_baked_wf_I_0"]["samples"]
# baked_pulse_Q = config["waveforms"]["qe1_baked_wf_Q_0"]["samples"]
# t = np.arange(0, len(baked_pulse_I), 1)
# plt.plot(t, baked_pulse_I)
# plt.plot(t, baked_pulse_Q)

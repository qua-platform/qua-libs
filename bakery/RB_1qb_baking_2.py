from qm import SimulationConfig
from qm.QmJob import QmJob
from qm.QuantumMachinesManager import QuantumMachinesManager
from RB_1qb_configuration import config, pulse_len
from rb_1_with_table_guide import *

d_max = 10
state_tracker = [int] * d_max  # Keeps track of all transformations done on qubit state
state_init = 0
revert_op = [int] * d_max  # Keeps track of the inverse op associated to each sequence
duration_tracker = [0] * d_max  # Keeps track of the duration of each Clifford
with baking(config) as b:
    for d in range(d_max):
        i = np.random.randint(0, len(c1_ops))
        duration_tracker[d] = d  # Set the duration to the value of the sequence step
        for op in c1_ops[i]:  # Check the case op is I
            b.play(op, "qe1")
            duration_tracker[d] += 1  # Add additional duration for each pulse played to build Clifford

        if d == 0:  # Handle the case for qubit set to original/ground state
            state_tracker[d] = c1_table[state_init][i]
        else:  # Get the newly transformed state within th Cayley table based on previous step
            state_tracker[d] = c1_table[d-1][i]
        revert_op[d] = find_revert_op(state_tracker[d])

#  Store here all the Cliffords gates through different baked waveforms to call them
#  easily within QUA
baked_cliffords = []
for i in range(len(c1_ops)):
    with baking(config) as b2:
        for op in c1_ops[i]:
            b2.play(op, "qe1")
    baked_cliffords.append(b2)


with program() as RB:
    truncate = declare(int)
    truncate2 = declare(int)
    truncate_array = declare(int, value= duration_tracker)

    revert_op_QUA = declare(int, value=revert_op)
    inverse_op = declare(int)

    I = declare(fixed)
    state = declare(bool)
    out_str = declare_stream()

    with for_each_((truncate,inverse_op), (truncate_array, revert_op_QUA)):
        assign(truncate2, truncate * pulse_len)
        play(b.operations["qe1"], 'qe1', truncate=truncate2)
        play_revert_op(inverse_op, baked_cliffords)

        measure_state(state, I)
        save(state, out_str)
        active_reset(state)


qmm = QuantumMachinesManager()
job: QmJob = qmm.simulate(config, RB, SimulationConfig(1500))





from qm import SimulationConfig
from qm.QmJob import QmJob
from qm.qua import *
import numpy as np
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
from RB_1qb_configuration import *
from rb_1_with_table_guide import *

d_max = 10
state_tracker = [int] * d_max
state_init = 0
revert_op = [int] * d_max
with baking(config) as b:
    for d in range(d_max):
        i = np.random.randint(0, len(c1_ops))
        for op in c1_ops[i]: # Check the case op is I
            b.play(op, "qe1")
        if d == 0:
            state_tracker[d]= c1_table[state_init][i]
        else:
            state_tracker[d] = c1_table[d-1][i]
        revert_op[d] = find_revert_op(state_tracker[d])

baked_cliffords = []
for i in range(24):
    with baking(config) as b2:
        for op in c1_ops[i]:
            b2.play(op, "qe1")
    baked_cliffords.append(b2)
    
def play_revert_op(index:int, baked_cliffords: list[Baking]):
    with switch_(index):
        for i in range (len(baked_cliffords)):
            with case_(i):
                baked_cliffords[i].run()

with program() as RB:
    truncate = declare(int)
    I = declare(fixed)
    state = declare(bool)
    out_str = declare_stream()
    revert_op_QUA = declare(int, value=revert_op)
    truncate2 = declare(int)
    with for_(truncate, 0, truncate < d_max, truncate + 1):
        assign(truncate2, truncate * pulse_len)
        play(b.operations["qe1"], 'qe1', truncate=truncate2)
        play_revert_op(revert_op_QUA[truncate])

        measure_state(state, I)
        save(state, out_str)
        active_reset(state)


qmm = QuantumMachinesManager()
job: QmJob = qmm.simulate(config,
                          RB,
                          SimulationConfig(1500))





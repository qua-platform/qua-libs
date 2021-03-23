from random import randint
from bakary import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
from RB_1qb_configuration import *

N_avg = 1
circuit_depth_vec = list(range(1, 10, 2))
# circuit_depth_vec=list(set(np.logspace(0,2,10).astype(int).tolist()))
t1 = 10
b_list = []

for depth in circuit_depth_vec:
    with baking(config, padding_method="right") as b:
        generate_cliffords(b, "qe1", pulse_length=16)
        final_state = randomize_and_play_circuit(depth, b)
        play_clifford(recovery_clifford(final_state), final_state, b)

    b_list.append(b)

# Open communication with the server.
QMm = QuantumMachinesManager("3.122.60.129")
QM1 = QMm.open_qm(config)


with program() as RBprog:
    N = declare(int)
    I = declare(fixed)
    state = declare(bool)
    out_str = declare_stream()

    with for_(N, 0, N < N_avg, N + 1):
        for i in range(len(circuit_depth_vec)):
            b_list[i].run()
            align("rr", "qe1")
            measure_state(state, I)
            save(state, out_str)
            active_reset(state)

    with stream_processing():
        out_str.boolean_to_int().buffer(len(circuit_depth_vec)).average().save(
            "out_stream"
        )

job = QM1.simulate(RBprog, SimulationConfig(int(100000)))
res = job.result_handles
avg_state = res.out_stream.fetch_all()

samples = job.get_simulated_samples()

samples.con1.plot()

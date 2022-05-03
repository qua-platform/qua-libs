from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import config
from rb_lib import *

# Open communication with the server.
QMm = QuantumMachinesManager()

QM1 = QMm.open_qm(config)

N_avg = 10
circuit_depth_vec = list(range(1, 10, 1))
t1 = 10  # in nS, allowing for return to ground state
interleave_clifford = ["X", "Y/2", "X/2"]

with program() as RBprog:
    N = declare(int)
    I = declare(fixed)
    state = declare(bool)
    out_str = declare_stream()

    with for_(N, 0, N < N_avg, N + 1):
        for depth in circuit_depth_vec:
            final_state = randomize_interleaved_circuit(interleave_clifford, depth)
            play_clifford(recovery_clifford(final_state), final_state)
            align("rr", "qe1")
            measure_state(state, I)
            save(state, out_str)
            wait(10 * t1, "qe1")

    with stream_processing():
        out_str.boolean_to_int().buffer(len(circuit_depth_vec)).average().save("out_stream")

job = QM1.simulate(RBprog, SimulationConfig(int(10000)))
res = job.result_handles
avg_state = res.out_stream.fetch_all()

samples = job.get_simulated_samples()

samples.con1.plot()

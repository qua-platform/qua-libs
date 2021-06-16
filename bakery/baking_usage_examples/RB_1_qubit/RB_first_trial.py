from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from RB_1qb_configuration import *

N_avg = 1
circuit_depth_vec = list(range(1, 10, 2))
t1 = 10
b_list = []

# Prepare baked waveforms, one baking for each circuit depth
for depth in circuit_depth_vec:
    with baking(config, padding_method="right") as b:
        generate_cliffords(b, "qe1", pulse_length=10)
        final_state = randomize_and_play_circuit(depth, b)
        play_clifford(recovery_clifford(final_state), final_state, b)

    b_list.append(b)

# Open communication with the server.
QMm = QuantumMachinesManager()
QM1 = QMm.open_qm(config, close_other_machines=True)

# QUA Program for 1 qubit RB:
with program() as RBprog:
    N = declare(int)
    I = declare(fixed)
    state = declare(bool)
    out_str = declare_stream()

    with for_(N, 0, N < N_avg, N + 1):
        for i in range(len(circuit_depth_vec)):
            align("rr", "qe1")
            b_list[i].run()
            align("rr", "qe1")
            measure_state(state, I)
            save(state, out_str)
            active_reset(state)

    with stream_processing():
        out_str.boolean_to_int().buffer(len(circuit_depth_vec)).average().save(
            "out_stream"
        )

job = QM1.simulate(RBprog, SimulationConfig(int(1000)))
res = job.result_handles
avg_state = res.out_stream.fetch_all()

samples = job.get_simulated_samples()

samples.con1.plot()

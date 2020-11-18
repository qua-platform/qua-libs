from random import randint

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
from configuration import config

# Open communication with the server.
QMm = QuantumMachinesManager()

cliffords = [['I'], ['X'], ['Y'], ['Y', 'X'],
             ['X/2', 'Y/2'], ['X/2', '-Y/2'], ['-X/2', 'Y/2'], ['-X/2', '-Y/2'], ['Y/2', 'X/2'], ['Y/2', '-X/2'],
             ['-Y/2', 'X/2'], ['-Y/2', '-X/2'],
             ['X/2'], ['-X/2'], ['Y/2'], ['-Y/2'], ['-X/2', 'Y/2', 'X/2'], ['-X/2', '-Y/2', 'X/2'],
             ['X', 'Y/2'], ['X', '-Y/2'], ['Y', 'X/2'], ['Y', '-X/2'], ['X/2', 'Y/2', 'X/2'], ['-X/2', 'Y/2', '-X/2']]


def recovery_clifford(state):
    # operations = {'x': ['I'], '-x': ['Y'], 'y': ['X/2', '-Y/2'], '-y': ['-X/2', '-Y/2'], 'z': ['-Y/2'], '-z': ['Y/2']}
    operations = {'z': ['I'], '-x': ['Y/2'], 'y': ['X/2'], '-y': ['-X/2'], 'x': ['-Y/2'], '-z': ['X']}
    return operations[state]


def transform_state(input_state: str, transformation: str):
    transformations = {'x': {'I': 'x', 'X/2': 'x', 'X': 'x', '-X/2': 'x', 'Y/2': 'z', 'Y': '-x', '-Y/2': '-z'},
                       '-x': {'I': '-x', 'X/2': '-x', 'X': '-x', '-X/2': '-x', 'Y/2': '-z', 'Y': 'x', '-Y/2': 'z'},
                       'y': {'I': 'y', 'X/2': 'z', 'X': '-y', '-X/2': '-z', 'Y/2': 'y', 'Y': 'y', '-Y/2': 'y'},
                       '-y': {'I': '-y', 'X/2': '-z', 'X': 'y', '-X/2': 'z', 'Y/2': '-y', 'Y': '-y', '-Y/2': '-y'},
                       'z': {'I': 'z', 'X/2': '-y', 'X': '-z', '-X/2': 'y', 'Y/2': '-x', 'Y': '-z', '-Y/2': 'x'},
                       '-z': {'I': '-z', 'X/2': 'y', 'X': 'z', '-X/2': '-y', 'Y/2': 'x', 'Y': 'z', '-Y/2': '-x'}}

    return transformations[input_state][transformation]


def play_clifford(clifford, state: str):
    for op in clifford:
        state = transform_state(state, op)
        if op != 'I':
            play(op, 'qe1')
    return state


def randomize_and_play_circuit(n_gates: int, init_state: str = 'x'):
    state = init_state
    for ind in range(n_gates):
        state = play_clifford(cliffords[np.random.randint(0, len(cliffords))], state)
    return state


QM1 = QMm.open_qm(config)

N_avg = 10
circuit_depth_vec = list(range(1, 10, 1))
# circuit_depth_vec=list(set(np.logspace(0,2,10).astype(int).tolist()))

t1 = 10

with program() as RBprog:
    N = declare(int)
    I = declare(fixed)
    s1 = declare(int, val=1)
    s0 = declare(int, val=0)
    th = declare(fixed, val=0)
    state = declare(bool)
    out_str = declare_stream()

    with for_(N, 0, N < N_avg, N + 1):
        for depth in circuit_depth_vec:
            final_state = randomize_and_play_circuit(depth)
            play_clifford(recovery_clifford(final_state), final_state)
            align('rr', 'qe1')
            measure('readout', 'rr', None, integration.full('integW1', I))
            # assign(state,I>th)
            # save(state, out_str)
            with if_(I > th):
                save(s1, out_str)
            with else_():
                save(s0, out_str)
            wait(10 * t1, 'qe1')

    with stream_processing():
        out_str.buffer(len(circuit_depth_vec)).average().save('out_stream')

job = QM1.simulate(RBprog,
                   SimulationConfig(int(100000)))
res=job.result_handles
avg_state=res.out_stream.fetch_all()

samples = job.get_simulated_samples()

samples.con1.plot()

"""vqgo.py: Variational quantum gate optimization on QUA
Author: Arthur Strauss - Quantum Machines
Created: 01/02/2021
QUA version : 0.8.439
"""

from configuration import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import random as rand
import numpy as np
from scipy.optimize import minimize, differential_evolution, brute
import matplotlib.pyplot as plt

n = 2  # number of qubits
d = 1  # depth of the quantum circuit
N_shots = 10  # Define number of shots necessary to compute exp
optimizer = "COBYLA"
qm1 = QuantumMachinesManager()
QM = qm1.open_qm(config)

target_gate = np.array([[1, 0, 0, 0],   # CNOT gate
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]
                        ])
states, readout_ops = generate_random_set(target_gate)
# QUA program

with program() as VQGO:
    rep = declare(int)  # Iterator for sampling
    k = declare(int)
    timing = declare(int, value=0)

    I = declare(fixed)
    Q = declare(fixed)
    state_estimate = declare(int)

    state_streams, I_streams, Q_streams = [declare_stream()] * n, \
                                          [declare_stream()] * n, \
                                          [declare_stream()] * n
    state_strings, I_strings, \
    Q_strings, q_strings, RR_strings = ["state" + str(i) for i in range(n)], \
                                       ["I" + str(i) for i in range(n)], \
                                       ["Q" + str(i) for i in range(n)], \
                                       ["q" + str(i) for i in range(n)], \
                                       ["RR" + str(i) for i in range(n)]

    rotation_angles = [[declare(fixed, size=3)] * n] * (d + 1)
    source_params = [[declare(fixed, size=n - 1)] * (d + 1)]

    with infinite_loop_():
        pause()

        # Retrieve classical parameters for VQO
        for i in range(d + 1):
            for j in range(n):
                with for_(k, init=0, cond=i < 3, update=i + 1):
                    pause()
                    assign(rotation_angles[i][j][k], IO1)
            with for_(k, init=0, cond=i < n - 1, update=i + 1):
                pause()
                assign(source_params[i][k], IO2)

        # Generate random selection of input states & tomography operations
        # for direct fidelity estimation

        with for_(rep, init=0, cond=rep <= N_shots, update=rep + 1):  # Do N_shots times the same quantum circuit
            # Run the computation for each input state and tomography operator the parametrized process
            for state in states.keys():
                for op in readout_ops.keys():
                    # Prepare input state
                    prepare_state(state)

                    # Apply parametrized U
                    for i in range(d + 1):
                        for j in range(n):
                            Rx(rotation_angles[i][j][0], q_strings[j])
                            Ry(rotation_angles[i][j][1], q_strings[j])
                            Rx(rotation_angles[i][j][2], q_strings[j])
                        if i != d:
                            play_source_gate(U_source[i], source_params[i])

                    # Measurement and state determination

                    # Apply change of basis to retrieve Pauli expectation
                    change_basis(op)
                    for k in range(n):  # for each RR
                        measurement(RR_strings[k], I, Q)
                        state_saving(I, Q, state_estimate, state_streams[k])
                        # Active reset, flip the qubit if measured state is 1
                        with if_(state_estimate == 1):
                            Rx(π, q_strings[k])
                        # Save I & Q variables, if deemed necessary by the user
                        raw_saving(I, Q, I_streams[k], Q_streams[k])

                assign(timing, timing + 1)
                save(timing, "timing")

    with stream_processing():
        for i in range(n):
            state_streams[i].boolean_to_int().buffer(len(states.keys()), len(readout_ops.keys())).save_all(
                state_strings[i])
            I_streams[i].buffer(len(states.keys()), len(readout_ops.keys())).save_all(I_strings[i])
            Q_streams[i].buffer(len(states.keys()), len(readout_ops.keys())).save_all(Q_strings[i])

job = QM.execute(VQGO)


def encode_params_in_IO(rotation_angles, source_params):
    # Insert angles values using IOs by keeping track
    # of where we are in the QUA program
    for angle in rotation_angles:
        while not (job.is_paused()):
            time.sleep(0.0001)
        QM.set_io1_value(angle)
        job.resume()
    for param in source_params:
        while not (job.is_paused()):
            time.sleep(0.0001)
        QM.set_io2_value(param)
        job.resume()


def AGI(params):  # Calculate cost function

    job.resume()
    rotation_angles = params[0:3 * n * (d + 1)]
    source_params = params[3 * n * (d + 1):]

    encode_params_in_IO(rotation_angles, source_params)

    # Implement routine here to set IO values and resume the program, and adapt prior part (or remove it)

    while not (job.is_paused()):
        time.sleep(0.0001)
    results = job.result_handles
    # time.sleep()
    output_states = []
    timing = results.timing.fetch_all()["value"]
    print(timing[-1])
    for d in range(n):
        output_states.append(results.get("state" + str(d)).fetch_all()["value"])

    counts = {}  # Dictionary containing statistics of measurement of each bitstring obtained
    expectation_values = {}

    for i in range(N_shots):
        bitstring = ""
        for j, st in list(enumerate(states.keys())):
            counts[st] = {}
            expectation_values[st] = {}
            for k, ope in list(enumerate(readout_ops.keys())):
                counts[st][ope] = {}
                expectation_values[st][ope] = {}
                for l in range(len(output_states)):
                    bitstring += str(output_states[l][i][j][k])
                    if bitstring in counts[st][ope]:
                        counts[st][ope][bitstring] = 0
                        expectation_values[st][ope][bitstring] = 0
                    else:
                        counts[st][ope][bitstring] += 1
                        if tomography_set[ope]["Ref"] == "ID__σ_z":
                            if bitstring == "00" or bitstring == "10":
                                expectation_values[st][ope][bitstring] += 1/N_shots
                            elif bitstring == "01" or bitstring == "11":
                                expectation_values[st][ope][bitstring] -= 1 / N_shots
                        if tomography_set[ope]["Ref"] == "σ_z__σ_z":
                            if bitstring == "00" or bitstring == "11":
                                expectation_values[st][ope][bitstring] += 1/N_shots
                            elif bitstring == "10" or bitstring == "01":
                                expectation_values[st][ope][bitstring] -= 1 / N_shots
                        if tomography_set[ope]["Ref"] == "σ_z__ID":
                            if bitstring == "00" or bitstring == "01":
                                expectation_values[st][ope][bitstring] += 1/N_shots
                            elif bitstring == "10" or bitstring == "11":
                                expectation_values[st][ope][bitstring] -= 1 / N_shots

    cost = 0
    return cost


init_rotation_angles = np.random.uniform(0, 2*π, 3*n*(d+1))
init_source_params = np.random.uniform(0., 2., d)
init_angles = list(init_rotation_angles)
for param in init_source_params:
    init_angles.append(param)

Result = minimize(AGI, init_angles, method=optimizer)
print(Result)

job.halt()

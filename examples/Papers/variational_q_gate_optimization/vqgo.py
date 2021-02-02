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
d = 1  # Define the depth of the quantum circuit, aka number of adiabatic evolution blocks
N_shots = 10  # Define number of shots necessary to compute exp
# QM configuration based on IBM Quantum Computer :
# Yorktown device https://github.com/Qiskit/ibmq-device-information/tree/master/backends/yorktown/V1
qm1 = QuantumMachinesManager()
QM = qm1.open_qm(config)

# QUA program

with program() as VQGO:
    rep = declare(int)  # Iterator for sampling (necessary to have good statistics for expectation value estimation)
    t = declare(int, value=10)  # Assume knowledge of relaxation time
    timing = declare(int, value=0)

    I = declare(fixed)
    Q = declare(fixed)
    state_estimate = declare(int)

    state_streams, I_streams, Q_streams = [None] * n, [None] * n, [None] * n
    state_strings, I_strings, Q_strings = [None] * n, [None] * n, [None] * n
    for i in range(n):
        state_strings[i] = "state" + str(i)
        I_strings[i] = "I" + str(i)
        Q_strings[i] = "Q" + str(i)
        state_streams[i], I_streams[i], Q_streams[i] = (
            declare_stream(),
            declare_stream(),
            declare_stream())

    with infinite_loop_():
        pause()
        γ = declare(fixed)
        β = declare(fixed)
        block = declare(int)
        γ_set = declare(fixed, size=d)
        β_set = declare(fixed, size=d)

        with for_(block, init=0, cond=block < d, update=i + 1):
            pause()
            assign(γ_set[block], IO1)
            assign(β_set[block], IO2)

        with for_(
                rep, init=0, cond=rep <= N_shots, update=rep + 1
        ):  # Do Nshots times the same quantum circuit

            # Measurement and state determination

            for k in range(n):  # Iterate over each resonator 'CPW' in the config file
                RR = "CPW" + str(k)
                q = "qubit" + str(k)
                measurement(RR, I, Q)

                state_saving(
                    I, Q, state_estimate, state_streams[k]
                )  # Do state discrimination based on each IQ point and save it in a variable called state0,1,..
                # Active reset, flip the qubit is measured state is 1
                with if_(state_estimate == 1):
                    Rx(π, q)

                raw_saving(
                    I, Q, I_streams[k], Q_streams[k]
                )  # Save I & Q variables, if deemed necessary by the user

        assign(timing, timing + 1)
        save(timing, "timing")
    with stream_processing():
        for i in range(n):
            state_streams[i].buffer(N_shots).save(state_strings[i])
            I_streams[i].buffer(N_shots).save(I_strings[i])
            Q_streams[i].buffer(N_shots).save(Q_strings[i])

job = QM.execute(VQGO)

# Try out the QAOA for the problem defined by the Maxcut for graph G

job.halt()

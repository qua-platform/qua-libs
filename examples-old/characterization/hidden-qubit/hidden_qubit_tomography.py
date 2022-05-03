"""hidden_qubit_tomography.py: Performing process tomography with one control and one hidden qubit
Author: Arthur Strauss - Quantum Machines
Created: 08/01/2021
QUA version used : 0.8.439
"""

# QM imports
from configuration import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import LoopbackInterface
from qm import SimulationConfig
from qm.qua import *

qm1 = QuantumMachinesManager()
QM = qm1.open_qm(config)

N_shots = 1

nb_processes = len(list(processes.keys()))
nb_basis_states = len(list(state_prep.keys()))
nb_Pauli_op = len(list(tomography_set.keys()))

with program() as hidden_qubit_tomography:
    N = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    state_c = declare(bool)
    state_h = declare(bool)
    stream_c = declare_stream()
    stream_h = declare_stream()
    with for_(var=N, init=0, cond=N < N_shots, update=N + 1):
        for process in processes.keys():  # 4
            for input_state in state_prep.keys():  # 16
                for readout_operator in tomography_set.keys():  # 15
                    align("RR", "control", "TC")
                    for pulse in state_prep[input_state]:
                        play_pulse(pulse)
                    for pulse in processes[process]:
                        play_pulse(pulse)

                    for op in tomography_set[readout_operator]:
                        play_pulse(op)

                    align("RR", "control", "TC")
                    measure_and_reset_state("RR", I, Q, state_c, state_h, stream_c, stream_h)

    with stream_processing():
        stream_c.boolean_to_int().buffer(N_shots, nb_processes, nb_basis_states, nb_Pauli_op).save_all("state_c")
        stream_h.boolean_to_int().buffer(N_shots, nb_processes, nb_basis_states, nb_Pauli_op).save_all("state_h")

job = qm1.simulate(
    config,
    hidden_qubit_tomography,
    SimulationConfig(int(50000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)  # Use LoopbackInterface to simulate the response of the qubit

results = job.result_handles

control = results.state_c.fetch_all()["value"]
hidden = results.state_h.fetch_all()["value"]

counts = {}
frequencies = {}
expectation_values = {}
ρ = {}
for i, process in list(enumerate(processes.keys())):  # 4 #Initialize all the dictionaries
    counts[process] = {}
    frequencies[process] = {}
    expectation_values[process] = {}
    ρ[process] = {}
    for j, input_state in list(enumerate(state_prep.keys())):
        counts[process][input_state] = {}  # 16
        frequencies[process][input_state] = {}
        expectation_values[process][input_state] = {}
        ρ[process][input_state] = {}
        for k, readout_operator in list(enumerate(tomography_set.keys())):
            state = str(control[i][j][k]) + str(hidden[i][j][k])
            if state in counts[process][input_state][readout_operator]:
                counts[process][input_state][readout_operator] = {
                    "00": 0,
                    "01": 0,
                    "10": 0,
                    "11": 0,
                }
                frequencies[process][input_state][readout_operator] = {
                    "00": 0,
                    "01": 0,
                    "10": 0,
                    "11": 0,
                }
                expectation_values[process][input_state][readout_operator] = 0
            else:
                counts[process][input_state][readout_operator][state] += 1
                # frequencies[process][input_state][readout_operator][state] += 1. / N_shots
                if (state == "00") or (state == "01"):
                    expectation_values[process][input_state][readout_operator] += 1.0 / N_shots
                else:
                    expectation_values[process][input_state][readout_operator] -= 1.0 / N_shots


ID = np.array([[1, 0], [0, 1]])
σ_x = np.array([[0, 1], [1, 0]])
σ_y = np.array([[0, -1j], [1j, 0]])
σ_z = np.array([[1, 0], [0, -1]])
Pauli_matrices = [ID, σ_x, σ_y, σ_z]

Pauli_basis = {
    # "ID__ID": np.kron(ID, ID),
    "ID__σ_x": np.kron(ID, σ_x),
    "ID__σ_y": np.kron(ID, σ_y),
    "ID__σ_z": np.kron(ID, σ_z),
    "σ_z__ID": np.kron(σ_z, ID),
    "σ_z__σ_x": np.kron(σ_z, σ_x),
    "σ_z__σ_y": np.kron(σ_z, σ_y),
    "σ_z__σ_z": np.kron(σ_z, σ_z),
    "σ_y__ID": np.kron(σ_y, ID),
    "σ_y__σ_x": np.kron(σ_y, σ_x),
    "σ_y__σ_y": np.kron(σ_y, σ_y),
    "σ_y__σ_z": np.kron(σ_y, σ_z),
    "σ_x__ID": np.kron(σ_x, ID),
    "σ_x__σ_x": np.kron(σ_x, σ_x),
    "σ_x__σ_y": np.kron(σ_x, σ_y),
    "σ_x__σ_z": np.kron(σ_x, σ_z),
}

# Choose here the state tomography method to be used, then the process tomography scheme
#:

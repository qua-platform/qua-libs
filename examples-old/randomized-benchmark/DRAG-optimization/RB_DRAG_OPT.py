from random import randint

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from configuration import config, gauss, gauss_der
from scipy import optimize

# Open communication with the server.
QMm = QuantumMachinesManager()

cliffords = [
    ["I"],
    ["X"],
    ["Y"],
    ["Y", "X"],
    ["X/2", "Y/2"],
    ["X/2", "-Y/2"],
    ["-X/2", "Y/2"],
    ["-X/2", "-Y/2"],
    ["Y/2", "X/2"],
    ["Y/2", "-X/2"],
    ["-Y/2", "X/2"],
    ["-Y/2", "-X/2"],
    ["X/2"],
    ["-X/2"],
    ["Y/2"],
    ["-Y/2"],
    ["-X/2", "Y/2", "X/2"],
    ["-X/2", "-Y/2", "X/2"],
    ["X", "Y/2"],
    ["X", "-Y/2"],
    ["Y", "X/2"],
    ["Y", "-X/2"],
    ["X/2", "Y/2", "X/2"],
    ["-X/2", "Y/2", "-X/2"],
]

# for simulation purposes
clifford_fidelity = {
    "I": 1,
    "X/2": 0.99,
    "X": 0.99,
    "-X/2": 0.99,
    "Y/2": 0.99,
    "Y": 0.99,
    "-Y/2": 0.99,
}


def get_error_dep_fidelity(err, op):
    return clifford_fidelity[op] * np.exp(-err / 10)


def get_simulated_fidelity(ops_list, err=0):
    fidelity = 1
    for op in ops_list:
        fidelity = fidelity * get_error_dep_fidelity(err, op)
    return fidelity


def recovery_clifford(state):
    # operations = {'x': ['I'], '-x': ['Y'], 'y': ['X/2', '-Y/2'], '-y': ['-X/2', '-Y/2'], 'z': ['-Y/2'], '-z': ['Y/2']}
    operations = {
        "z": ["I"],
        "-x": ["-Y/2"],
        "y": ["X/2"],
        "-y": ["-X/2"],
        "x": ["Y/2"],
        "-z": ["X"],
    }
    return operations[state]


def transform_state(input_state: str, transformation: str):
    transformations = {
        "x": {
            "I": "x",
            "X/2": "x",
            "X": "x",
            "-X/2": "x",
            "Y/2": "z",
            "Y": "-x",
            "-Y/2": "-z",
        },
        "-x": {
            "I": "-x",
            "X/2": "-x",
            "X": "-x",
            "-X/2": "-x",
            "Y/2": "-z",
            "Y": "x",
            "-Y/2": "z",
        },
        "y": {
            "I": "y",
            "X/2": "z",
            "X": "-y",
            "-X/2": "-z",
            "Y/2": "y",
            "Y": "y",
            "-Y/2": "y",
        },
        "-y": {
            "I": "-y",
            "X/2": "-z",
            "X": "y",
            "-X/2": "z",
            "Y/2": "-y",
            "Y": "-y",
            "-Y/2": "-y",
        },
        "z": {
            "I": "z",
            "X/2": "-y",
            "X": "-z",
            "-X/2": "y",
            "Y/2": "-x",
            "Y": "-z",
            "-Y/2": "x",
        },
        "-z": {
            "I": "-z",
            "X/2": "y",
            "X": "z",
            "-X/2": "-y",
            "Y/2": "x",
            "Y": "z",
            "-Y/2": "-x",
        },
    }

    return transformations[input_state][transformation]


def play_clifford(clifford, state: str):
    for op in clifford:
        state = transform_state(state, op)
        if op != "I":
            play(op, "qubit")
    return state


def randomize_and_play_circuit(n_gates: int, init_state: str = "x"):
    state = init_state
    for ind in range(n_gates):
        state = play_clifford(cliffords[np.random.randint(0, len(cliffords))], state)
    return state


def randomize_interleaved_circuit(interleave_op: str, d: int, init_state: str = "x"):
    state = init_state
    ops_list = []
    for ind in range(d):
        c = cliffords[np.random.randint(0, len(cliffords))]
        for op in c:
            ops_list.append(op)
        state = play_clifford(c, state)
        state = play_clifford(interleave_op, state)
    return state, ops_list


QM1 = QMm.open_qm(config)

N_avg = 1
# circuit_depth_vec = list(range(1, 10, 2))


t1 = 10


def drag_prog(e, d=20):
    with program() as drag_RBprog:
        N = declare(int)
        I = declare(fixed)
        out_str = declare_stream()

        F = declare(fixed)
        F_str = declare_stream()
        with for_(N, 0, N < N_avg, N + 1):
            # for depth in circuit_depth_vec:
            final_state, ops_list = randomize_interleaved_circuit(["I"], d)
            assign(F, get_simulated_fidelity(ops_list, err=e))
            save(F, F_str)
            play_clifford(recovery_clifford(final_state), final_state)
            align("rr", "qubit")
            measure("readout", "rr", None, integration.full("integW1", I))
            save(I, out_str)
            wait(10 * t1, "qubit")

        with stream_processing():
            out_str.save_all("out_stream")
            F_str.save_all("F_stream")
    return drag_RBprog


def cost(x):
    # x[0] = alpha, x[1]=beta
    config["waveforms"]["DRAG_gauss_wf"]["samples"] = gauss(x[0] * 0.2, 0, 6, 0, 100)  # update the config
    config["waveforms"]["DRAG_gauss_wf"]["samples"] = gauss_der(x[1] * 0.2, 0, 6, 0, 100)  # update the config
    QM1 = QMm.open_qm(config)  # reopen the QM using new config file e.g. new waveform
    optimal_x = [1, 0.5]
    e = np.sqrt(np.sum((optimal_x - x) ** 2))
    job = QM1.simulate(drag_prog(e=e), SimulationConfig(int(3000)))
    res = job.result_handles
    F = res.F_stream.fetch_all()["value"]
    # F = F.reshape(N_avg, len(circuit_depth_vec))
    F_avg = F.mean(axis=0)
    err = 1 - F_avg
    print(err)

    return err


res = optimize.minimize(cost, x0=[1.2, 0.3], method="nelder-mead", options={"xatol": 1e-2, "disp": True})
x_the = np.array([1, 0.5])
x_0 = np.array([1.2, 0.3])
opt_x = res.x
e = np.sqrt(np.sum((x_0 - x_the) ** 2))
e_f = np.sqrt(np.sum((opt_x - x_the) ** 2))

config["waveforms"]["DRAG_gauss_wf"]["samples"] = gauss(x_0[0] * 0.2, 0, 6, 0, 100)  # update the config
config["waveforms"]["DRAG_gauss_wf"]["samples"] = gauss_der(x_0[1] * 0.2, 0, 6, 0, 100)  # update the config
QM1 = QMm.open_qm(config)  # reopen the QM using new config file e.g. new waveform

circuit_depth_vec = np.sort(list(set(np.logspace(0, 2, 10).astype(int)))).tolist()

# run with x_0 params
def RB_scan(e):
    with program() as drag_RBprog:
        N = declare(int)
        I = declare(fixed)
        out_str = declare_stream()

        F = declare(fixed)
        F_str = declare_stream()
        with for_(N, 0, N < N_avg, N + 1):
            for depth in circuit_depth_vec:
                final_state, ops_list = randomize_interleaved_circuit(["I"], depth)
                assign(F, get_simulated_fidelity(ops_list, err=e))
                save(F, F_str)
                play_clifford(recovery_clifford(final_state), final_state)
                align("rr", "qubit")
                measure("readout", "rr", None, integration.full("integW1", I))
                save(I, out_str)
                wait(10 * t1, "qubit")

        with stream_processing():
            out_str.save_all("out_stream")
            F_str.save_all("F_stream")
    return drag_RBprog


job = QM1.simulate(RB_scan(e), SimulationConfig(int(30000)))

res_0 = job.result_handles
F_0 = res_0.F_stream.fetch_all()["value"]
F_0 = F_0.reshape(N_avg, len(circuit_depth_vec))
F_0_avg = F_0.mean(axis=0)

job = QM1.simulate(RB_scan(e_f), SimulationConfig(int(30000)))

res_f = job.result_handles
F_f = res_f.F_stream.fetch_all()["value"]
F_f = F_f.reshape(N_avg, len(circuit_depth_vec))
F_f_avg = F_f.mean(axis=0)

plt.figure()
plt.plot(circuit_depth_vec, F_0_avg, "o-")
plt.plot(circuit_depth_vec, F_f_avg, "o-")
plt.show()
# job = QM1.simulate(drag_RBprog,
#                    SimulationConfig(int(10000)))
#
# samples = job.get_simulated_samples()
#
# samples.con1.plot()
# res=job.result_handles
#

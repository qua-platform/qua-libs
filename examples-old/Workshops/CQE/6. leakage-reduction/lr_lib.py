from configuration import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
import numpy as np
from scipy.interpolate import interp1d


def get_program(config, params, t, N_avg, d):
    """
    A function to generate the QUA program
    :param config: the QM config dictionary
    :param params: parameter list for optimization
    :param t: duration of DRAG pulses in ns.
    :param N_avg: number of runs per RB circuit realization
    :param d: depth of the randomized circuit
    :return:
    """
    th = 0
    state, op_list = update_waveforms(params, d, config, t)
    with program() as drag_RB_prog:
        N = declare(int)
        I = declare(fixed)
        state_estimate = declare(bool)
        out_str = declare_stream()
        F = declare(fixed)
        F_str = declare_stream()
        update_frequency("qubit", params[2])
        with for_(N, 0, N < N_avg, N + 1):
            play("random_clifford_seq", "qubit")
            ## compute the recovery operation
            recovery_op = recovery_clifford(state)[0]
            if recovery_op == "I":
                wait(gauss_len, "qubit")
            else:
                play(recovery_op, "qubit")
            assign(F, get_simulated_fidelity(op_list, err=e))
            save(F, F_str)
            align("rr", "qubit")
            measure("readout", "rr", None, integration.full("integW1", I))
            assign(state_estimate, I > th)
            save(state_estimate, out_str)
            wait(500, "qubit")
        with stream_processing():
            out_str.save_all("out_stream")
            F_str.save_all("F_stream")
    return drag_RB_prog


def get_result(prog, duration, K=10):
    """
    Upload the waveforms to the configuration and re-open the QM

    :param prog: QUA program
    :param duration: simulation duration
    :return:
    """

    QMm = QuantumMachinesManager()
    QMm.close_all_quantum_machines()
    QM = QMm.open_qm(config)
    F_avg = []
    for _ in range(K):
        job = QM.simulate(prog, SimulationConfig(duration))
        res = job.result_handles
        F = res.F_stream.fetch_all()["value"]
        F_avg.append(F.mean(axis=0))
    err = 1 - np.array(F_avg).mean()
    return err


def cost_DRAG(params):
    """
    Get the cost of an unmodified DRAG pulse
    :param params: parameter list for optimization
    :return:
    """

    _params = list(params) + [0] * n_params
    prog = get_program(config, _params, pulse_duration, N_avg, depth)
    return get_result(prog, 1000)


def cost_optimal_pulse(params):
    """
    Get the cost of the modified DRAG pulse
    :param params: parameter list for optimization
    :return:
    """
    prog = get_program(config, params, pulse_duration, N_avg, depth)
    return get_result(prog, 1000)


def DRAG_I(A, mu, sigma):
    """
    Generate a Gaussian waveform
    :param A:
    :param mu:
    :param sigma:
    :return:
    """

    def f(t):
        return A * np.exp(-((t - mu) ** 2) / (2.0 * sigma) ** 2)

    return f


def DRAG_Q(B, mu, sigma):
    """
    Generate a Gaussian derivative waveform
    :param B:
    :param mu:
    :param sigma:
    :return:
    """

    def f(t):
        return B * (-2.0 * (t - mu)) * np.exp(-((t - mu) ** 2) / (2.0 * sigma**2))

    return f


def manual_ssb(IQ_pair, IF, time_stamp):
    IQ_pair = np.array(IQ_pair)
    upconverted_IQ = np.zeros_like(IQ_pair)
    for idx, pair in enumerate([IQ_pair[:, x] for x in range(IQ_pair.shape[1])]):
        theta = IF * (time_stamp + idx)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        upconverted_IQ[:, idx] = np.matmul(R, pair)

    return (upconverted_IQ[0, :].tolist(), upconverted_IQ[1, :].tolist())


def get_DRAG_pulse(gate: str, params: list, t: float):
    """
    Generate a modified DRAG pulse based on the params structure
    :param gate:
    :param params:
    :param t:
    :return:
    """

    ## params [A, B, freq, a0, a1, a2, .... an-1, b0, b1 .....bn-1]
    _n_params = (len(params) - 3) // 2
    _ts = np.linspace(0.0, t, _n_params)
    if np.sum(params[3:]) == 0.0:
        an_func = lambda x: 0
        bn_func = lambda x: 0
    else:
        an_func = interp1d(params[3 : _n_params + 3], _ts, fill_value="extrapolate")
        bn_func = interp1d(params[_n_params + 3 :], _ts, fill_value="extrapolate")
    ns = int(t)
    ts = np.linspace(0.0, t, ns)
    ts[-1] -= 0.01
    ts[0] += 0.01
    I_t = DRAG_I(params[0], t / 2, t)  # we suppose that we optimize the I,Q quadratures for X/2 gate
    Q_t = DRAG_Q(params[1], t / 2, t)
    if gate == "X/2":
        I = [(I_t(_t) + an_func(_t)) for _t in ts]
        Q = [(Q_t(_t) + bn_func(_t)) for _t in ts]
        return (I, Q)
    elif gate == "-X/2":
        I = [(-I_t(_t) - an_func(_t)) for _t in ts]
        Q = [(Q_t(t - _t) + bn_func(t - _t)) for _t in ts]
        return (I, Q)
    elif gate == "Y/2":
        I = [(Q_t(_t) + bn_func(_t)) for _t in ts]
        Q = [(-I_t(_t) - an_func(_t)) for _t in ts]
        return (I, Q)
    elif gate == "-Y/2":
        I = [(-Q_t(_t) - bn_func(_t)) for _t in ts]
        Q = [(I_t(t - _t) + an_func(t - _t)) for _t in ts]
        return (I, Q)
    elif gate == "X":
        I = [(I_t(_t) + an_func(_t)) for _t in ts] * 2
        Q = [(Q_t(_t) + bn_func(_t)) for _t in ts] * 2
        return (I, Q)
    elif gate == "Y":
        I = [(I_t(_t) + an_func(_t)) for _t in ts] * 2
        Q = [(Q_t(_t) + bn_func(_t)) for _t in ts] * 2
        return (I, Q)


def recovery_clifford(state):
    """
    Returns the recovery clifford operation matching the final state (after the RB circuit)
    :param state:
    :return:
    """

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
    """
    A function to update the state after each clifford operation
    :param input_state:
    :param transformation:
    :return:
    """
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


def update_waveforms(params: list, d: int, config: dict, t: float):
    """
    Randomize the circuit and update configuration with the waveforms
    :param params:
    :param d:
    :param config:
    :param t:
    :return:
    """
    state = "-z"  # initial state
    cliffords = ["X/2", "-X/2", "Y/2", "-Y/2"]  # generating clifford operations
    op_list = []
    I, Q = [], []
    config["pulses"]["random_sequence"]["length"] = 0
    for gate_num in range(d):
        c = cliffords[np.random.randint(4)]
        op_list.append(c)
        state = transform_state(state, c)
        I_Q = get_DRAG_pulse(c, params, t)

        if use_manual_ssb:
            I_Q = manual_ssb(I_Q, manual_ssb_IF, gate_num * t)

        I += I_Q[0]
        Q += I_Q[1]
        config["pulses"]["random_sequence"]["length"] += len(I_Q[0])
    config["waveforms"]["random_I"]["samples"] = I
    config["waveforms"]["random_Q"]["samples"] = Q

    for gate in cliffords + ["X", "Y"]:
        I = np.zeros(16)
        Q = np.zeros(16)  # 16ns is the minimum duration of a play statement
        Ir, Qr = get_DRAG_pulse(gate, params, t)
        I[: len(Ir)] = Ir
        Q[: len(Qr)] = Qr
        config["pulses"]["DRAG_PULSE_" + gate]["length"] = len(I)
        config["waveforms"]["DRAG_gauss_wf_" + gate]["samples"] = I
        config["waveforms"]["DRAG_gauss_der_wf_" + gate]["samples"] = Q
    return state, op_list


def get_error_dep_fidelity(err, op):
    """
    Function to emulate increasing performance with decreasing optimization cost
    (For a single clifford operation)
    :param err:
    :param op:
    :return:
    """
    return clifford_fidelity[op] * np.exp(-err / 10)


def get_simulated_fidelity(ops_list, err=0):
    """
    Function to emulate increasing performance with decreasing optimization cost
    (For a single a full circuit)
    :param ops_list:
    :param err:
    :return:
    """
    fidelity = 1
    for op in ops_list:
        fidelity = fidelity * get_error_dep_fidelity(err, op)
    return fidelity


# globals

clifford_fidelity = {
    "I": 1,
    "X/2": 0.99,
    "X": 0.99,
    "-X/2": 0.99,
    "Y/2": 0.99,
    "Y": 0.99,
    "-Y/2": 0.99,
}
pulse_duration = 4.19
n_params = 20
N_avg = 1
depth = 1000

x_the = np.array([1, 2.3, 100e6])
x_0 = np.array([1.2, 2.2, 110e6])
e = np.sqrt(np.sum((x_0 - x_the) ** 2))
ts = np.linspace(0.0, readout_len, readout_len)
config["waveforms"]["readout_wf"]["samples"] = [DRAG_I(1.0, 0.5 * readout_len, readout_len)(t) for t in ts]

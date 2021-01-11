import numpy as np
from scipy.interpolate import interp1d
import cma

from configuration import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface


def DRAG_I(A, mu, sigma):
    def f(t):
        return A*np.exp(-(t-mu) ** 2/( 2.0 * sigma) ** 2)
    return f

def DRAG_Q(B, mu, sigma):
    def f(t):
        return B * (-2.0*(t-mu))*np.exp(-((t - mu) ** 2) / (2.0 * sigma ** 2))
    return f

def get_DRAG_pulse(gate: str, params: list, t: float):
    ## params [A, B, freq, a0, a1, a2, .... an-1, b0, b1 .....bn-1]
    _ts = np.linspace(0.0, t, n_params//2)
    if np.sum(params[3:]) == 0:
        an_func = lambda x:0
        bn_func = lambda x:0
    else:
        an_func = interp1d(params[3:n_params//2+3], _ts)
        bn_func = interp1d(params[n_params//2+3:], _ts)
    ns = int(t)
    ts = np.linspace(0.0, t, ns)
    I_t = DRAG_I(params[0], t/2, t)
    Q_t = DRAG_Q(params[1], t/2, t)
    I = [(I_t(_t) + an_func(_t)) for _t in ts]
    Q = [(Q_t(_t) + bn_func(_t)) for _t in ts]
    return (I, Q)

def recovery_clifford(state):
    # operations = {'x': ['I'], '-x': ['Y'], 'y': ['X/2', '-Y/2'], '-y': ['-X/2', '-Y/2'], 'z': ['-Y/2'], '-z': ['Y/2']}
    operations = {'z': ['I'], '-x': ['-Y/2'], 'y': ['X/2'], '-y': ['-X/2'], 'x': ['Y/2'], '-z': ['X']}
    return operations[state]

def transform_state(input_state: str, transformation: str):
    transformations = {'x': {'I': 'x', 'X/2': 'x', 'X': 'x', '-X/2': 'x', 'Y/2': 'z', 'Y': '-x', '-Y/2': '-z'},
                       '-x': {'I': '-x', 'X/2': '-x', 'X': '-x', '-X/2': '-x', 'Y/2': '-z', 'Y': 'x', '-Y/2': 'z'},
                       'y': {'I': 'y', 'X/2': 'z', 'X': '-y', '-X/2': '-z', 'Y/2': 'y', 'Y': 'y', '-Y/2': 'y'},
                       '-y': {'I': '-y', 'X/2': '-z', 'X': 'y', '-X/2': 'z', 'Y/2': '-y', 'Y': '-y', '-Y/2': '-y'},
                       'z': {'I': 'z', 'X/2': '-y', 'X': '-z', '-X/2': 'y', 'Y/2': '-x', 'Y': '-z', '-Y/2': 'x'},
                       '-z': {'I': '-z', 'X/2': 'y', 'X': 'z', '-X/2': '-y', 'Y/2': 'x', 'Y': 'z', '-Y/2': '-x'}}
    return transformations[input_state][transformation]

def update_randomized_waveform(params: list, d: int, config: dict, t: float):
    state = '-z' 
    cliffords = ["X/2", "-X/2", "Y/2", "-Y/2"]
    op_list = []
    I, Q = [], []
    config["pulses"]["random_sequence"]["length"] = 0
    for _ in range(d):
        c=cliffords[np.random.randint(4)]
        op_list.append(c)
        state = transform_state(state, c)
        I_Q = get_DRAG_pulse(c, params, t)
        I += I_Q[0]
        Q += I_Q[1]
        config["pulses"]["random_sequence"]["length"] += len(I_Q[0])
    config["waveforms"]["random_I"]["samples"] = I
    config["waveforms"]["random_Q"]["samples"] = Q
    return state, op_list

clifford_fidelity = {'I': 1, 'X/2': 0.99, 'X': 0.99, '-X/2': 0.99, 'Y/2': 0.99, 'Y': 0.99, '-Y/2': 0.99}

def get_error_dep_fidelity(err,op):
    return clifford_fidelity[op]*np.exp(-err/10)

def get_simulated_fidelity(ops_list,err=0):
    fidelity=1
    for op in ops_list:
        fidelity=fidelity*get_error_dep_fidelity(err,op)
    return fidelity

def get_program(config, params, t, N_avg, d):
    state, op_list = update_randomized_waveform(params, d, config, t)
    with program() as drag_RB_prog:
         N = declare(int)
         I = declare(fixed)
         out_str = declare_stream()
         F = declare(fixed)
         F_str=declare_stream()
         update_frequency("qubit", params[2])
         with for_(N, 0, N<N_avg, N+1): 
            play('random_clifford_seq', 'qubit')
            ## compute the recovery operation
            recovery_op = recovery_clifford(state)[0]
            if recovery_op == 'I':
                wait(gauss_len, 'element')
            else:
                play(recovery_op, 'qubit')
            assign(F, get_simulated_fidelity(op_list, err=e))
            save(F, F_str)
            align('rr', 'qubit')
            measure('readout', 'rr', None, integration.full('integW1', I))
            save(I, out_str)
            wait(500, 'qubit')
         with stream_processing():
             out_str.save_all('out_stream')
             F_str.save_all('F_stream')
    return drag_RB_prog

def get_result(prog, duration): 
    QMm = QuantumMachinesManager(host='3.122.60.129')
    QM = QMm.open_qm(config)
    job = QM.simulate(prog,
                      SimulationConfig(duration))
    res = job.result_handles
    F = res.F_stream.fetch_all()['value']
    F_avg = F.mean(axis=0)
    err=1-F_avg
    return err

def cost_DRAG(params):
    _params = list(params) + [0]*n_params
    prog = get_program(config, _params, pulse_duration, N_avg, depth)
    return get_result(prog, 1000)

def cost_optimal_pulse(params):
    prog = get_program(config, params, pulse_duration, N_avg, depth)
    return get_result(prog, 1000)

pulse_duration = 4.19
n_params = 20
N_avg = 10
depth = 20

x_the=np.array([1, 2.3, 100e6])
x_0=np.array([1.2, 2.2, 110e6])
e = np.sqrt(np.sum((x_0 - x_the) ** 2))
gauss_len = 100

ts = np.linspace(0.0, 4.0*gauss_len, gauss_len) 
config["pulses"]["DRAG_PULSE"]["length"] = gauss_len
config["waveforms"]["gauss_wf"]["samples"] = [DRAG_I(1.0, 2.0*gauss_len, 4.0*gauss_len)(t) for t in ts]
config["waveforms"]["DRAG_gauss_wf"]["samples"] = [DRAG_I(1.0, 2.0*gauss_len, 4.0*gauss_len)(t) for t in ts]
config["waveforms"]["DRAG_gauss_der_wf"]["samples"] = [DRAG_Q(1.0, 2.0*gauss_len, 4.0*gauss_len)(t) for t in ts]

##optimize DRAG
es1 = cma.CMAEvolutionStrategy(np.random.rand(3), 0.5)
es1.optimize(cost_DRAG)
    
#optimize pulse
## use A, B, freq as initial guess for the full optimization
es2 = cma.CMAEvolutionStrategy((3 + 2*int(pulse_duration)) * [0], 0.5)
es2.optimize(cost_optimal_pulse)

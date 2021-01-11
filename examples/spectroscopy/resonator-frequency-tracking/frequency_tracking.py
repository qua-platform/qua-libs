from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from qm.qua import *
from config import config


def track_frequency(qe, pulse, current_w, w_step):
    res_w = declare(fixed)
    res_w_bigger = declare(fixed)
    res_w_smaller = declare(fixed)

    measure(qe, pulse, current_w - w_step, res_w_smaller)
    measure(qe, pulse, current_w, res_w)
    measure(qe, pulse, current_w + w_step, res_w_bigger)

    best_w = Util.cond(res_w_smaller < res_w, current_w - w_step,
                       Util.cond(res_w <= res_w_bigger, current_w, current_w + w_step))
    assign(current_w, best_w)
    update_frequency(qe, current_w)


qmm = QuantumMachinesManager(host='127.0.0.1')

with program() as w_if_tracking:
    qe = "qe1"
    pulse = "readout_pulse"
    current_w = declare(int, value=int(10e6))
    calibration_step = declare(int, value=int(1e6))

    N = declare(int)
    with for_(N, 0, N < 100, N + 1):
        track_frequency(qe, pulse, current_w, calibration_step)
        #####################
        # rest of code here #
        #####################

job = qmm.simulate(config, w_if_tracking, SimulationConfig(
    duration=1000,  # duration of simulation in units of 4ns
))
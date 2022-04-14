from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from Configuration import config

# from Configuration import qm


def measure_spin(basis, threshold, state):
    meas_len = 1e3

    result1 = declare(int, size=int(meas_len / 500))
    resultLen1 = declare(int)
    result2 = declare(int, size=int(meas_len / 500))
    resultLen2 = declare(int)
    counts = declare(int)

    if basis == "x":
        play("pi2", "spin_qubit")
    align("spin_qubit", "readout", "readout1", "readout2")
    play("on", "readout", duration=800)
    measure(
        "readout",
        "readout1",
        None,
        time_tagging.analog(result1, meas_len, targetLen=resultLen1),
    )
    measure(
        "readout",
        "readout2",
        None,
        time_tagging.analog(result2, meas_len, targetLen=resultLen2),
    )
    assign(counts, resultLen1 + resultLen2)
    assign(state, counts > threshold)


def reset_spin(threshold):
    se = declare(bool)
    measure_spin("z", threshold, se)
    with while_(se):
        play("pi", "spin_qubit")
        measure_spin("z", threshold, se)

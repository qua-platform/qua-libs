from xeb_utils import XEB
from xeb_config import config
from qm import SimulationConfig
from qm.QmJob import QmJob
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager

m_max = 100
xeb = XEB(config, m_max, ["q1", "q2", "coupler"])
xeb_sequence = xeb.baked_sequence
xeb_op_list = xeb.operations_list
xeb_duration_tracker = xeb.duration_tracker

with program() as xeb_program:
    update_frequency("q1", 0)
    update_frequency("q2", 0)
    align("q1", "q2")
    truncate = declare(int)
    truncate_array = declare(int, value=[x // 4 for x in xeb_duration_tracker])

    I1 = declare(fixed)
    I2 = declare(fixed)
    with for_each_(truncate, truncate_array):
        align(*xeb_sequence.elements)
        play(xeb_sequence.operations["q1"], "q1", truncate=truncate)
        play(xeb_sequence.operations["q2"], "q2", truncate=truncate)
        play(xeb_sequence.operations["coupler"], "coupler", truncate=truncate)
        align(*xeb_sequence.elements)
        measure("readout", "rr", None, demod.full("integW1", I1, "out1"))
        save(I1, "I1")
        measure("readout", "rr", None, demod.full("integW1", I2, "out1"))
        save(I2, "I2")

qmm = QuantumMachinesManager()
job: QmJob = qmm.simulate(config, xeb_program, SimulationConfig(1500))

from qualang_tools.bakery.bakery import Baking

from qualang_tools.bakery.xeb import XEB, XEBOpsSingleQubit
from xeb_config import config, pulse_len
from qm import SimulationConfig
from qm.QmJob import QmJob
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager

import matplotlib.pyplot as plt


def id1(baking: Baking):
    baking.wait(pulse_len, "q1")


def id2(baking: Baking):
    baking.wait(pulse_len, "q2")


def baked_cphase(baking: Baking):
    baking.play("coupler_op", "coupler")


def sx1(baking: Baking):
    baking.play("sx", "q1")


def sx2(baking: Baking):
    baking.play("sx", "q2")


def sy1(baking: Baking):
    baking.play("sy", "q1")


def sy2(baking: Baking):
    baking.play("sy", "q2")


def sw1(baking: Baking):
    baking.frame_rotation_2pi(0.125, "q1")
    baking.play("sx", "q1")
    baking.frame_rotation_2pi(-0.125, "q1")


def sw2(baking: Baking):
    baking.frame_rotation_2pi(0.125, "q2")
    baking.play("sx", "q2")
    baking.frame_rotation_2pi(-0.125, "q2")


def align_op(baking: Baking):
    baking.align("q1", "q2", "coupler")


xeb = XEB(
    config,
    m_max=10,
    q1_ops=XEBOpsSingleQubit(id=id1, sx=sx1, sy=sy1, sw=sw1),
    q2_ops=XEBOpsSingleQubit(id=id2, sx=sx2, sy=sy2, sw=sw2),
    two_qubit_op=baked_cphase,
    align_op=align_op,
)

with program() as prog:
    truncate = declare(int)
    truncate_array = declare(int, value=[x // 4 for x in xeb.duration_tracker])

    I1 = declare(fixed)
    I2 = declare(fixed)
    with for_each_(truncate, truncate_array):
        for element in xeb.baked_sequence.elements:
            play(xeb.baked_sequence.operations[element], element, truncate=truncate)
        align()
        measure("readout", "rr", None, demod.full("integW1", I1, "out1"))
        save(I1, "I1")
        measure("readout", "rr", None, demod.full("integW1", I2, "out1"))
        save(I2, "I2")

qmm = QuantumMachinesManager()
job: QmJob = qmm.simulate(config, prog, SimulationConfig(1500))
job.get_simulated_samples().con1.plot()

plt.show()

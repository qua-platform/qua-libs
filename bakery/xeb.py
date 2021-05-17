import bakery

from qm import SimulationConfig
from qm.QmJob import QmJob
from qm.qua import *
import numpy as np
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt

rnd_gate_map = {
    0: 'sx',
    1: 'sy',
    2: 'sw'
}
m_max = 1600
np.random.seed(0)
gate_sequence = []
rand_seq1 = np.random.randint(3, size=m_max)
rand_seq2 = np.random.randint(3, size=m_max)

with bakery.baking(config) as b:
    for rnd1, rnd2 in zip(rand_seq1, rand_seq2):
        b.align('q1', 'q2', 'coupler')
        b.play(rnd_gate_map[rnd1], 'q1')
        b.play(rnd_gate_map[rnd2], 'q2')
        b.align('q1', 'q2', 'coupler')
        b.play('coupler_op', 'coupler')

with program() as xeb_concat:
    update_frequency('q1', 0)
    update_frequency('q2', 0)
    align()
    truncate = declare(int)

    I1 = declare(fixed)
    I2 = declare(fixed)
    with for_(truncate, 10, truncate < m_max * 10, truncate + 10):
        align(b.get_qe_set())
        play(b.get_Op_name('q1'), 'q1', truncate=truncate)
        play(b.get_Op_name('q2'), 'q2', truncate=truncate)
        play(b.get_Op_name('coupler'), 'coupler', truncate=truncate)
        align()
        measure('readout', 'rr1', None, demod.full('integW_cos', I1, 'out1'))
        save(I1, 'I1')
        measure('readout', 'rr2', None, demod.full('integW_cos', I2, 'out1'))
        save(I2, 'I2')


# todo: if there are multipule baking environments, what is the correct way to merge the configs of each one?
# (need to ask Tal/Guy)
qmm = QuantumMachinesManager()
job: QmJob = qmm.simulate(config,
                          xeb_concat,
                          SimulationConfig(1500))


samples = job.get_simulated_samples()
q1_xy = samples.con1.analog['1'] + 1j * samples.con1.analog['2']
q2_xy = samples.con1.analog['3'] + 1j * samples.con1.analog['4']
rr_xy = samples.con1.analog['9'] + 1j * samples.con1.analog['10']
q1_z = samples.con1.analog['5']
q2_z = samples.con1.analog['7']
c12_z = samples.con1.analog['6']

fig, ax = plt.subplots(6, 1, sharex='all')
plot_channel(ax[0], q1_xy, 'q1_xy')
plot_channel(ax[1], q2_xy, 'q2_xy')
plot_channel(ax[2], q1_z, 'q1_z')
plot_channel(ax[3], q2_z, 'q2_z')
plot_channel(ax[4], c12_z, 'c12_z')
plot_channel(ax[5], rr_xy, 'rr_xy')
ax[5].set_xlabel('samples')
plt.show()

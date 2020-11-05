from StateDiscriminator import StateDiscriminator
from configuration import config

from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

simulation_config = SimulationConfig(
    duration=800000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.1 ** 2
    )
)

N = [200, 200, 200, 200]
states = ['g', 'e', 'f', 'h']

wait_time = 10
with program() as training_program:
    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    for state, shots in zip(states, N):
        with for_(n, 0, n < shots, n + 1):
            measure("readout_pulse_" + state, "rr1a", "adc",
                    demod.full("integW_cos", I1, "out1"),
                    demod.full("integW_sin", Q1, "out1"),
                    demod.full("integW_cos", I2, "out2"),
                    demod.full("integW_sin", Q2, "out2"))
            assign(I, I1 + Q2)
            assign(Q, -Q1 + I2)
            save(I, 'I')
            save(Q, 'Q')
            wait(wait_time, "rr1a")

qmm = QuantumMachinesManager()
discriminator = StateDiscriminator(qmm, config, 'rr1a', len(states), 'gef_disc_params.npz')
discriminator.train(program=training_program, plot=True, dry_run=True, simulate=simulation_config)

with program() as test_program:
    n = declare(int)
    res = declare(int)
    seq0 = []
    for state, shots in zip(states, N):
        with for_(n, 0, n < shots, n + 1):
            discriminator.measure_state("readout_pulse_" + state, "out1", "out2", res)

            save(res, 'res')
            wait(wait_time, "rr1a")

qm = qmm.open_qm(config)
job = qm.execute(test_program, duration_limit=0, data_limit=0, dry_run=True, simulate=simulation_config)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()['value']

p_s = np.zeros(shape=(len(states), len(states)))
measures_per_state = [0] + list(np.cumsum(N))
for i in range(len(states)):
    res_i = res[measures_per_state[i]:measures_per_state[i + 1]]
    p_s[i, :] = np.array([np.mean(res_i == j) for j in range(len(states))])

fig = plt.figure()
ax = plt.subplot()
sns.heatmap(p_s, annot=True, ax=ax, fmt='g', cmap='Blues')
ax.set_xlabel('Predicted states')
ax.set_ylabel('True states')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(states)
ax.yaxis.set_ticklabels(states)
plt.show()

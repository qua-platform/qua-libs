"""
multilevel_discriminator.py: Multilevel discriminator for qubit state measurement
Author: Ilan Mitnikov, Nir Halay - Quantum Machines
Created: 8/11/2020
Created on QUA version: 0.5.170
"""

from StateDiscriminator import StateDiscriminator
from configuration import config

from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

simulation_config = SimulationConfig(
    duration=int(2e5),
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)],
        latency=230,
        noisePower=0.1**2,
    ),
)

states = ["g", "e", "f"]  # state labels
N = 200  # number of shots per state
"""
Training
"""
wait_time = 10
with program() as training_program:
    """
    This program prepares the qubits in each of the states and then measures the readout response
    """
    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    for state in states:
        with for_(n, 0, n < N, n + 1):
            # prepare qubit state
            play("prepare_" + state, "qb1a")
            # send a mixed IQ readout pulse, and demodulate in order to get the I and Q components
            # In an experiment with a real physical system use just 'readout_pulse' instead.
            align("qb1a", "rr1a")
            measure(
                "readout_pulse_" + state,
                "rr1a",
                "adc",
                demod.full("integW_cos", I1, "out1"),
                demod.full("integW_sin", Q1, "out1"),
                demod.full("integW_cos", I2, "out2"),
                demod.full("integW_sin", Q2, "out2"),
            )
            assign(I, I1 + Q2)
            assign(Q, -Q1 + I2)
            save(I, "I")
            save(Q, "Q")
            wait(wait_time, "rr1a")

qmm = QuantumMachinesManager()
discriminator = StateDiscriminator(qmm, config, "rr1a", len(states), "discriminator_params.npz")

# train the discriminator on the prepared states and readout response, for best state assignment during measurement
# for simulating:
discriminator.train(program=training_program, plot=True, dry_run=True, simulate=simulation_config)
# when running a real experiment:
# discriminator.train(program=training_program, plot=True)

"""
Testing
"""
# after training the discriminator one can use it to measure the qubit states
with program() as test_program:
    """
    This program prepares the qubit state and measures the readout response.
    Then determines the state of the qubit according to the discriminator
    """
    n = declare(int)
    res = declare(int)
    seq0 = []
    for state in states:
        with for_(n, 0, n < N, n + 1):
            # # prepare qubit state
            # align('qb1a', 'rr1a')
            # play('prepare_' + state, 'qb1a')

            # measure using loopback simulation
            discriminator.measure_state("readout_pulse_" + state, "out1", "out2", res)
            # measure a real physical system
            # discriminator.measure_state("readout_pulse", "out1", "out2", res)
            save(res, "res")
            wait(wait_time, "rr1a")

qm = qmm.open_qm(config)

# for simulating do:
job = qm.execute(
    test_program,
    duration_limit=0,
    data_limit=0,
    dry_run=True,
    simulate=simulation_config,
)
# when running a real experiment simply specify the desired duration and data limits:
# job = qm.execute(test_program, duration_limit=, data_limit=)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get("res").fetch_all()["value"]

# plot the confusion matrix to see how well the discriminator does
p_s = np.zeros(shape=(len(states), len(states)))
measures_per_state = [i * N for i in range(len(states) + 1)]
for i in range(len(states)):
    res_i = res[measures_per_state[i] : measures_per_state[i + 1]]
    p_s[i, :] = np.array([np.mean(res_i == j) for j in range(len(states))])

fig = plt.figure()
ax = plt.subplot()
sns.heatmap(p_s, annot=True, ax=ax, fmt="g", cmap="Blues")
ax.set_xlabel("Predicted states")
ax.set_ylabel("True states")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(states)
ax.yaxis.set_ticklabels(states)
plt.show()

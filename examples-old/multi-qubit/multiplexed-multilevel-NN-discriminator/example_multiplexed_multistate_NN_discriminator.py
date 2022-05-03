"""
example_multiplexed_multistate_NN_discriminator.py: Single qubit active reset
Author: Ilan Mitnikov - Quantum Machines
Created: 29/12/2020
Created on QUA version: 0.6.433
"""

import NNStateDiscriminator
from example_configuration import config, rr_num
from qm import QuantumMachinesManager
from qm.qua import *
import numpy as np

qmm = QuantumMachinesManager.QuantumMachinesManager()
resonators = ["rr" + str(i) for i in range(rr_num)]
qubits = ["qb" + str(i) for i in range(rr_num)]
path = "folder_name"
calibrate_with = ["rr0"]
discriminator = NNStateDiscriminator.NNStateDiscriminator(qmm, config, resonators, qubits, calibrate_with, path)


def prepare_qubits(state, qubits):
    align(*qubits)
    for i, s in enumerate(state):
        play("prepare" + str(s), qubits[i])


num_of_combinations = 243
states = np.random.randint(0, discriminator.num_of_states, (num_of_combinations, discriminator.rr_num))
wait_time = 100  # wait time between measurement and next state preparation
n_avg = 15  # repeat the measurement n_avg times and average the result in the training

discriminator.generate_training_data(prepare_qubits, "readout", n_avg, states, wait_time)

discriminator.train()


# Testing program
def test(state):
    with program() as multi_read:
        # initialize discriminator
        discriminator.initialize()

        # prepare qubits in a certain state
        prepare_qubits(state, discriminator.qubits)
        align(*discriminator.qubits, *discriminator.resonators)

        # measure the state of the qubits with one statement
        discriminator.measure_state("readout", "result")

    return multi_read


results = []
states2 = np.random.randint(0, discriminator.num_of_states, (10, discriminator.rr_num))
for state in states2:
    qm = qmm.open_qm(discriminator.config)
    job1 = qm.execute(test(state))

    job1.result_handles.wait_for_all_values()
    res = job1.result_handles.get("result").fetch_all()["value"]
    results.extend(res.reshape((-1, discriminator.rr_num)))

    print("prediction")
    print(res.reshape((-1, discriminator.rr_num)))
    print("actual")
    print(state)

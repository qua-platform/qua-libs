import NNStateDiscriminator
from example_configuration import config, i_port, q_port, rr_num
from qm import QuantumMachinesManager, SimulationConfig, LoopbackInterface
from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
simulation_config = SimulationConfig(
    duration=int(1e5),
    simulation_interface=LoopbackInterface(
        [("con1", i_port, "con1", 1), ("con1", q_port, "con1", 2),
         ("con2", i_port, "con1", 1), ("con2", q_port, "con1", 2),
         ("con2", i_port, "con2", 1), ("con2", q_port, "con2", 2),
         ("con1", i_port, "con2", 1), ("con1", q_port, "con2", 2)], latency=192, noisePower=0.05 ** 2
    ),
    # include_analog_waveforms=True
)

qmm = QuantumMachinesManager.QuantumMachinesManager()
resonators = ["test_rr" + str(i) for i in range(rr_num)]
qubits = ["test_qb" + str(i) for i in range(rr_num)]
path = "try35"
calibrate_with = ["test_rr1", "test_qb1", "test_rr0"]
discriminator = NNStateDiscriminator.NNStateDiscriminator(qmm, config, resonators, qubits, calibrate_with, path)


def prepare_qubits(state, qubits):
    align(*qubits)
    for i, s in enumerate(state):
        play("prepare" + str(s), qubits[i])
    pass


states = [random.choices([i for i in range(discriminator.num_of_states)], k=discriminator.rr_num) for j in range(250)]
#
wait_time = 4
n_avg = 1
discriminator.generate_training_data(prepare_qubits, "readout", n_avg, states, wait_time,
                                     # calibrate_dc_offset=False,
                                     # simulate=simulation_config, dry_run=True
                                     )

discriminator.train(epochs=300,
                    # kernel_initializer=tf.keras.initializers.Constant(1),
                    # calibrate_dc_offset=False,
                    # simulate=simulation_config, dry_run=True
                    )


def test(states):
    with program() as multi_read:
        out1 = declare(fixed, size=discriminator.rr_num)
        out2 = declare(fixed, size=discriminator.rr_num)
        res = declare(fixed, size=discriminator.num_of_states)
        temp = declare(int)
        for state in states:
            discriminator.measure_state(state, "result", out1, out2, res, temp)

    return multi_read


results = []
jump = 1
states2 = [random.choices([i for i in range(discriminator.num_of_states)], k=discriminator.rr_num) for j in range(10)]
for i in range(0, 10, jump):
    # job1 = qmm.simulate(config, test(states2[i:i + jump]), simulation_config)

    qm = qmm.open_qm(discriminator.config)
    job1 = qm.execute(test(states2[i:i + jump]))

    job1.result_handles.wait_for_all_values()
    res = job1.result_handles.get("result").fetch_all()['value']
    results.extend(res.reshape((-1, discriminator.rr_num)))

    print(i)
    print(res.reshape((-1, discriminator.rr_num)))
    print(states2[i])

# results = np.vstack(results)
# states = np.array(states)
# fig, axs = plt.subplots(2, rr_num // 2)
# for j in range(rr_num):
#     p_s = np.zeros(shape=(disc.num_of_states, disc.num_of_states))
#     for i in range(disc.num_of_states):
#         res_i = results[:, j][np.array(states[:, j]) == i]
#         p_s[i, :] = np.around(np.array([np.mean(res_i == 0), np.mean(res_i == 1), np.mean(res_i == 2)]), 3)
#     labels = ['g', 'e', 'f']
#     sns.heatmap(p_s, annot=True, ax=axs[j // (rr_num // 2), j % (rr_num // 2)], fmt='g', cmap='Blues')
#     # labels, title and ticks
#     axs[j // (rr_num // 2), j % (rr_num // 2)].set_xlabel('Predicted labels')
#     axs[j // (rr_num // 2), j % (rr_num // 2)].set_ylabel('True labels')
#     axs[j // (rr_num // 2), j % (rr_num // 2)].set_title('Confusion Matrix for Qubit ' + str(j))
#     axs[j // (rr_num // 2), j % (rr_num // 2)].xaxis.set_ticklabels(labels)
#     axs[j // (rr_num // 2), j % (rr_num // 2)].yaxis.set_ticklabels(labels)
#     plt.show()

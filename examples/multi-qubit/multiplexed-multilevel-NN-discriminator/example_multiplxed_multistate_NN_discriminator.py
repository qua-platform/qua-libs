import NNStateDiscriminator
from example_configuration import config, i_port, q_port, rr_num
from qm import QuantumMachinesManager, SimulationConfig, LoopbackInterface
from qm.qua import *
import numpy as np

simulation_config = SimulationConfig(
    duration=int(1e6),
    simulation_interface=LoopbackInterface(
        [("con1", i_port, "con1", 1), ("con1", q_port, "con1", 2),
         ("con2", i_port, "con2", 1), ("con2", q_port, "con2", 2),
         ], latency=192, noisePower=0.001 ** 2
    ),
    # include_analog_waveforms=True
)

qmm = QuantumMachinesManager.QuantumMachinesManager()
resonators = ["test_rr" + str(i) for i in range(rr_num)]
qubits = ["test_qb" + str(i) for i in range(rr_num)]
path = "try52"
calibrate_with = ["test_rr1", "test_qb1", "test_rr0"]
discriminator = NNStateDiscriminator.NNStateDiscriminator(qmm, config, resonators, qubits, calibrate_with, path)


def prepare_qubits(state, qubits):
    align(*qubits)
    for i, s in enumerate(state):
        play("prepare" + str(s), qubits[i])


# states = [random.choices([i for i in range(discriminator.num_of_states)], k=discriminator.rr_num) for j in range(350)]
num_of_combinations = 333
states = np.random.randint(0, discriminator.num_of_states, (num_of_combinations, discriminator.rr_num))
wait_time = 4  # wait time between measurement and next state preparation
n_avg = 15  # repeat the measurement n_avg times and average the result
discriminator.generate_training_data(prepare_qubits, "readout", n_avg, states, wait_time, with_timestamps=False
#                                      simulate=simulation_config, dry_run=True, calibrate_dc_offset=False
                                     )

discriminator.train(epochs=300,
                    # simulate=simulation_config, dry_run=True, calibrate_time_diff=False
                    )


def test(states):
    with program() as multi_read:
        out1 = declare(fixed, size=discriminator.rr_num)
        out2 = declare(fixed, size=discriminator.rr_num)
        res = declare(fixed, size=discriminator.num_of_states)
        temp = declare(int)

        # TODO uncomment any of the lines below to break

        # prepare_qubits(states[0], discriminator.qubits)
        # align(*discriminator.qubits)
        play("prepare0", "test_qb0")
        for state in states:
            prepare_qubits(state, discriminator.qubits)
            align(*discriminator.qubits, *discriminator.resonators)
            # play("prepare0", "test_qb0")
            discriminator.measure_state(state, "result", out1, out2, res, temp, adc='adc')
    return multi_read


simulation_config2 = SimulationConfig(
    duration=int(1e3),
    simulation_interface=LoopbackInterface(
        [("con1", i_port, "con1", 1), ("con1", q_port, "con1", 2),
         ("con2", i_port, "con2", 1), ("con2", q_port, "con2", 2),
         ], latency=192, noisePower=0.001 ** 2
    ),
    # include_analog_waveforms=True
)

results = []
jump = 2
# states2 = [random.choices([i for i in range(discriminator.num_of_states)], k=discriminator.rr_num) for j in range(10)]
states2 = np.random.randint(0, discriminator.num_of_states, (10, discriminator.rr_num))

# states2 = [[1, 0, 2, 1, 0], [1, 1, 1, 2, 0], [2, 0, 2, 1, 2], [1, 1, 0, 0, 0]]
for i in range(0, 10, jump):
    # job1 = qmm.simulate(discriminator.config, test(states2[i:i + jump]), simulation_config2)
    #
    qm = qmm.open_qm(discriminator.config)
    job1 = qm.execute(test(states2[i:i + jump]))

    job1.result_handles.wait_for_all_values()
    res = job1.result_handles.get("result").fetch_all()['value']
    results.extend(res.reshape((-1, discriminator.rr_num)))

    print(i)
    print("prediction")
    print(res.reshape((-1, discriminator.rr_num)))
    print("actual")
    print(states2[i:i + jump])

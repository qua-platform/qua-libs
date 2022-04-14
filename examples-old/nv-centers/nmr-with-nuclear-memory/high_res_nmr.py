import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *

all_elements = ["sensor", "sample", "memory"]
N_avg = 100
N_SSR = 10  # Should be increased
hf_splitting = 1.5e6  # N15 hyperfine splitting
t_e = 2000
tau_vec = [int(i) for i in np.arange(1e3, 5e4, 5e3)]


def init_nuclear_spin(target_state):
    state = declare(int)
    SSR(N_SSR, state)
    with while_(state == ~target_state):
        play("pi_x", "memory")
        SSR(N_SSR, state)


def SSR(N, result):
    """Determine the state of the nuclear spin"""
    i = declare(int)
    res_length = declare(int, value=10)
    res_vec = declare(int, size=10)
    ssr_count = declare(int)

    # run N repetitions
    assign(ssr_count, 0)
    with for_(i, 0, i < N, i + 1):
        wait(100, "sensor")
        update_frequency("sensor", NV_IF - hf_splitting)  # Changes in RT, about 200ns
        play("pi_x", "sensor")
        measure(
            "readout",
            "sensor",
            None,
            time_tagging.raw(res_vec, 300, targetLen=res_length),
        )
        assign(ssr_count, ssr_count + res_length)

        wait(100, "sensor")
        update_frequency("sensor", NV_IF + hf_splitting)
        play("pi_x", "sensor")
        measure(
            "readout",
            "sensor",
            None,
            time_tagging.raw(res_vec, 300, targetLen=res_length),
        )
        assign(ssr_count, ssr_count - res_length)

    # compare photon count to threshold and save result in variable "state"
    with if_(ssr_count > 0):
        assign(result, 1)
    with else_():
        assign(result, -1)


def CnNOTe(condition_state):
    """
    CNOT-gate on the electron spin.
    condition_state is in [-1, 0, 1] for a spin 1 nuclear or [-1, 1] for a spin half nuclear and gives the nuclear spin
    state for which the electron spin is flipped.
    """
    align(*all_elements)
    update_frequency("sensor", NV_IF + condition_state * hf_splitting)
    play("pi_x", "sensor")
    align(*all_elements)


def encode(t):
    """
    Play the encoding sequence with wait time t.
    """
    align(*all_elements)
    reset_frame("memory")
    play("pi_2_x", "memory")
    CnNOTe(1)
    wait(t // 4, "sensor")
    CnNOTe(1)
    play("pi", "sample")
    CnNOTe(-1)
    wait(t // 4, "sensor")
    CnNOTe(1)
    play("pi_2_y", "memory")
    align(*all_elements)


def decode(t):
    """
    Play the decoding sequence with wait time t.
    """
    align(*all_elements)
    play("pi_2_x", "memory")
    CnNOTe(-1)
    wait(t // 4, "sensor")
    CnNOTe(1)
    play("pi", "sample")
    CnNOTe(-1)
    wait(t // 4, "sensor")
    CnNOTe(-1)
    play("pi_2_y", "memory")
    align(*all_elements)


with program() as prog:
    n = declare(int)
    tau = declare(int)
    result_vec = declare(int, size=len(tau_vec))
    c = declare(int)

    with for_(n, 0, n < N_avg, n + 1):
        assign(c, 0)
        with for_each_(tau, tau_vec):
            init_nuclear_spin(1)
            encode(t_e)
            align(*all_elements)
            play("pi_2", "sample")
            wait(tau, "sample")
            play("pi_2", "sample")
            align(*all_elements)
            play("laser", "sensor")
            decode(t_e)
            SSR(N_SSR, result_vec[c])
            assign(c, c + 1)

    with for_(n, 0, n < result_vec.length(), n + 1):
        save(result_vec[n], "result")


qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

job = qm.simulate(prog, simulate=SimulationConfig(duration=200000))
samples = job.get_simulated_samples()

plt.figure()
offset = 0
plt.plot(samples.con1.digital["1"] + 3.5)
for i in [4, 3, 1, 2]:
    plt.plot(samples.con1.analog[str(i)] + offset)
    offset += 1
plt.yticks(
    [0, 1, 2, 3, 4],
    ["sample", "mem", "I", "Q", "Laser"],
    rotation="vertical",
    va="center",
)
plt.show()

print(samples)

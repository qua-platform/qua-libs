from configuration import *

from qm.qua import *
from qm.QuantumMachinesManager import (
    QuantumMachinesManager,
    SimulationConfig,
    LoopbackInterface,
)
import numpy as np
import matplotlib.pyplot as plt


def state_estimate(I, Q, state_var):
    with if_(Q > 0):
        assign(state_var, 1)
    with else_():
        assign(state_var, 0)


def active_reset(I, Q):
    with while_(Q > 0):
        play("pi_g_e", "charge_line")
        measure("readout", "readout_resonator", None, demod.full("integW_sin", Q, "out1"))


def prepare_e():
    update_frequency("charge_line", ge_IF)
    play("pi_g_e", "charge_line")


def prepare_f():
    prepare_e()
    update_frequency("charge_line", ef_IF)
    play("pi_e_f", "charge_line")


n_modes = 11


def extract_modes_times(population):
    """
    Extract from the 2D time-freq sweep the frequency of the modes and the iswap times
    :param population:
    :return:
    """
    # numerical values just for simulation
    return (
        6.45 - 0.5 * np.cos(np.linspace(1, n_modes, n_modes) * np.pi / (n_modes + 1)),
        np.linspace(20, 100, n_modes),
    )


def time_freq_sweep(prepare, N, sb_freqs, t_init, t_final, step):
    with program() as prog:
        freq = declare(int)
        t = declare(int)
        n = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        I1 = declare(fixed)
        Q1 = declare(fixed)
        I2 = declare(fixed)
        Q2 = declare(fixed)
        state_var = declare(int)
        state = declare_stream()
        with for_(n, 0, n < N, n + 1):
            with for_each_(freq, sb_freqs):
                update_frequency("flux_line", freq)
                with for_(t, t_init, t < t_final, t + step):
                    # prepare the transmon in the e or f states
                    prepare()
                    align("charge_line", "flux_line")

                    # modulate the flux bias to create sidebands
                    play("modulate", "flux_line", duration=t)
                    align("flux_line", "readout_resonator")

                    # measure the transmon state
                    measure(
                        "readout",
                        "readout_resonator",
                        None,
                        demod.full("integW_cos", I1, "out1"),
                        demod.full("integW_sin", Q1, "out1"),
                        demod.full("integW_cos", I2, "out2"),
                        demod.full("integW_sin", Q2, "out2"),
                    )
                    active_reset("charge_line", Q1)
                    assign(I, I1 + Q2)
                    assign(Q, -Q1 + I2)

                    # estimate and assign state
                    state_estimate(I, Q, state_var)
                    save(state_var, state)

        with stream_processing():
            state.buffer(len(sb_freqs), int((t_final - t_init) / step)).average().save("state")
        return prog


qmm = QuantumMachinesManager()
job = qmm.simulate(
    config,
    time_freq_sweep(prepare_e, 3, [1, 2, 3, 4], 32, 300, 20),
    simulate=SimulationConfig(10000),
)
e_population = job.result_handles.state.fetch_all()

sim = job.get_simulated_samples().con1
plt.plot(sim.analog["3"])
# get the modes reachable using the g-e transition
modes_ge, exchange_times_ge = extract_modes_times(e_population)


def modulate_mode(freq, t, pulse):
    """
    Modulate the flux line
    :param freq: the modulation frequency
    :param t: the modulation time
    :param pulse: the modulation pulse
    :return:
    """
    detuning = freq - (qubit_IF + qubit_LO) - flux_LO
    update_frequency("flux_line", detuning)
    play(pulse, "flux_line", duration=t)


def iswap(modes, times, k):
    """
    Perform iswap using the calibrated pulse
    :param modes:
    :param times:
    :param k:
    :return:
    """
    modulate_mode(modes[k], times[k], "iswap")


def iswap_inv(modes, times, k):
    iswap(modes, 3 * times, k)


def apply_op(modes, times, k, rotation_op):
    """
    Apply single qubit gate to the mode using the transmon
    :param modes: the frequency of the eigen modes
    :param times: the times for iswap for each mode
    :param k: the target mode
    :param rotation_op: the gate
    :return:
    """
    iswap(modes, times, k)
    play(rotation_op, "charge_line")
    iswap_inv(modes, times, k)


#
job = qmm.simulate(
    config,
    time_freq_sweep(prepare_f, 3, [1, 2, 3, 4], 32, 60, 4),
    simulate=SimulationConfig(10000),
)
f_population = job.result_handles.state.fetch_all()
# get the modes reachable using the e-f transition
modes_ef, exchange_times_ef = extract_modes_times(f_population)


def cz_transmon(k):
    iswap(modes_ef, exchange_times_ef, k)
    iswap(modes_ef, exchange_times_ef, k)


def cx_transmon(k):
    iswap(modes_ef, exchange_times_ef, k)
    play("X", "charge_line")
    iswap(modes_ef, exchange_times_ef, k)


def cy_transmon(k):
    iswap(modes_ef, exchange_times_ef, k)
    play("Y", "charge_line")
    iswap(modes_ef, exchange_times_ef, k)


def cz_mode(j, k):
    """

    :param j: control mode
    :param k: target mode
    :return:
    """
    iswap(modes_ge, exchange_times_ge, j)
    cz_transmon(k)
    iswap_inv(modes_ge, exchange_times_ge, j)


def cx_mode(j, k):
    """

    :param j: control mode
    :param k: target mode
    :return:
    """
    iswap(modes_ge, exchange_times_ge, j)
    cx_transmon(k)
    iswap_inv(modes_ge, exchange_times_ge, j)


def cy_mode(j, k):
    """

    :param j: control mode
    :param k: target mode
    :return:
    """
    iswap(modes_ge, exchange_times_ge, j)
    cy_transmon(k)
    iswap_inv(modes_ge, exchange_times_ge, j)


def swap(j, k):
    """
    Swap modes j and k
    :param j:
    :param k:
    :return:
    """
    iswap(modes_ge, exchange_times_ge, j)
    iswap(modes_ef, exchange_times_ef, k)
    iswap(modes_ge, exchange_times_ge, k)
    iswap(modes_ef, exchange_times_ef, k)
    iswap_inv(modes_ge, exchange_times_ge, j)


def max_entangled():
    """
    Create maximally entangled gate
    :return:
    """
    play("rotate", "charge_line")
    for k in range(n_modes - 1):
        play("pi_e_f", "charge_line")
        iswap(modes_ef, exchange_times_ef, k)
    iswap(modes_ge, exchange_times_ge, n_modes)

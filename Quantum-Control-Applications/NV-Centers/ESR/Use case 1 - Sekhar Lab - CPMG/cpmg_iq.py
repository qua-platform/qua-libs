import numpy as np
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from qm.simulate.credentials import create_credentials

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

########################
# Open quantum machine #
########################

qm = qmm.open_qm(config)

###################
# The QUA program #
###################

tau_len_min = 2e3 // 4
tau_len_max = 100e3 // 4
tau_array = np.round(np.exp(np.linspace(np.log(tau_len_min), np.log(tau_len_max), 10)))

tau_array_int = [i for i in range(len(tau_array))]

for i in range(len(tau_array)):
    tau_array_int[i] = int(tau_array[i])  # already in clock cycles

num_taus = len(tau_array)
latestecho = 10e6 // 4
numecho_array = [i for i in range(num_taus)]

for i_tau in range(0, num_taus):
    numecho_array[i_tau] = int(np.floor(latestecho / (2 * tau_array[i_tau])))  # already in clock cycles

totalnumechoes = np.sum(numecho_array)

N_shots = 10

with program() as hello_qua:

    n = declare(int)
    n_st = declare_stream()

    I = declare(fixed)
    Q = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    tau_st = declare_stream()
    i_echo_st = declare_stream()

    pls1_len = declare(int, value=240 // 4)  # 250 cycles = 1 us
    pls2_len = declare(int, value=480 // 4)
    switch_len = declare(int, value=480 // 4)
    rcvr_len = declare(int, value=1200 // 4)

    tau_array_q = declare(int, value=tau_array_int)
    tau = declare(int)
    i_tau = declare(int)

    numecho_array_q = declare(int, value=numecho_array)
    i_echo = declare(int)

    foo = declare(int)

    with for_(n, 0, n < N_shots, n + 1):

        with for_(i_tau, 0, i_tau < num_taus, i_tau + 1):

            reset_phase("ensemble")
            reset_phase("resonator")
            reset_frame("ensemble")

            with for_(foo, 0, foo < 100, foo + 1):
                play("initialization", "green_laser", duration=int(10e4 // 4))

            align()

            assign(tau, tau_array_q[i_tau])
            play("activate", "switch_1", duration=pls1_len)
            play("activate", "switch_2", duration=pls1_len)
            play("const", "ensemble", duration=pls1_len)
            wait(tau - Cast.mul_int_by_fixed(pls1_len, 1.5) - 26, "ensemble")
            frame_rotation_2pi(0.25, "ensemble")

            align()

            with for_(i_echo, 0, i_echo < numecho_array_q[i_tau], i_echo + 1):
                play("activate", "switch_1", duration=pls2_len)
                play("activate", "switch_2", duration=pls2_len)
                play("const", "ensemble", duration=pls2_len)

                align("resonator", "ensemble", "switch_1", "switch_2", "switch_receiver")

                wait(tau - pls1_len - Cast.mul_int_by_fixed(rcvr_len, 0.5) - 15, "switch_receiver")

                play("activate", "switch_receiver", duration=rcvr_len)  # 250 cycles = 1 us
                wait(tau - pls1_len - Cast.mul_int_by_fixed(readout_len, 0.125) - 5, "resonator")

                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st)
                save(Q, Q_st)
                save(tau, tau_st)
                save(i_echo, i_echo_st)

                wait(tau - pls1_len - Cast.mul_int_by_fixed(rcvr_len, 0.5) - 41, "switch_receiver", "resonator")

        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        I_st.buffer(totalnumechoes).average().save("I")
        Q_st.buffer(totalnumechoes).average().save("Q")
        tau_st.buffer(totalnumechoes).average().save("tau")
        i_echo_st.buffer(totalnumechoes).average().save("i_echo")

#######################
# Simulate or execute #
#######################

simulate = False

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=1000,
        include_analog_waveforms=True,
        include_digital_waveforms=True,
        simulation_interface=LoopbackInterface(([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]), latency=180),
    )
    job = qmm.simulate(config, hello_qua, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

    analog_wf = job.simulated_analog_waveforms()
    digital_wf = job.simulated_digital_waveforms()

else:
    job = qm.execute(hello_qua)  # execute QUA program

    res_hand = job.result_handles

    iteration_handle = res_hand.get("iteration")
    iteration_handle.wait_for_values(1)
    i_echo_handle = res_hand.get("i_echo")
    i_echo_handle.wait_for_values(1)
    tau_handle = res_hand.get("tau")
    tau_handle.wait_for_values(1)

    I_handle = res_hand.get("I")
    I_handle.wait_for_values(1)
    Q_handle = res_hand.get("Q")
    Q_handle.wait_for_values(1)

    plt.figure()

    while res_hand.is_processing():
        try:
            iteration = iteration_handle.fetch_all()
            I = I_handle.fetch_all()
            Q = Q_handle.fetch_all()
            i_echo = i_echo_handle.fetch_all()
            tau = tau_handle.fetch_all()

            plt.plot(i_echo, I, "o")
            plt.title(f"iteration: {iteration}")
            plt.pause(0.2)

        except Exception as e:
            pass

    iteration = iteration_handle.fetch_all()
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    i_echo = i_echo_handle.fetch_all()
    tau = tau_handle.fetch_all()

    Z = I + 1j * Q

    plt.figure()
    # plt.plot(i_echo, np.abs(Z), 'o')

    for i_tau in range(num_taus):
        # Find indices of echos corresponding to this tau value
        inds = np.where(tau == tau_array[i_tau])
        # create reduced arrays using these indices
        # 1 needs to be added for echo array so it starts with 1
        i_echo_red = i_echo[inds] + 1
        tau_red = tau[inds]
        Z_red = Z[inds]
        # the time of the echo is 2*i_echo*tau. A factor of 4 converts from clock cycles to ns. A factor of 1e-3 converts to us.
        t = 8e-3 * i_echo_red * tau_red
        plt.plot(t, abs(Z_red), label=np.round(tau_array[i_tau] * 4e-3, 2))

    ax = plt.gca()
    plt.xlabel("Time of echo (us)")
    plt.ylabel("Magnitude of echo (a.u.)")
    plt.title("CPMG for NV center at B = 13 mT, f = 2.5 GHz")
    ax.legend(title="tau (us)")
    ax.set_xscale("log")
    plt.show()

    np.savez("Data_cpmg_iq.npz", I, Q, i_echo, tau)

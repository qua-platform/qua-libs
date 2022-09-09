"""
A script that measures the echo amplitude for a wide range of delays between pi pulses in a CPMG pulse sequence
"""
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from macros import get_c2c_time

###################
# The QUA program #
###################

pi_len = 320 // 4  # Calibrated pi-pulse
pi_half_len = 160 // 4  # Calibrated pi/2 pulse

tau_min = 1000 // 4  # tau is half the time between pi pulses
tau_max = 50 * u.us // 4  # tau is half the time between pi pulses
tau_array = np.round(np.logspace(np.log10(tau_min), np.log10(tau_max), 30))  # outputs results in float

tau_array_int = tau_array.astype(int).tolist()
num_taus = len(tau_array)

# Here we want to keep to total duration of the sequence constant so we need to adapt the number of echo pulses
# according to the waiting time 2*tau between two successive (pi-readout) segments.
latest_echo = 4 * u.ms // 4  # this is the maximum amount of time I will run CPMG

num_echo_array = (np.floor(latest_echo / (2 * tau_array))).astype(int).tolist()
total_num_echoes = np.sum(num_echo_array)  # total number of echoes played in the whole sequence

n_avg = 100

with program() as cpmg:

    n = declare(int)
    n_st = declare_stream()

    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    tau = declare(int)  # Delay between pi-pulses
    tau_array_q = declare(int, value=tau_array_int)  # Array containing the logarithmic taus
    i_tau = declare(int)  # Index to loop over the logarithmic taus
    tau_st = declare_stream()

    num_echo_array_q = declare(int, value=num_echo_array)  # Array containing the numbers of echo pulses
    i_echo = declare(int)  # index of echo pulses
    i_echo_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(i_tau, 0, i_tau < num_taus, i_tau + 1):

            # initialization
            play("initialization", "green_laser")

            align()

            # we reset_phase the 'ensemble' to be able to collect signals with 'resonator'
            # with the same phase every run. Thus, when the analog traces are averaged they
            # do not wash out. Furthermore, because the control signal is associated with
            # 'ensemble' and demodulated in 'resonator', we reset the phase of the 'resonator'
            # as well so that there is no random phase in the demodulation stage
            reset_phase("ensemble")
            reset_phase("resonator")
            reset_frame("ensemble")

            assign(tau, tau_array_q[i_tau])
            play("const", "ensemble", duration=pi_half_len)
            # we delay the switches because `duration` for digital pulses is faster than for analog
            # We use the simulator to make the adjustments and find `8`
            wait(8, "switch_1", "switch_2")
            play("activate", "switch_1", duration=pi_half_len)
            play("activate", "switch_2", duration=pi_half_len)

            frame_rotation_2pi(-0.5, "ensemble")

            # Here we want to wait `tau` ns between the center of the pi/2 pulse and pi pulse.
            # We thus need to remove half pi/2 and half pi from tau --> 1.5*pi/2.
            # the additional 5 cycles are added to compensate for the overhead of
            # real time calculations.
            # We use the simulator to make the adjustments and find `5`
            wait(tau - pi_half_len * 1.5 - 5, "ensemble")

            align()
            # Now we play a set of pi-pulse + measure statements
            with for_(i_echo, 0, i_echo < num_echo_array_q[i_tau], i_echo + 1):
                play("activate", "switch_1", duration=pi_len)
                play("activate", "switch_2", duration=pi_len)
                play("const", "ensemble", duration=pi_len)

                align()  # global align

                # Here we want to wait `tau` ns between the center of the pi pulse and readout pulse.
                # We thus need to remove half pi (=pi/2=pulse1_len) and half readout (readout/2/4 because it is in ns) from tau.
                wait(tau - (pi_len + readout_len // 4) // 2, "switch_receiver", "resonator")

                play("activate_resonator", "switch_receiver")
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

                # Here we want to wait `tau` ns between the center of the pi pulse and readout pulse.
                # We thus need to remove half pi (=pi/2=pulse1_len) and half readout (readout/2/4 because it is in ns) from tau.
                # the additional 56 cycles are added to compensate for the overhead of real time calculations.
                # We use the simulator to make the adjustments and find `56`.
                wait(tau - (pi_len + readout_len // 4) // 2 - 56, "switch_receiver", "resonator")

        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        I_st.buffer(total_num_echoes).average().save("I")
        Q_st.buffer(total_num_echoes).average().save("Q")
        tau_st.buffer(total_num_echoes).average().save("tau")
        i_echo_st.buffer(total_num_echoes).average().save("i_echo")


################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager(qop_ip)

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=4000,
        include_analog_waveforms=True,
        simulation_interface=LoopbackInterface(([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]), latency=180),
    )
    # the simulation is uses to assert the pulse positions and to make final adjustments
    # to the QUA program
    job = qmm.simulate(config, cpmg, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

    # The lines of code below allow you to retrieve information from the simulated waveform to assert
    # their position in time.
    # ver_t1: center-to-center time between first two pulses arriving to 'ensemble'
    ver_t1 = get_c2c_time(job, ("ensemble", 0), ("ensemble", 2))
    print("center to center time between 1st and 2nd pulse", ver_t1)
    # ver_t2: center-to-center time between the readout window to the second pulse arriving to 'ensemble'
    ver_t2 = get_c2c_time(job, ("ensemble", 2), ("resonator", 0))
    print("center to center time between readout and 2nd pulse", ver_t2)
    # ver_t3: center-to-center time between the thrid pulse arriving to 'ensemble' and the last readout window
    ver_t3 = -get_c2c_time(job, ("ensemble", 4), ("resonator", 0))
    print("center to center time between 3rd and readout pulse", ver_t3)
    # ver_t4: center-to-center time between first two pulses arriving to 'ensemble'
    ver_t4 = -get_c2c_time(job, ("ensemble", 4), ("ensemble", 2))
    print("center to center time between 3rd and 2nd pulse", ver_t4)

else:
    qm = qmm.open_qm(config)

    job = qm.execute(cpmg)  # execute QUA program

    # Get results from QUA program
    results = fetching_tool(job, data_list=["i_echo", "tau", "I", "Q", "iteration"], mode="live")

    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        i_echo, tau, I, Q, iteration = results.fetch_all()
        # Display progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(i_echo, I, ".", label="I")
        plt.plot(i_echo, Q, ".", label="Q")
        plt.xlabel("Number of echoes")
        plt.ylabel("Echo magnitude I & Q [a. u.]")
        plt.legend()
        plt.tight_layout()
        plt.title(f"iteration: {iteration}")
        plt.pause(0.2)

"""
A script that measures the echo amplitude for a wide range of delays between pi pulses in a CPMG pulse sequence
"""
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from qm.simulate.credentials import create_credentials

###################
# The QUA program #
###################

tau_min = 2000 // 4  # tau is half the time between pi pulses
tau_max = 5e4 // 4  # tau is half the time between pi pulses
tau_array = np.round(np.logspace(np.log10(tau_min), np.log10(tau_max), 30))  # outputs results in float

tau_array_int = tau_array.astype(int).tolist()

num_taus = len(tau_array)

latest_echo = 50e6 // 4  # this is the maximum amount of time I will run CPMG

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

    tau_st = declare_stream()
    i_echo_st = declare_stream()

    tau_array_q = declare(int, value=tau_array_int)
    tau = declare(int)
    i_tau = declare(int)

    num_echo_array_q = declare(int, value=num_echo_array)
    i_echo = declare(int)

    echo = declare_stream(adc_trace=True)

    pulse1_len = declare(int, value=240 // 4)  # 240 and 480 ns are for now arbitrary but these should be
    pulse2_len = declare(int, value=480 // 4)  # the calibrated pi/2 and pi pulses

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

            # we delay the switches because `duration` in digital pulses
            # takes less cycles to compute than in analog ones
            wait(11, "switch_1", "switch_2")
            play("activate", "switch_1", duration=pulse1_len)
            play("activate", "switch_2", duration=pulse1_len)
            play("const", "ensemble", duration=pulse1_len)

            frame_rotation_2pi(-0.5, "ensemble")

            # the additional 27 cycles are added to compensate for the overhead of
            # real time calculations such as Cast.mul_int_by_fixed(readout_len, 0.125)
            # We use the simulator to make the adjustments and find `27`
            wait(tau - Cast.mul_int_by_fixed(pulse1_len, 1.5) - 27, "ensemble")

            align()

            with for_(i_echo, 0, i_echo < num_echo_array_q[i_tau], i_echo + 1):
                play("activate", "switch_1", duration=pulse2_len)
                play("activate", "switch_2", duration=pulse2_len)
                play("const", "ensemble", duration=pulse2_len)

                align()  # global align

                # the additional 5 cycles are added to compensate for the overhead of
                # real time calculations such as Cast.mul_int_by_fixed(readout_len, 0.125)
                # We use the simulator to make the adjustments and find `5`
                wait(tau - pulse1_len - Cast.mul_int_by_fixed(readout_len, 0.125) - 5, "switch_receiver", "resonator")

                play("activate_resonator", "switch_receiver")
                measure(
                    "readout",
                    "resonator",
                    echo,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st)
                save(Q, Q_st)
                save(tau, tau_st)
                save(i_echo, i_echo_st)

                # the additional 66 cycles are added to compensate for the overhead of
                # real time calculations such as Cast.mul_int_by_fixed(readout_len, 0.125)
                # We use the simulator to make the adjustments and find `66`
                wait(tau - pulse1_len - Cast.mul_int_by_fixed(readout_len, 0.125) - 66, "switch_receiver", "resonator")

        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        echo.input1().buffer(total_num_echoes).average().save("echo1")
        echo.input2().buffer(total_num_echoes).average().save("echo2")
        I_st.buffer(total_num_echoes).average().save("I")
        Q_st.buffer(total_num_echoes).average().save("Q")
        tau_st.buffer(total_num_echoes).average().save("tau")
        i_echo_st.buffer(total_num_echoes).average().save("i_echo")


################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

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

    analog_wf = job.simulated_analog_waveforms()

    # ver_t1: center-to-center time between first two pulses arriving to 'ensemble'
    ver_t1 = (
        analog_wf["elements"]["ensemble"][2]["timestamp"] + (analog_wf["elements"]["ensemble"][2]["duration"] / 2)
    ) - (analog_wf["elements"]["ensemble"][0]["timestamp"] + (analog_wf["elements"]["ensemble"][0]["duration"] / 2))
    # ver_t2: center-to-center time between the readout window to the second pulse arriving to 'ensemble'
    ver_t2 = (
        analog_wf["elements"]["resonator"][0]["timestamp"] + (analog_wf["elements"]["resonator"][0]["duration"] / 2)
    ) - (analog_wf["elements"]["ensemble"][2]["timestamp"] + (analog_wf["elements"]["ensemble"][2]["duration"] / 2))
    # ver_t3: center-to-center time between the thrid pulse arriving to 'ensemble' and the last readout window
    ver_t3 = (
        analog_wf["elements"]["resonator"][0]["timestamp"] + (analog_wf["elements"]["resonator"][0]["duration"] / 2)
    ) - (analog_wf["elements"]["ensemble"][4]["timestamp"] + (analog_wf["elements"]["ensemble"][4]["duration"] / 2))
    # ver_t4: center-to-center time between first two pulses arriving to 'ensemble'
    ver_t4 = (
        analog_wf["elements"]["ensemble"][2]["timestamp"] + (analog_wf["elements"]["ensemble"][2]["duration"] / 2)
    ) - (analog_wf["elements"]["ensemble"][4]["timestamp"] + (analog_wf["elements"]["ensemble"][4]["duration"] / 2))

    print("center to center time between 1st and 2nd pulse", ver_t1)
    print("center to center time between readout and 2nd pulse", ver_t2)
    print("center to center time between 3rd and readout pulse", -1 * ver_t3)
    print("center to center time between 3rd and 2nd pulse", -1 * ver_t4)

else:
    qm = qmm.open_qm(config)

    job = qm.execute(cpmg)  # execute QUA program

    res_handle = job.result_handles

    i_echo_handle = res_handle.get("i_echo")
    i_echo_handle.wait_for_values(1)
    tau_handle = res_handle.get("tau")
    tau_handle.wait_for_values(1)

    I_handle = res_handle.get("I")
    I_handle.wait_for_values(1)
    Q_handle = res_handle.get("Q")
    Q_handle.wait_for_values(1)
    echo1_handle = res_handle.get("echo1")
    echo1_handle.wait_for_values(1)
    echo2_handle = res_handle.get("echo2")
    echo2_handle.wait_for_values(1)
    iteration_handle = res_handle.get("iteration")
    iteration_handle.wait_for_values(1)
    next_percent = 0.1  # First time print 10%

    def on_close(event):
        event.canvas.stop_event_loop()
        job.halt()

    f = plt.figure()
    f.canvas.mpl_connect("close_event", on_close)
    print("Progress =", end=" ")

    while res_handle.is_processing():
        try:
            I = I_handle.fetch_all()
            Q = Q_handle.fetch_all()
            iteration = iteration_handle.fetch_all()
            i_echo = i_echo_handle.fetch_all()
            tau = tau_handle.fetch_all()
            if iteration / n_avg > next_percent:
                percent = 10 * round(iteration / n_avg * 10)  # Round to nearest 10%
                print(f"{percent}%", end=" ")
                next_percent = percent / 100 + 0.1  # Print every 10%

            plt.plot(i_echo, I, "o")
            plt.title(f"iteration: {iteration}")
            plt.pause(0.2)
            plt.clf()

        except Exception as e:
            pass

    plt.cla()
    iteration = iteration_handle.fetch_all()
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    echo1 = echo1_handle.fetch_all()
    echo2 = echo1_handle.fetch_all()
    i_echo = i_echo_handle.fetch_all()
    tau = tau_handle.fetch_all()
    print(f"{round(iteration/n_avg * 100)}%")

    plt.plot(i_echo, I, "o", label="I")
    plt.plot(i_echo, Q, "o", label="Q")
    plt.xlabel("Number of echoes [ns]")
    plt.ylabel("Echo magnitude I & Q [a. u.]")
    plt.legend()
    plt.tight_layout()

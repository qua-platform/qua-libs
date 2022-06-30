"""
Measures T1 either from |0> or |1> to the thermal state, i.e., prior to initialization
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

pi_len = 320 // 4
pi_half_len = 160 // 4

wait_min = 1500 // 4
wait_max = 20000 // 4
dwait = 1000 // 4

waits = np.arange(wait_min, wait_max + 0.1, dwait)

cooldown_time = int(10e6 // 4)

n_avg = 100

pulse_delay_p = int(2000 // 4 - pi_half_len * 0.5 - pi_len * 0.5)
readout_delay_p = int(2000 // 4 - pi_len * 0.5 - readout_len * 0.125 - 5)

with program() as T1:

    n = declare(int)
    wait_len = declare(int)
    pulse_delay = declare(int, value=pulse_delay_p)
    readout_delay = declare(int, value=readout_delay_p)

    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):

        with for_(wait_len, wait_min, wait_len <= wait_max, wait_len + dwait):

            # initialization
            play("initialization", "green_laser")

            align()

            wait(wait_len)  # delay for T1 measurement

            # we reset_phase the 'ensemble' to be able to collect signals with 'resonator'
            # with the same phase every run. Thus, when the analog traces are averaged they
            # do not wash out. Furthermore, because the control signal is associated with
            # 'ensemble' and demodulated in 'resonator', we reset the phase of the 'resonator'
            # as well so that there is no random phase in the demodulation stage
            reset_phase("ensemble")  # makes the phase of 'ensemble' pulse identical every run
            reset_phase("resonator")  # makes the phase of 'resonator' pulse identical every run
            reset_frame("ensemble")  # bring to 0 the -pi phase added to 'ensemble'

            echo = declare_stream(adc_trace=True)

            # we delay the switches because `duration` in digital pulses
            # takes less cycles to compute than in analog ones
            # We use the simulator to make the adjustments and find `8`
            wait(8, "switch_1", "switch_2")
            play("activate", "switch_1", duration=pi_half_len)
            play("activate", "switch_2", duration=pi_half_len)
            play("const", "ensemble", duration=pi_half_len)

            wait(pulse_delay, "ensemble", "switch_1", "switch_2")

            frame_rotation_2pi(-0.5, "ensemble")
            play("activate", "switch_1", duration=pi_len)
            play("activate", "switch_2", duration=pi_len)
            play("const", "ensemble", duration=pi_len)

            align()  # global align

            wait(readout_delay, "resonator", "switch_receiver")

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

    with stream_processing():
        echo.input1().buffer(len(waits)).average().save("echo1")
        echo.input2().buffer(len(waits)).average().save("echo2")
        I_st.buffer(len(waits)).average().save("I")
        Q_st.buffer(len(waits)).average().save("Q")

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
        duration=2000,
        include_analog_waveforms=True,
        simulation_interface=LoopbackInterface(([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]), latency=180),
    )
    # the simulation is uses to assert the pulse positions and to make final adjustments
    # to the QUA program
    job = qmm.simulate(config, T1, simulate_config)  # do simulation with qmm
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

    print("center to center time between 1st and 2nd pulse", ver_t1)
    print("center to center time between readout and 2nd pulse", ver_t2)

else:
    qm = qmm.open_qm(config)

    job = qm.execute(T1)  # execute QUA program

    res_handle = job.result_handles

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
            if iteration / n_avg > next_percent:
                percent = 10 * round(iteration / n_avg * 10)  # Round to nearest 10%
                print(f"{percent}%", end=" ")
                next_percent = percent / 100 + 0.1  # Print every 10%

            plt.plot(waits * 4, I)
            plt.plot(waits * 4, Q)
            plt.title(f"iteration: {iteration}")
            plt.pause(0.2)
            plt.clf()

        except Exception as e:
            pass

    plt.cla()
    echo1 = echo1_handle.fetch_all()
    echo2 = echo1_handle.fetch_all()
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    iteration = iteration_handle.fetch_all()
    print(f"{round(iteration/n_avg * 100)}%")

    plt.plot(waits * 4, I, label="I")
    plt.plot(waits * 4, Q, label="Q")
    plt.xlabel("Decay time [ns]")
    plt.ylabel("Echo magnitude I & Q [a. u.]")
    plt.legend()
    plt.tight_layout()

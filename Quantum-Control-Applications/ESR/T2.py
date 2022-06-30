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

delay_min = 1500 // 4
delay_max = 20000 // 4
ddelay = 1000 // 4

delays = np.arange(delay_min, delay_max + 0.1, ddelay)

cooldown_time = int(10e6 // 4)

n_avg = 100

with program() as T2:

    n = declare(int)
    n_st = declare_stream()
    delay_len = declare(int)
    pulse_delay = declare(int)
    readout_delay = declare(int)

    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):

        with for_(delay_len, delay_min, delay_len <= delay_max, delay_len + ddelay):

            # initialization
            play('initialization', 'green_laser')

            align()

            # we reset_phase the 'ensemble' to be able to collect signals with 'resonator'
            # with the same phase every run. Thus, when the analog traces are averaged they
            # do not wash out. Furthermore, because the control signal is associated with
            # 'ensemble' and demodulated in 'resonator', we reset the phase of the 'resonator'
            # as well so that there is no random phase in the demodulation stage
            reset_phase("ensemble")
            reset_phase("resonator")
            reset_frame("ensemble")

            echo = declare_stream(adc_trace=True)

            assign(pulse_delay, delay_len - pi_half_len * 0.5 - pi_len * 0.5)
            assign(readout_delay, delay_len - pi_len * 0.5 - readout_len * 0.125 - 5)

            # we delay the switches because `duration` in digital pulses
            # takes less cycles to compute than in analog ones
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
        save(n, n_st)

    with stream_processing():
        echo.input1().buffer(len(delays)).average().save("echo1")
        echo.input2().buffer(len(delays)).average().save("echo2")
        I_st.buffer(len(delays)).average().save("I")
        Q_st.buffer(len(delays)).average().save("Q")
        n_st.save("iteration")

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
    job = qmm.simulate(config, T2, simulate_config)  # do simulation with qmm
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

    job = qm.execute(T2)  # execute QUA program

    res_handle = job.result_handles

    I_handle = res_handle.get("I")
    I_handle.wait_for_values(1)
    Q_handle = res_handle.get("Q")
    Q_handle.wait_for_values(1)
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

            plt.plot(delays * 4, I)
            plt.plot(delays * 4, Q)
            plt.title(f"iteration: {iteration}")
            plt.pause(0.2)
            plt.clf()

        except Exception as e:
            pass

    plt.cla()
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    iteration = iteration_handle.fetch_all()
    print(f"{round(iteration/n_avg * 100)}%")

    plt.plot(delays * 4, I)
    plt.plot(delays * 4, Q)

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

pulse1_min = 40 // 4
pulse1_max = 400 // 4
dpulse1 = 40 // 4

pulses1 = np.arange(pulse1_min, pulse1_max + 0.1, dpulse1)

cooldown_time = int(10e6 // 4)

n_avg = 100

readout_delay_p = int(2000 // 4 - pi_len * 0.5 - readout_len * 0.125 - 5)

with program() as time_rabi:

    n = declare(int)
    n_st = declare_stream()
    pulse1_len = declare(int)
    pulse_delay = declare(int)
    readout_delay = declare(int, value=readout_delay_p)

    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    echo = declare_stream(adc_trace=True)

    with for_(n, 0, n < n_avg, n + 1):

        with for_(pulse1_len, pulse1_min, pulse1_len <= pulse1_max, pulse1_len + dpulse1):

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

            assign(pulse_delay, 2000 // 4 - Cast.mul_int_by_fixed(pulse1_len, 0.5) - pi_len*0.5 - 4)

            # we delay the switches because `duration` in digital pulses
            # takes less cycles to compute than in analog ones
            wait(11, "switch_1", "switch_2")
            play("activate", "switch_1", duration=pulse1_len)
            play("activate", "switch_2", duration=pulse1_len)
            play("const", "ensemble", duration=pulse1_len)

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
        echo.input1().buffer(len(pulses1)).average().save("echo1")
        echo.input2().buffer(len(pulses1)).average().save("echo2")
        I_st.buffer(len(pulses1)).average().save("I")
        Q_st.buffer(len(pulses1)).average().save("Q")
        n_st.save("iteration")


################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

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
    job = qmm.simulate(config, time_rabi, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

    analog_wf = job.simulated_analog_waveforms()

    ver_t1 = (
        analog_wf["elements"]["ensemble"][2]["timestamp"] + (analog_wf["elements"]["ensemble"][2]["duration"] / 2)
    ) - (analog_wf["elements"]["ensemble"][0]["timestamp"] + (analog_wf["elements"]["ensemble"][0]["duration"] / 2))
    ver_t3 = (
        analog_wf["elements"]["resonator"][0]["timestamp"] + (analog_wf["elements"]["resonator"][0]["duration"] / 2)
    ) - (analog_wf["elements"]["ensemble"][2]["timestamp"] + (analog_wf["elements"]["ensemble"][2]["duration"] / 2))

    print("center to center time between 1st and 2nd pulse", ver_t1)
    print("center to center time between readout and 2nd pulse", ver_t3)

else:
    qm = qmm.open_qm(config)

    job = qm.execute(time_rabi)  # execute QUA program

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

            plt.plot(pulses1 * 4, I)
            plt.plot(pulses1 * 4, Q)
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

    plt.plot(pulses1 * 4, I)
    plt.plot(pulses1 * 4, Q)

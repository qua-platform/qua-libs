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

pulse1_len = 400 // 4
pulse2_len = pulse1_len * 2
pulse_delay = 2000 // 4 - pulse1_len / 2 - pulse2_len / 2
readout_delay = 2000 // 4 - pulse2_len / 2 - readout_len // 8 - 8

cooldown_time = int(10e6 // 4)

n_avg = 100

with program() as signal_test:

    n = declare(int)
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):

        # wait(cooldown_time)

        reset_phase("ensemble")
        reset_phase("resonator")
        reset_frame("ensemble")

        echo = declare_stream(adc_trace=True)

        wait(8, "switch_1", "switch_2")
        play("activate", "switch_1", duration=pulse1_len)
        play("activate", "switch_2", duration=pulse1_len)
        play("const", "ensemble", duration=pulse1_len)

        wait(pulse_delay, "ensemble", "switch_1", "switch_2")

        frame_rotation_2pi(-0.5, "ensemble")
        play("activate", "switch_1", duration=pulse2_len)
        play("activate", "switch_2", duration=pulse2_len)
        play("const", "ensemble", duration=pulse2_len)

        align()  # global align

        wait(readout_delay, "resonator", "switch_receiver")

        play("activate_resonator", "switch_receiver")
        measure("readout", "resonator", echo)

        save(n, n_st)

    with stream_processing():
        echo.input1().average().save("echo1")
        echo.input2().average().save("echo2")
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
    job = qmm.simulate(config, signal_test, simulate_config)  # do simulation with qmm
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

    job = qm.execute(signal_test)  # execute QUA program

    res_handle = job.result_handles

    echo1_handle = res_handle.get("echo1")
    echo1_handle.wait_for_values(1)
    echo2_handle = res_handle.get("echo2")
    echo2_handle.wait_for_all_values(1)
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
            echo1 = echo1_handle.fetch_all()
            echo2 = echo2_handle.fetch_all()
            iteration = iteration_handle.fetch_all()
            if iteration / n_avg > next_percent:
                percent = 10 * round(iteration / n_avg * 10)  # Round to nearest 10%
                print(f"{percent}%", end=" ")
                next_percent = percent / 100 + 0.1  # Print every 10%

            plt.plot(echo1)
            plt.plot(echo2)
            plt.title(f"iteration: {iteration}")
            plt.pause(0.2)
            plt.clf()

        except Exception as e:
            pass

    plt.cla()
    echo1 = echo1_handle.fetch_all()
    echo2 = echo2_handle.fetch_all()
    iteration = iteration_handle.fetch_all()
    print(f"{round(iteration/n_avg * 100)}%")

    plt.plot(echo1)
    plt.plot(echo2)

"""
A script that mimics a pi/2 - pi pulse sequence but with arbitrary pulse duration.
Helps you check if signal is being generated from your setup
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

pulse1_len = 400 // 4  # Pi/2 pulse
pulse2_len = pulse1_len * 2  # Pi pulse

cooldown_time = 10 * u.ms // 4  # Resonator or qubit relaxation time
safe_delay = 2 * u.us // 4  # Delay to safely avoid sending pulses during measurement windows

# Center to center time between first and second pulse
pulse_delay = safe_delay - (pulse1_len + pulse2_len) // 2
# Center to center time betwen second pulse and readout
readout_delay = safe_delay - (pulse2_len + readout_len // 4) // 2

n_avg = 100

with program() as signal_test:

    n = declare(int)
    n_st = declare_stream()
    echo = declare_stream(adc_trace=True)

    with for_(n, 0, n < n_avg, n + 1):

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

        # Play 1st pulse (pi/2)
        play("const", "ensemble", duration=pulse1_len)
        # we delay the switches because `duration` for digital pulses is faster than for analog
        # We use the simulator to make the adjustments and find `8`
        wait(8, "switch_1", "switch_2")
        play("activate", "switch_1", duration=pulse1_len)
        play("activate", "switch_2", duration=pulse1_len)
        # Wait some time corresponding to the echo time which also avoids sending pulses in the measurement window
        wait(pulse_delay, "ensemble", "switch_1", "switch_2")

        # Play 2nd pulse (pi) along -X (phaseshift of pi) not sure why though...
        frame_rotation_2pi(-0.5, "ensemble")
        play("activate", "switch_1", duration=pulse2_len)
        play("activate", "switch_2", duration=pulse2_len)
        play("const", "ensemble", duration=pulse2_len)

        align()  # global align
        # Wait the same amount of time as earlier in order to let the spin rephase after the echo
        wait(readout_delay, "resonator", "switch_receiver")
        # Readout
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

qmm = QuantumMachinesManager(qop_ip)

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

    # The lines of code below allow you to retrieve information from the simulated waveform to assert
    # their position in time and manually estimate internal delays.
    # ver_t1: center-to-center time between first two pulses arriving to 'ensemble'
    ver_t1 = get_c2c_time(job, ("ensemble", 0), ("ensemble", 2))
    print(
        f"center to center time between 1st and 2nd pulse is {ver_t1} --> internal delay to add: {ver_t1-4*safe_delay} ns"
    )
    # ver_t2: center-to-center time between the readout window to the second pulse arriving to 'ensemble'
    ver_t2 = get_c2c_time(job, ("ensemble", 2), ("resonator", 0))
    print(
        f"center to center time between 2nd pulse and readout is {ver_t2} --> internal delay to add: {ver_t2-4*safe_delay} ns"
    )

else:
    # Open quantum machine
    qm = qmm.open_qm(config)
    # Execute QUA program
    job = qm.execute(signal_test)
    # Fetch results
    res_handle = job.result_handles
    echo1_handle = res_handle.get("echo1")
    echo1_handle.wait_for_values(1)
    echo2_handle = res_handle.get("echo2")
    echo2_handle.wait_for_all_values(1)
    iteration_handle = res_handle.get("iteration")
    iteration_handle.wait_for_values(1)
    # results = fetching_tool(job, data_list=["echo1", "echo2", "iteration"], mode="live")

    # Plot results
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while res_handle.is_processing():
        # echo1, echo2, iteration = results.fetch_all()
        echo1 = u.raw2volts(echo1_handle.fetch_all())
        echo2 = u.raw2volts(echo2_handle.fetch_all())
        iteration = iteration_handle.fetch_all()
        progress_counter(iteration, n_avg)

        plt.plot(echo1)
        plt.plot(echo2)
        plt.xlabel("Time [ns]")
        plt.ylabel("Signal amplitude [V]")
        plt.tight_layout()
        plt.pause(0.2)

    plt.cla()
    echo1 = u.raw2volts(echo1_handle.fetch_all())
    echo2 = u.raw2volts(echo2_handle.fetch_all())

    plt.plot(echo1)
    plt.plot(echo2)
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal amplitude [V]")
    plt.tight_layout()

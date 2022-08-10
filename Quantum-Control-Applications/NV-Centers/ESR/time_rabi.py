"""
Having calibrated roughly a pi pulse, this script allows you to fix the pi pulse duration and change the duration of the
first pulse to obtain Rabi oscillations throughout the sequence.
This allows measuring all the delays in the system, as well as the NV initialization duration
"""
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from macros import get_c2c_time

###################
# The QUA program #
###################
# Pi pulse duration calibrated with 'pi_pulse_calibration.py'
pi_len = 320 // 4

pulse1_min = 40 // 4
pulse1_max = 400 // 4
dpulse1 = 4 // 4
pulse1_vec = np.arange(pulse1_min, pulse1_max + 0.1, dpulse1)

cooldown_time = 10 * u.ms // 4

n_avg = 1000
# This delay is defined as the duration between the center of the pi pulse and the center of the readout pulse
readout_delay = safe_delay - (pi_len + readout_len // 4) // 2 - 5

with program() as time_rabi:

    n = declare(int)
    n_st = declare_stream()
    pulse1_len = declare(int)
    pulse_delay = declare(int)

    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(pulse1_len, pulse1_vec)):

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

            assign(pulse_delay, safe_delay - Cast.mul_int_by_fixed(pulse1_len, 0.5) - pi_len // 2 - 4)
            # Rabi pulse
            play("const", "ensemble", duration=pulse1_len)
            # we delay the switches because `duration` for digital pulses is faster than for analog
            # We use the simulator to make the adjustments and find `8`
            wait(8, "switch_1", "switch_2")
            play("activate", "switch_1", duration=pulse1_len)
            play("activate", "switch_2", duration=pulse1_len)
            # Wait some time corresponding to the echo time which also avoids sending pulses in the measurement window
            wait(pulse_delay, "ensemble", "switch_1", "switch_2")
            # Pi pulse
            frame_rotation_2pi(-0.5, "ensemble")
            play("const", "ensemble", duration=pi_len)
            play("activate", "switch_1", duration=pi_len)
            play("activate", "switch_2", duration=pi_len)

            align()  # global align
            # Wait the same amount of time as earlier in order to let the spin rephase after the echo
            wait(readout_delay, "resonator", "switch_receiver")
            # Readout
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
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(pulse1_vec)).average().save("I")
        Q_st.buffer(len(pulse1_vec)).average().save("Q")
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
    # the simulation is uses to assert the pulse positions and to make final adjustments
    # to the QUA program
    job = qmm.simulate(config, time_rabi, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

    # The lines of code below allow you to retrieve information from the simulated waveform to assert
    # their position in time.
    # ver_t1: center-to-center time between first two pulses arriving to 'ensemble'
    ver_t1 = get_c2c_time(job, ("ensemble", 0), ("ensemble", 2))
    print(
        f"center to center time between 1st and 2nd pulse is {ver_t1} --> internal delay to add: {ver_t1 - 4 * safe_delay} ns"
    )
    # ver_t2: center-to-center time between the readout window to the second pulse arriving to 'ensemble'
    ver_t2 = get_c2c_time(job, ("ensemble", 2), ("resonator", 0))
    print(
        f"center to center time between 2nd pulse and readout is {ver_t2} --> internal delay to add: {ver_t2 - 4 * safe_delay} ns"
    )

else:
    qm = qmm.open_qm(config)

    job = qm.execute(time_rabi)  # execute QUA program

    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(pulse1_vec * 4, I, label="I")
        plt.plot(pulse1_vec * 4, Q, label="Q")
        plt.title(f"iteration: {iteration}")
        plt.xlabel("pi/2 pulse length [ns]")
        plt.ylabel("Echo magnitude I & Q [a. u.]")
        plt.legend()
        plt.tight_layout()
        plt.pause(0.2)

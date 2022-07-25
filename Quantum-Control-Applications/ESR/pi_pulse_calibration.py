"""
A script that changes the duration of the pulses send to the ensemble to determine
which pulse duration maximizes the echo amplitude
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

# Pi/2 pulse duration to be scanned
pulse1_min = 20 // 4
pulse1_max = 400 // 4
dpulse1 = 40 // 4
pulse1_vec = np.arange(pulse1_min, pulse1_max + 0.1, dpulse1)

# Resonator or qubit relaxation time
cooldown_time = 10 * u.ms // 4

n_avg = 100

with program() as pi_pulse_cal:

    n = declare(int)
    pulse1_len = declare(int)
    pulse2_len = declare(int)
    pulse_delay = declare(int)
    readout_delay = declare(int)

    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    n_st = declare_stream()
    echo = declare_stream(adc_trace=True)

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

            # Set pulse2_len to 2*pulse1_len so that pulse2 is pi if pulse1 is pi/2
            assign(pulse2_len, 2 * pulse1_len)
            # Set pulse_delay to safe_delay - center_to_center - internal_delay (found with simulation)
            assign(
                pulse_delay,
                safe_delay - Cast.mul_int_by_fixed(pulse1_len + pulse2_len, 0.5) - 8,
            )
            # Set pulse_delay to safe_delay - center_to_center - internal_delay (found with simulation)
            assign(
                readout_delay,
                safe_delay - Cast.mul_int_by_fixed(pulse2_len + readout_len // 4, 0.5) - 5,
            )
            # Play first pulse along X
            play("const", "ensemble", duration=pulse1_len)
            # we delay the switches because `duration` for digital pulses is faster than for analog
            # We use the simulator to make the adjustments and find `8`
            wait(8, "switch_1", "switch_2")
            play("activate", "switch_1", duration=pulse1_len)
            play("activate", "switch_2", duration=pulse1_len)
            # Wait some time corresponding to the echo time which also avoids sending pulses in the measurement window
            wait(pulse_delay, "ensemble", "switch_1", "switch_2")
            # Play second pulse along -X
            frame_rotation_2pi(-0.5, "ensemble")
            play("activate", "switch_1", duration=pulse2_len)
            play("activate", "switch_2", duration=pulse2_len)
            play("const", "ensemble", duration=pulse2_len)

            align()  # global align
            # Wait the same amount of time as earlier in order to let the spin rephase after the echo
            wait(readout_delay, "resonator", "switch_receiver")
            # Readout
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
        echo.input1().buffer(len(pulse1_vec)).average().save("echo1")
        echo.input2().buffer(len(pulse1_vec)).average().save("echo2")
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
        duration=4000,
        include_analog_waveforms=True,
        simulation_interface=LoopbackInterface(([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]), latency=180),
    )
    job = qmm.simulate(config, pi_pulse_cal, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

    # The lines of code below allow you to retrieve information from the simulated waveform to assert
    # their position in time and manually estimate internal delays.
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

    job = qm.execute(pi_pulse_cal)  # execute QUA program

    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert I & Q to Volts
        I = u.demod2volts(I, readout_len)
        Q = u.demod2volts(Q, readout_len)
        # Display progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(pulse1_vec * 4, I, label="I")
        plt.plot(pulse1_vec * 4, Q, label="Q")
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Signal amplitude [V]")
        plt.legend()
        plt.pause(0.2)

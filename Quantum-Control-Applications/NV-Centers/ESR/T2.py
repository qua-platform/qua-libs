"""
A script that measures T2 after initialization of the ensemble
"""
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from macros import get_c2c_time
from qualang_tools.loops import from_array

###################
# The QUA program #
###################

pi_len = 320 // 4  # Calibrated pi-pulse
pi_half_len = 160 // 4  # Calibrated pi/2 pulse

delay_min = safe_delay
delay_max = 20000 // 4
ddelay = 1000 // 4
delay_vec = np.arange(delay_min, delay_max + 0.1, ddelay)

cooldown_time = 10e6 // 4

n_avg = 100

with program() as T2:

    n = declare(int)
    t_delay = declare(int)
    pulse_delay = declare(int)
    readout_delay = declare(int)

    I = declare(fixed)
    Q = declare(fixed)
    n_st = declare_stream()
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):

        with for_(*from_array(t_delay, delay_vec)):

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

            assign(pulse_delay, t_delay - (pi_half_len + pi_len) // 2)
            assign(readout_delay, t_delay - (pi_len + readout_len // 4) // 2 - 5)

            # Pi/2 pulse
            play("const", "ensemble", duration=pi_half_len)
            # we delay the switches because `duration` for digital pulses is faster than for analog
            # We use the simulator to make the adjustments and find `8`
            wait(8, "switch_1", "switch_2")
            play("activate", "switch_1", duration=pi_half_len)
            play("activate", "switch_2", duration=pi_half_len)
            # Wait some time corresponding to the echo time which also avoids sending pulses in the measurement window
            wait(pulse_delay, "ensemble", "switch_1", "switch_2")

            # Pi pulse
            frame_rotation_2pi(-0.5, "ensemble")
            play("activate", "switch_1", duration=pi_len)
            play("activate", "switch_2", duration=pi_len)
            play("const", "ensemble", duration=pi_len)

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
        I_st.buffer(len(delay_vec)).average().save("I")
        Q_st.buffer(len(delay_vec)).average().save("Q")
        n_st.save("iteration")

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager(qop_ip)

#######################
# Simulate or execute #
#######################

simulate = False

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=2000,
        include_analog_waveforms=True,
        simulation_interface=LoopbackInterface(([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]), latency=180),
    )
    # the simulation is uses to assert the pulse positions and to make final adjustments
    # to the QUA program
    job = qmm.simulate(config, T2, simulate_config)  # do simulation with qmm
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

    job = qm.execute(T2)  # execute QUA program

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
        plt.plot(delay_vec * 4, I, label="I")
        plt.plot(delay_vec * 4, Q, label="Q")
        plt.xlabel("Delay before refocusing pulse [ns]")
        plt.ylabel("Echo magnitude I & Q [a. u.]")
        plt.legend()
        plt.pause(0.2)

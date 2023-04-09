"""
Measures T1 either from |0> or |1> to the thermal state, i.e., prior to initialization
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

wait_min = 1500 // 4
wait_max = 20 * u.us // 4
dwait = 1000 // 4
wait_vec = np.arange(wait_min, wait_max + 0.1, dwait)

cooldown_time = 10 * u.ms // 4

n_avg = 100

pulse_delay = safe_delay - (pi_half_len + pi_len) // 2
readout_delay = safe_delay - (pi_len + readout_len // 4) // 2

with program() as T1:

    n = declare(int)
    t_wait = declare(int)

    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t_wait, wait_vec)):
            # initialization
            play("initialization", "green_laser")

            align()

            wait(t_wait)  # delay for T1 measurement

            # we reset_phase the 'ensemble' to be able to collect signals with 'resonator'
            # with the same phase every run. Thus, when the analog traces are averaged they
            # do not wash out. Furthermore, because the control signal is associated with
            # 'ensemble' and demodulated in 'resonator', we reset the phase of the 'resonator'
            # as well so that there is no random phase in the demodulation stage
            reset_phase("ensemble")  # makes the phase of 'ensemble' pulse identical every run
            reset_phase("resonator")  # makes the phase of 'resonator' pulse identical every run
            reset_frame("ensemble")  # bring to 0 the -pi phase added to 'ensemble'

            # Pi/2 pulse
            play("const", "ensemble", duration=pi_half_len)
            # we delay the switches because `duration` for digital pulses is faster than for analog
            # We use the simulator to make the adjustments and find `8`
            wait(8, "switch_1", "switch_2")
            play("activate", "switch_1", duration=pi_half_len)
            play("activate", "switch_2", duration=pi_half_len)
            # Wait some time corresponding to the echo time which also avoids sending pulses in the measurement window
            wait(pulse_delay, "ensemble", "switch_1", "switch_2")

            frame_rotation_2pi(-0.5, "ensemble")
            play("activate", "switch_1", duration=pi_len)
            play("activate", "switch_2", duration=pi_len)
            play("const", "ensemble", duration=pi_len)

            align()  # global align
            # Wait the same amount of time as earlier in order to let the spin rephase after the echo
            wait(readout_delay, "resonator", "switch_receiver")

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
        I_st.buffer(len(wait_vec)).average().save("I")
        Q_st.buffer(len(wait_vec)).average().save("Q")
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
    # the simulation is used to assert the pulse positions and to make final adjustments
    # to the QUA program
    job = qmm.simulate(config, T1, simulate_config)  # do simulation with qmm
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

    job = qm.execute(T1)  # execute QUA program

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
        plt.plot(wait_vec * 4, I, label="I")
        plt.plot(wait_vec * 4, Q, label="Q")
        plt.xlabel("Decay time [ns]")
        plt.ylabel("Echo magnitude I & Q [V]")
        plt.legend()
        plt.pause(0.2)

"""
        RABI-LIKE CHEVRON - using a combination of the baking tool and real-time QUA (1ns granularity for long pulses)
The goal of the script is to acquire delta-g driven coherent oscillations by sweeping the interaction time and detuning.
The QUA program is divided into three sections:
    1) step between the initialization point and the measurement point using sticky elements (long timescale).
    2) pulse the detuning to a region where delta-g dominates using non-sticky elements (short timescale).
    3) measure the state of the qubit using either RF reflectometry or dc current sensing via PSB or Elzerman readout.
A compensation pulse can be added to the long timescale sequence in order to ensure 0 DC voltage on the fast line of
the bias-tee. Alternatively one can obtain the same result by changing the offset of the slow line of the bias-tee.

In the current implementation, the qubit pulse is played using both the baking tool, that allows for playing arbitrarily
short pulses with 1ns resolution, and real-time pulse manipulation of the OPX for playing arbitrarily long pulse without
any memory issue.
Also note that the qubit pulses are played at the end of the "idle" level whose duration is fixed.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the DC offsets of the external DC voltage source.
    - Connecting the OPX to the fast line of the plunger gates.
    - Having calibrated the initialization and readout point from the charge stability map and updated the configuration.

Before proceeding to the next node:
    - Identify the pi and pi/2 pulse parameters, Rabi frequency...
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.bakery import baking
import matplotlib.pyplot as plt
from macros import RF_reflectometry_macro, DC_current_sensing_macro


###################
# The QUA program #
###################

n_avg = 100
# Pulse duration sweep in ns
durations = np.arange(0, 500, 1)
# Pulse amplitude sweep as the absolute voltage level in V
pi_levels = np.arange(0.21, 0.3, 0.01)

seq = VoltageGateSequence(config, ["P1_sticky", "P2_sticky"])
seq.add_points("initialization", level_init, duration_init)
seq.add_points("idle", level_manip, duration_manip)
seq.add_points("readout", level_readout, duration_readout)

# Bake the Rabi pulses
pi_list = []
pi_list_4ns = []
for t in range(16):  # Create the different baked sequences
    t = int(t)
    with baking(config, padding_method="left") as b:  # don't use padding to assure error if timing is incorrect
        if t == 0:
            wf1 = [0.0] * 16
            wf2 = [0.0] * 16
        else:
            wf1 = [0.25] * t
            wf2 = [0.25] * t

        # Add the baked operation to the config
        b.add_op("pi_baked", "P1", wf1)
        b.add_op("pi_baked", "P2", wf2)

        # Baked sequence
        b.wait(16 - t, "P1")  # Wait time to take gaps into account and always play right before reading out
        b.wait(16 - t, "P2")  # Wait time to take gaps into account and always play right before reading out
        b.play("pi_baked", "P1")  # Play the qubit pulse
        b.play("pi_baked", "P2")  # Play the qubit pulse
    if t < 4:
        with baking(config, padding_method="left") as b4ns:  # don't use padding to assure error if timing is incorrect
            wf1 = [0.25] * t
            wf2 = [0.25] * t

            # Add the baked operation to the config
            b4ns.add_op("pi_baked2", "P1", wf1)
            b4ns.add_op("pi_baked2", "P2", wf2)

            # Baked sequence
            b4ns.wait(32 - t, "P1")  # Wait time to take gaps into account and always play right before reading out
            b4ns.wait(32 - t, "P2")  # Wait time to take gaps into account and always play right before reading out
            b4ns.play("pi_baked2", "P1")  # Play the qubit pulse
            b4ns.play("pi_baked2", "P2")  # Play the qubit pulse

    # Append the baking object in the list to call it from the QUA program
    pi_list.append(b)
    if t < 4:
        pi_list_4ns.append(b4ns)


with program() as Rabi_prog:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    t = declare(int)  # QUA variable for the qubit pulse duration
    t_cycles = declare(int)  # QUA variable for the qubit pulse duration
    t_left_ns = declare(int)  # QUA variable for the remainder
    Vpi = declare(fixed)  # QUA variable for the qubit drive amplitude
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
        save(n, n_st)
        with for_(*from_array(Vpi, pi_levels)):
            with for_(*from_array(t, durations)):
                with strict_timing_():
                    # Navigate through the charge stability map
                    seq.add_step(voltage_point_name="initialization")
                    seq.add_step(voltage_point_name="readout")
                    seq.add_compensation_pulse(duration=duration_compensation_pulse)

                # Short qubit pulse: baking only
                with if_(t < 16):
                    # switch case to select the baked waveform corresponding to the burst duration
                    with switch_(t, unsafe=True):
                        for ii in range(16):
                            with case_(ii):
                                # Drive the singlet-triplet qubit using an exchange pulse at the end of the manipulation step
                                wait(duration_init * u.ns - 4 - 9, "P1", "P2")
                                pi_list[ii].run(
                                    amp_array=[("P1", (Vpi - level_init[0]) * 4), ("P2", (-Vpi - level_init[1]) * 4)]
                                )

                # Long qubit pulse: baking and play combined
                with else_():
                    assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                    assign(t_left_ns, t - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                    # switch case to select the baked waveform corresponding to the burst duration
                    with switch_(t_left_ns, unsafe=True):
                        for ii in range(4):
                            with case_(ii):
                                # Drive the singlet-triplet qubit using an exchange pulse at the end of the manipulation step
                                wait(duration_init * u.ns - t_cycles - 4 - 29, "P1", "P2")
                                pi_list_4ns[ii].run(
                                    amp_array=[("P1", (Vpi - level_init[0]) * 4), ("P2", (-Vpi - level_init[1]) * 4)]
                                )
                                play("step" * amp((Vpi - level_init[0]) * 4), "P1", duration=t_cycles)
                                play("step" * amp((-Vpi - level_init[1]) * 4), "P2", duration=t_cycles)

                # Measure the dot right after the qubit manipulation
                wait(duration_init * u.ns, "tank_circuit", "TIA")
                I, Q, I_st, Q_st = RF_reflectometry_macro()
                dc_signal, dc_signal_st = DC_current_sensing_macro()

                # Ramp the background voltage to zero to avoid propagating floating point errors
                seq.ramp_to_zero()
    # Stream processing section used to process the data before saving it.
    with stream_processing():
        n_st.save("iteration")
        # Cast the data into a 2D matrix and performs a global averaging of the received 2D matrices together.
        # RF reflectometry
        I_st.buffer(len(durations)).buffer(len(pi_levels)).average().save("I")
        Q_st.buffer(len(durations)).buffer(len(pi_levels)).average().save("Q")
        # DC current sensing
        dc_signal_st.buffer(len(durations)).buffer(len(pi_levels)).average().save("dc_signal")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, Rabi_prog, simulation_config)
    # Plot the simulated samples
    plt.figure()
    plt.subplot(211)
    job.get_simulated_samples().con1.plot()
    plt.axhline(level_init[0], color="k", linestyle="--")
    plt.axhline(level_manip[0], color="k", linestyle="--")
    plt.axhline(level_readout[0], color="k", linestyle="--")
    plt.axhline(level_init[1], color="k", linestyle="--")
    plt.axhline(level_manip[1], color="k", linestyle="--")
    plt.axhline(level_readout[1], color="k", linestyle="--")
    plt.axhline(pi_amps[0], color="k", linestyle="--")
    plt.axhline(pi_amps[1], color="k", linestyle="--")
    plt.yticks(
        [
            pi_amps[1],
            level_readout[1],
            level_manip[1],
            level_init[1],
            0.0,
            level_init[0],
            level_manip[0],
            level_readout[0],
            pi_amps[0],
        ],
        ["pi", "readout", "manip", "init", "0", "init", "manip", "readout", "pi"],
    )
    plt.legend("")
    from macros import get_filtered_voltage

    plt.subplot(212)
    get_filtered_voltage(job.get_simulated_samples().con1.analog["1"], 1e-9, bias_tee_cut_off_frequency, True)

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(Rabi_prog)
    # Get results from QUA program and initialize live plotting
    results = fetching_tool(job, data_list=["I", "Q", "dc_signal", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch the data from the last OPX run corresponding to the current slow axis iteration
        I, Q, DC_signal, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, reflectometry_readout_length, single_demod=True)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        DC_signal = u.demod2volts(DC_signal, readout_len, single_demod=True)
        # Progress bar
        progress_counter(iteration, n_avg)
        # Plot data
        plt.subplot(121)
        plt.cla()
        plt.title(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        plt.pcolor(durations, pi_levels, R)
        plt.xlabel("Qubit pulse duration [ns]")
        plt.ylabel("Vpi [V]")
        plt.subplot(122)
        plt.cla()
        plt.title("Phase [rad]")
        plt.pcolor(durations, pi_levels, phase)
        plt.xlabel("Qubit pulse duration [ns]")
        plt.ylabel("Vpi [V]")
        plt.tight_layout()
        plt.pause(0.1)

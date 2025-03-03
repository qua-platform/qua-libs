"""
        RABI CHEVRON - using a combination of the baking tool and real-time QUA (1ns granularity for long pulses)
The goal of the script is to acquire the Rabi oscillations by EDSR pulse frequency and duration.
The QUA program is divided into three sections:
    1) step between the initialization point and the measurement point using sticky elements (long timescale).
    2) send the MW pulse to drive the EDSR transition (short timescale).
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
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100
# Pulse duration sweep in ns
durations = np.arange(0, 500, 1)
# Pulse frequency sweep in Hz
frequencies = np.arange(-100 * u.MHz, 100 * u.MHz, 100 * u.kHz)
# Delay in ns before stepping to the readout point after playing the qubit pulse - must be a multiple of 4ns and >= 16ns
delay_before_readout = 16

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
            wf_I = [0.0] * 16
            wf_Q = [0.0] * 16  # Otherwise the baked pulse will be empty
        else:
            wf_I = [pi_amp] * t
            wf_Q = [0.0] * t  # The baked waveforms (only the "I" quadrature)

        # Add the baked operation to the config
        b.add_op("pi_baked", "qubit", [wf_I, wf_Q])
        # Baked sequence
        b.wait(16 - t, "qubit")  # Wait time to take gaps into account and always play right before reading out
        b.play("pi_baked", "qubit")  # Play the qubit pulse

    if t < 4:
        with baking(config, padding_method="left") as b4ns:  # don't use padding to assure error if timing is incorrect
            wf_I = [pi_amp] * t
            wf_Q = [0.0] * t  # The baked waveforms (only the "I" quadrature)

            # Add the baked operation to the config
            b4ns.add_op("pi_baked2", "qubit", [wf_I, wf_Q])
            # Baked sequence
            b4ns.wait(32 - t, "qubit")  # Wait time to take gaps into account and always play right before reading out
            b4ns.play("pi_baked2", "qubit")  # Play the qubit pulse
    # Append the baking object in the list to call it from the QUA program
    pi_list.append(b)
    pi_list_4ns.append(b4ns)


# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "IF_frequencies": frequencies,
    "durations": durations,
    "config": config,
}

###################
# The QUA program #
###################
with program() as Rabi_prog:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    t = declare(int)  # QUA variable for the qubit pulse duration
    t_cycles = declare(int)  # QUA variable for the qubit pulse duration
    t_left_ns = declare(int)  # QUA variable for the remainder
    f = declare(int)  # QUA variable for the qubit drive amplitude
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
        save(n, n_st)
        with for_(*from_array(f, frequencies)):
            update_frequency("qubit", f)
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
                                wait((duration_init - delay_before_readout) * u.ns - 4 - 9, "qubit")
                                pi_list[ii].run()

                # Long qubit pulse: baking and play combined
                with else_():
                    assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                    assign(t_left_ns, t - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                    # switch case to select the baked waveform corresponding to the burst duration
                    with switch_(t_left_ns, unsafe=True):
                        for ii in range(4):
                            with case_(ii):
                                # Drive the singlet-triplet qubit using an exchange pulse at the end of the manipulation step
                                wait((duration_init - delay_before_readout) * u.ns - t_cycles - 4 - 29, "qubit")
                                pi_list_4ns[ii].run()
                                play("pi", "qubit", duration=t_cycles)

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
        I_st.buffer(len(durations)).buffer(len(frequencies)).average().save("I")
        Q_st.buffer(len(durations)).buffer(len(frequencies)).average().save("Q")
        # DC current sensing
        dc_signal_st.buffer(len(durations)).buffer(len(frequencies)).average().save("dc_signal")

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
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples    plt.figure()
    plt.subplot(211)
    samples.con1.plot()
    plt.axhline(level_init[0], color="k", linestyle="--")
    plt.axhline(level_manip[0], color="k", linestyle="--")
    plt.axhline(level_readout[0], color="k", linestyle="--")
    plt.axhline(level_init[1], color="k", linestyle="--")
    plt.axhline(level_manip[1], color="k", linestyle="--")
    plt.axhline(level_readout[1], color="k", linestyle="--")
    plt.yticks(
        [
            level_readout[1],
            level_manip[1],
            level_init[1],
            0.0,
            level_init[0],
            level_manip[0],
            level_readout[0],
        ],
        ["readout", "manip", "init", "0", "init", "manip", "readout"],
    )
    plt.legend("")
    from macros import get_filtered_voltage

    plt.subplot(212)
    get_filtered_voltage(job.get_simulated_samples().con1.analog["1"], 1e-9, bias_tee_cut_off_frequency, True)
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
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
        plt.pcolor(durations, frequencies / u.MHz, R)
        plt.xlabel("Qubit pulse duration [ns]")
        plt.ylabel("Pulse intermediate frequency [MHz]")
        plt.subplot(122)
        plt.cla()
        plt.title("Phase [rad]")
        plt.pcolor(durations, frequencies / u.MHz, phase)
        plt.xlabel("Qubit pulse duration [ns]")
        plt.ylabel("Pulse intermediate frequency [MHz]")
        plt.tight_layout()
        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"DC_signal_data": DC_signal})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

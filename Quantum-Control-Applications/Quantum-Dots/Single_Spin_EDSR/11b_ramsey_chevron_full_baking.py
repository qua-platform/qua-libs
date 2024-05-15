"""
        RAMSEY-LIKE CHEVRON - using the baking tool (1ns granularity)
The goal of the script is to acquire exchange driven coherent oscillations by sweeping the idle time and detuning.
The QUA program is divided into three sections:
    1) step between the initialization, idle and measurement points using sticky elements (long timescale).
    2) apply two delta-g driven pi-half pulses separated by a low detuning pulse to increase J, using non-sticky elements (short timescale).
    3) measure the state of the qubit using either RF reflectometry or dc current sensing via PSB or Elzerman readout.
A compensation pulse can be added to the long timescale sequence in order to ensure 0 DC voltage on the fast line of
the bias-tee. Alternatively one can obtain the same result by changing the offset of the slow line of the bias-tee.

In the current implementation, the qubit pulses are played using the baking tool that allows for playing arbitrarily short
pulses with 1ns resolution. The drawback is that the pulses must be loaded to the OPX before the program starts (kind of
like an AWG), which quickly hit the waveform memory limit of the OPX (~65k samples per pulse processor).
Also note that the qubit pulses are played at the end of the global "idle" level whose duration is fixed.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the DC offsets of the external DC voltage source.
    - Connecting the OPX to the fast line of the plunger gates.
    - Having calibrated the initialization and readout point from the charge stability map and updated the configuration.
    - Having calibrated the delta-g driven pi-half parameters (detuning level and duration).

Before proceeding to the next node:
    - Extract the qubit frequency and T2*...
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.bakery import baking
from qualang_tools.addons.variables import assign_variables_to_element
import matplotlib.pyplot as plt
from macros import RF_reflectometry_macro, DC_current_sensing_macro


###################
# The QUA program #
###################

n_avg = 100
# Pulse duration sweep in ns
durations = np.arange(0, 101, 1)
assert max(durations) % 4 == 0
# Qubit detuning with respect to qubit_IF in Hz
detunings = np.arange(-10 * u.MHz, 10 * u.MHz, 100 * u.kHz)

# Add the relevant voltage points describing the "slow" sequence (no qubit pulse)
seq = OPX_virtual_gate_sequence(config, ["P1_sticky", "P2_sticky"])
seq.add_points("initialization", level_init, duration_init)
seq.add_points("idle", level_manip, duration_manip)
seq.add_points("readout", level_readout, duration_readout)

# Bake the Rabi pulses
pi_list = []
for t in durations:  # Create the different baked sequences
    t = int(t)
    with baking(config, padding_method="left") as b:  # don't use padding to assure error if timing is incorrect
        # Add the baked operation to the config
        wf_I = [pi_half_amp] * pi_half_length
        wf_Q = [0.0] * pi_half_length  # The baked waveforms (only the "I" quadrature)
        b.add_op("pi_half_baked", "qubit", [wf_I, wf_Q])

        # Baked sequence
        b.wait(max(durations) - t, "qubit")
        b.play("pi_half_baked", "qubit")  # Play the qubit pulse
        b.wait(t, "qubit")
        b.play("pi_half_baked", "qubit")  # Play the qubit pulse
    # Append the baking object in the list to call it from the QUA program
    pi_list.append(b)

with program() as Ramsey_chevron:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    t = declare(int)  # QUA variable for the qubit pulse duration
    f = declare(int)  # QUA variable for the qubit drive amplitude pre-factor
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    dc_signal = declare(fixed)  # QUA variable for the measured dc signal

    # Ensure that the result variables are assigned to the measurement elements
    assign_variables_to_element("tank_circuit", I, Q)
    assign_variables_to_element("TIA", dc_signal)

    with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
        save(n, n_st)
        with for_(*from_array(f, detunings)):  # Loop over the qubit pulse amplitude
            update_frequency("qubit", f + qubit_IF)
            with for_(t, 0, t < len(durations), t + 1):  # Loop over the qubit pulse duration
                with strict_timing_():  # Ensure that the sequence will be played without gap
                    # Navigate through the charge stability map
                    seq.add_step(voltage_point_name="initialization")
                    seq.add_step(voltage_point_name="idle")
                    seq.add_step(voltage_point_name="readout")
                    seq.add_compensation_pulse(duration=duration_compensation_pulse)

                    # Drive the singlet-triplet qubit using an exchange pulse at the end of the manipulation step
                    # switch case to select the baked waveform corresponding to the burst duration
                    with switch_(t, unsafe=True):
                        for ii in range(len(durations)):
                            with case_(ii):
                                # Drive the qubit by playing the MW pulse at the end of the manipulation step
                                wait(
                                    (duration_init + duration_manip - 2 * pi_half_length) * u.ns
                                    - max(durations) // 4
                                    - 4,
                                    "qubit",
                                )
                                wait(4, "qubit")  # Need 4 because of a gap
                                pi_list[ii].run()

                    # Measure the dot right after the qubit manipulation
                    wait((duration_init + duration_manip) * u.ns, "tank_circuit", "TIA")
                    I, Q, I_st, Q_st = RF_reflectometry_macro(I=I, Q=Q)
                    dc_signal, dc_signal_st = DC_current_sensing_macro(dc_signal=dc_signal)

                # Ramp the background voltage to zero to avoid propagating floating point errors
                seq.ramp_to_zero()

    # Stream processing section used to process the data before saving it.
    with stream_processing():
        n_st.save("iteration")
        # Cast the data into a 2D matrix and performs a global averaging of the received 2D matrices together.
        # RF reflectometry
        I_st.buffer(len(durations)).buffer(len(detunings)).average().save("I")
        Q_st.buffer(len(durations)).buffer(len(detunings)).average().save("Q")
        # DC current sensing
        dc_signal_st.buffer(len(durations)).buffer(len(detunings)).average().save("dc_signal")

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
    job = qmm.simulate(config, Ramsey_chevron, simulation_config)
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

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(Ramsey_chevron)
    # Get results from QUA program and initialize live plotting
    results = fetching_tool(job, data_list=["I", "Q", "dc_signal", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch the data from the last OPX run corresponding to the current slow axis iteration
        I, Q, DC_signal, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, reflectometry_readout_length)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        DC_signal = u.demod2volts(DC_signal, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg)
        # Plot data
        plt.subplot(121)
        plt.cla()
        plt.title(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        plt.pcolor(durations, (detunings + qubit_IF) / u.MHz, R)
        plt.xlabel("Qubit pulse duration [ns]")
        plt.ylabel("Pulse intermediate frequency [MHz]")
        plt.subplot(122)
        plt.cla()
        plt.title("Phase [rad]")
        plt.pcolor(durations, (detunings + qubit_IF) / u.MHz, phase)
        plt.xlabel("Idle time [ns]")
        plt.ylabel("Pulse intermediate frequency [MHz]")
        plt.tight_layout()
        plt.pause(0.1)

"""
        LANDAU-ZENER TRANSITIONS INVESTIGATION
The goal of the script is to investigate the dispersion relation by ramping (instead of stepping) across the inter-dot
transition with varying ramp durations and interaction times in steps of 4ns.

While the interaction time can be swept in QUA, the ramp duration needs to be scanned within a Python for loop.
The reason is that the OPX cannot update the ramp rate, the ramp duration and the length of the next pulse in real-time
using QUA variables without gaps. Such a sequence could be implemented in QUA, but only if the duration of the plateau
between the two ramps is larger than 64 ns.

To circumvent this limitation, the ramp duration (and ramp rate) is being swept in python by duplicating the while
sequence with the desired parameters set as python variables. The drawback of this method is that it significantly
increases the number of QUA commands sent to the OPX, such that the program memory limit can be reached for a large
number of ramp durations.

A compensation pulse can be added to the long timescale sequence in order to ensure 0 DC voltage on the fast line of
the bias-tee. Alternatively one can obtain the same result by changing the offset of the slow line of the bias-tee.

The state of the dot is measure using either RF reflectometry or dc current sensing.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the DC offsets of the external DC voltage source.
    - Connecting the OPX to the fast line of the plunger gates.
    - Having calibrated the initialization and readout point from the charge stability map and updated the configuration.

Before proceeding to the next node:
    - Identify the different adiabatic and diabatic regimes...
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.addons.variables import assign_variables_to_element
import matplotlib.pyplot as plt
from macros import RF_reflectometry_macro, DC_current_sensing_macro


###################
# The QUA program #
###################

n_avg = 1000
# Pulse duration sweep in ns - must be larger than 4 clock cycles
durations = np.arange(16, 200, 4)
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
ramp_durations = np.arange(16, 200, 4)

# Add the relevant voltage points describing the "slow" sequence (no qubit pulse)
seq = VoltageGateSequence(config, ["P1_sticky", "P2_sticky"])
seq.add_points("initialization", level_init, duration_init)
seq.add_points("idle", level_manip, duration_manip)
seq.add_points("readout", level_readout, duration_readout)

with program() as Landau_Zener_prog:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    t = declare(int)  # QUA variable for the qubit pulse duration
    t_R = declare(int)  # QUA variable for the qubit drive amplitude pre-factor
    rate = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    n_st = declare_stream()  # Stream for the iteration number (progress bar)

    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    dc_signal = declare(fixed)  # QUA variable for the measured dc signal
    I_st = declare_stream()  # Stream for the iteration number (progress bar)
    Q_st = declare_stream()  # Stream for the iteration number (progress bar)
    dc_signal_st = declare_stream()  # Stream for the iteration number (progress bar)

    # Ensure that the result variables are assigned to the measurement elements
    assign_variables_to_element("tank_circuit", I, Q)
    assign_variables_to_element("TIA", dc_signal)

    with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
        save(n, n_st)
        with for_(*from_array(t, durations)):  # Loop over the interaction duration
            # Here a python for loop is used to prevent gaps coming from dynamically changing the ramp rate, the ramp
            # duration and the duration of the next pulse in QUA.
            for t_R in ramp_durations:  # Loop over the ramp duration
                align()
                # Navigate through the charge stability map
                seq.add_step(voltage_point_name="initialization")
                seq.add_step(voltage_point_name="idle", ramp_duration=t_R, duration=t)
                seq.add_step(voltage_point_name="readout", ramp_duration=t_R)
                seq.add_compensation_pulse(duration=duration_compensation_pulse)

                # Measure the dot right after the qubit manipulation
                wait((duration_init + 2 * t_R) * u.ns + (t >> 2), "tank_circuit", "TIA")
                I, Q, I_st, Q_st = RF_reflectometry_macro(I=I, Q=Q, I_st=I_st, Q_st=Q_st)
                dc_signal, dc_signal_st = DC_current_sensing_macro(dc_signal=dc_signal, dc_signal_st=dc_signal_st)

                # with for_(*from_array(t_R, ramp_durations)):  # Loop over the qubit pulse amplitude
                seq.ramp_to_zero()
    # Stream processing section used to process the data before saving it.
    with stream_processing():
        n_st.save("iteration")
        # Cast the data into a 2D matrix and performs a global averaging of the received 2D matrices together.
        # RF reflectometry
        I_st.buffer(len(ramp_durations)).buffer(len(durations)).average().save("I")
        Q_st.buffer(len(ramp_durations)).buffer(len(durations)).average().save("Q")
        # DC current sensing
        dc_signal_st.buffer(len(ramp_durations)).buffer(len(durations)).average().save("dc_signal")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=100_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, Landau_Zener_prog, simulation_config)
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
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(Landau_Zener_prog)
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
        plt.pcolor(ramp_durations, durations, R)
        plt.xlabel("Ramp duration [ns]")
        plt.ylabel("Interaction pulse duration [ns]")
        plt.subplot(122)
        plt.cla()
        plt.title("Phase [rad]")
        plt.pcolor(ramp_durations, durations, phase)
        plt.xlabel("Ramp duration [ns]")
        plt.ylabel("Interaction duration [ns]")
        plt.tight_layout()
        plt.pause(0.1)

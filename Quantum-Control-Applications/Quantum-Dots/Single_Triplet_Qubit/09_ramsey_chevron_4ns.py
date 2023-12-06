"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""
import matplotlib.pyplot as plt
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.loops import from_array
from configuration import *
from scipy.optimize import minimize
from macros import RF_reflectometry_macro, DC_current_sensing_macro
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery import baking

from qm import generate_qua_script
from qualang_tools.addons.variables import assign_variables_to_element


###################
# The QUA program #
###################

n_avg = 100
# Pulse duration sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
durations = np.arange(16, 40, 20)
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
pi_levels = np.arange(0.21, 0.3, 0.01)

# Add the relevant voltage points describing the "slow" sequence (no qubit pulse)
seq = OPX_background_sequence(config, ["P1_sticky", "P2_sticky"])
seq.add_points("initialization", level_init, duration_init)
seq.add_points("manipulation", level_manip, duration_manip)
seq.add_points("readout", level_readout, duration_readout)

with program() as Ramsey_chevron:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    t = declare(int)  # QUA variable for the qubit pulse duration
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    n_st = declare_stream()  # Stream for the iteration number (progress bar)

    I = declare(fixed)
    Q = declare(fixed)
    dc_signal = declare(fixed)
    assign_variables_to_element("tank_circuit", I, Q)
    assign_variables_to_element("TIA", dc_signal)
    # seq.add_step(voltage_point_name="readout", duration=16)
    with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
        save(n, n_st)
        with for_(*from_array(a, pi_levels)):  # Loop over the qubit pulse amplitude
            with for_(*from_array(t, durations)):  # Loop over the qubit pulse duration
                with strict_timing_():  # Ensure that the sequence will be played without gap
                    # Navigate through the charge stability map
                    seq.add_step(voltage_point_name="initialization", ramp_duration=None)
                    seq.add_step(voltage_point_name="manipulation", level=[a, -a])
                    seq.add_step(voltage_point_name="readout")
                    seq.add_compensation_pulse(duration=duration_compensation_pulse)

                    # Drive the singlet-triplet qubit using an exchange pulse at the end of the manipulation step
                    wait((duration_init + duration_manip) * u.ns - (t>>2) -32//4 - 4, "P1", "P2") # Need -4 because of a gap
                    wait( 4, "P1", "P2") # Need -4 because of a gap
                    play("pi_half" * amp((pi_half_amps[0]-a) * 4), "P1")
                    play("pi_half" * amp((pi_half_amps[1]+a) * 4), "P2")
                    wait(t>>2, "P1", "P2") # Need -4 because of a gap
                    play("pi_half" * amp((pi_half_amps[0]-a) * 4), "P1")
                    play("pi_half" * amp((pi_half_amps[1]+a) * 4), "P2")
                    # Measure the dot right after the qubit manipulation
                    wait((duration_init + duration_manip) * u.ns, "tank_circuit", "TIA")
                    I, Q, I_st, Q_st = RF_reflectometry_macro(I=I, Q=Q)
                    dc_signal, dc_signal_st = DC_current_sensing_macro(dc_signal=dc_signal)
                ramp_to_zero("P1_sticky")
                ramp_to_zero("P2_sticky")
    # Stream processing section used to process the data before saving it.
    with stream_processing():
        n_st.save("iteration")
        # Cast the data into a 2D matrix and performs a global averaging of the received 2D matrices together.
        # RF reflectometry
        I_st.buffer(len(durations)).buffer(len(pi_levels)).average().save("I")
        Q_st.buffer(len(durations)).buffer(len(pi_levels)).average().save("Q")
        # DC current sensing
        dc_signal_st.buffer(len(durations)).buffer(len(pi_levels)).average().save("dc_signal")

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
    job.get_simulated_samples().con1.plot()
    plt.axhline(0.1, color="k", linestyle="--")
    plt.axhline(0.3, color="k", linestyle="--")
    plt.axhline(0.2, color="k", linestyle="--")
    plt.axhline(-0.1, color="k", linestyle="--")
    plt.axhline(-0.3, color="k", linestyle="--")
    plt.axhline(-0.2, color="k", linestyle="--")
    plt.yticks([-0.2, -0.3, -0.1, 0.0, 0.1, 0.3, 0.2], ["readout", "manip", "init", "0", "init", "manip", "readout"])
    plt.legend("")
    samples = job.get_simulated_samples()
    report = job.get_simulated_waveform_report()
    report.create_plot(samples, plot=True)
    from macros import get_filtered_voltage
    # get_filtered_voltage(list(job.get_simulated_samples().con1.analog["5"][8912:17639]) * 10, 1e-9, 1e3, True)
    get_filtered_voltage(job.get_simulated_samples().con1.analog["5"], 1e-9, 1e3, True)

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(Ramsey_chevron)

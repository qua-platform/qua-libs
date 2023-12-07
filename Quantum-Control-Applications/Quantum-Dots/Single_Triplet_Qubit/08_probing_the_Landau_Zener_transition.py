"""
        RABI CHEVRON - using standard QUA (pulse > 16ns and 4ns granularity)
The goal of the script is to acquire
To do so, the charge stability map is acquired by scanning the voltages provided by the QDAC2,
to the DC part of the bias-tees connected to the plunger gates, while 2 OPX channels are stepping the voltages on the fast
lines of the bias-tees to navigate through the triangle in voltage space (empty - random initialization - measurement).

Depending on the cut-off frequency of the bias-tee, it may be necessary to adjust the barycenter (voltage offset) of each
triangle so that the fast line of the bias-tees sees zero voltage in average. Otherwise, the high-pass filtering effect
of the bias-tee will distort the fast pulses over time. A function has been written for this.

In the current implementation, the OPX is also measuring (either with DC current sensing or RF-reflectometry) during the
readout window (last segment of the triangle).
A single-point averaging is performed and the data is extracted while the program is running to display the results line-by-line.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the parameters of the QDAC2 and preloading the two voltages lists for the slow and fast axes.
    - Connect the two plunger gates (DC line of the bias-tee) to the QDAC2 and two digital markers from the OPX to the
      QDAC2 external trigger ports.
    - Connect the OPX to the fast line of the plunger gates for playing the triangle pulse sequence.

Before proceeding to the next node:
    - Identify the PSB region and update the config.
"""
import matplotlib.pyplot as plt
import numpy as np
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

# TODO: tricky to sweep the ramp rate without gap...

###################
# The QUA program #
###################

n_avg = 100
# Pulse duration sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
durations = np.arange(16, 40, 20)
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
ramp_durations = np.arange(16, 200, 4)
eps = [0.25, -0.25]
rates = 1/ramp_durations

# Add the relevant voltage points describing the "slow" sequence (no qubit pulse)
seq = OPX_background_sequence(config, ["P1_sticky", "P2_sticky"])
seq.add_points("initialization", level_init, duration_init)
seq.add_points("idle", level_manip, duration_manip)
seq.add_points("readout", level_readout, duration_readout)

with program() as Rabi_chevron:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    t = declare(int)  # QUA variable for the qubit pulse duration
    t_R = declare(int)  # QUA variable for the qubit drive amplitude pre-factor
    rate = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    n_st = declare_stream()  # Stream for the iteration number (progress bar)

    # seq.add_step(voltage_point_name="readout", duration=16)
    with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
        save(n, n_st)
        with for_(*from_array(t_R, ramp_durations)):  # Loop over the qubit pulse amplitude
            assign(rate, Math.div(1, t_R))
            with for_(*from_array(t, durations)):  # Loop over the qubit pulse duration
                with strict_timing_():  # Ensure that the sequence will be played without gap
                    # Navigate through the charge stability map
                    seq.add_step(voltage_point_name="initialization")
                    seq.add_step(voltage_point_name="idle", ramp_duration=16, duration=t)
                    seq.add_step(voltage_point_name="readout", ramp_duration=16)
                    seq.add_compensation_pulse(duration=duration_compensation_pulse)
                    # Drive the singlet-triplet qubit using an exchange pulse at the end of the manipulation step
                    wait((duration_init + duration_manip) * u.ns - (t>>2) - 4, "P1", "P2") # Need -4 because of a gap
                    wait( 4, "P1", "P2") # Need -4 because of a gap
                    # play(ramp(eps[0]*rate), "P1", duration=t_R>>2)
                    # play(ramp(eps[1]*rate), "P2", duration=t_R>>2)
                    # play("step" * amp((eps[0]-level_manip[0]) * 4), "P1", duration=t>>2)
                    # play("step" * amp((eps[1]-level_manip[1]) * 4), "P2", duration=t>>2)
                    # Measure the dot right after the qubit manipulation
                    wait((duration_init + duration_manip) * u.ns, "tank_circuit", "TIA")
                    I, Q, I_st, Q_st = RF_reflectometry_macro()
                    dc_signal, dc_signal_st = DC_current_sensing_macro()
                ramp_to_zero("P1_sticky")
                ramp_to_zero("P2_sticky")
    # Stream processing section used to process the data before saving it.
    with stream_processing():
        n_st.save("iteration")
        # Cast the data into a 2D matrix and performs a global averaging of the received 2D matrices together.
        # RF reflectometry
        I_st.buffer(len(durations)).buffer(len(ramp_durations)).average().save("I")
        Q_st.buffer(len(durations)).buffer(len(ramp_durations)).average().save("Q")
        # DC current sensing
        dc_signal_st.buffer(len(durations)).buffer(len(ramp_durations)).average().save("dc_signal")

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, Rabi_chevron, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    plt.axhline(level_init[0], color="k", linestyle="--")
    plt.axhline(level_manip[0], color="k", linestyle="--")
    plt.axhline(level_readout[0], color="k", linestyle="--")
    plt.axhline(level_init[1], color="k", linestyle="--")
    plt.axhline(level_manip[1], color="k", linestyle="--")
    plt.axhline(level_readout[1], color="k", linestyle="--")
    plt.axhline(pi_half_amps[0], color="k", linestyle="--")
    plt.axhline(pi_half_amps[1], color="k", linestyle="--")
    plt.yticks([level_readout[1], level_manip[1], level_init[1], 0.0, level_init[0], level_manip[0], level_readout[0]],
               ["readout", "manip", "init", "0", "init", "manip", "readout"])

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
    job = qm.execute(Rabi_chevron)

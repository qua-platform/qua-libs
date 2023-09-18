"""
        T1 MEASUREMENT
The sequence consists in putting the qubit in the excited stated by playing the x180 pulse and measuring the resonator
after a varying time. The qubit T1 is extracted by fitting the exponential decay of the measured quadratures.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubit T1 in the state.
    - Save the current state by calling machine._save("current_state.json")
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from macros import qua_declaration, multiplexed_readout
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

#######################################################
# Get the config from the machine in configuration.py #
#######################################################

# Build the config
config = build_config(machine)

###################
# The QUA program #
###################
n_avg = 1000
cooldown_time = 5 * max(qb1.T1, qb2.T1)
# The wait time sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
# Linear sweep
t_delay = np.arange(4, 10000, 40)
# Log sweep
# t_delay = np.logspace(np.log10(4), np.log10(12 * u.us), 29)

with program() as T1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)  # QUA variable for the wait time

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(t, t_delay)):
            # qubit 1
            play("x180", qb1.name + "_xy")
            wait(t, qb1.name + "_xy")

            # qubit 2
            play("x180", qb2.name + "_xy")
            wait(t, qb2.name + "_xy")

            align()
            multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, weights="rotated_")
            # Wait for the qubits to decay to the ground state
            wait(cooldown_time * u.ns)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        # If log sweep, then the swept values will be slightly different from np.logspace because of integer rounding in QUA.
        # get_equivalent_log_array() is used to get the exact values used in the QUA program.
        if np.isclose(np.std(t_delay[1:] / t_delay[:-1]), 0, atol=1e-3):
            t_delay = get_equivalent_log_array(t_delay)
        for i in range(len(active_qubits)):
            I_st[i].buffer(len(t_delay)).average().save(f"I{i+1}")
            Q_st[i].buffer(len(t_delay)).average().save(f"Q{i+1}")
        n_st.save("n")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, T1, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(T1)
    # Get results from QUA program
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Convert the results into Volts
        I1, Q1 = u.demod2volts(I1, rr1.readout_pulse_length), u.demod2volts(Q1, rr1.readout_pulse_length)
        I2, Q2 = u.demod2volts(I2, rr2.readout_pulse_length), u.demod2volts(Q2, rr2.readout_pulse_length)
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot results
        plt.suptitle("T1")
        plt.subplot(221)
        plt.cla()
        plt.plot(t_delay * 4, I1, ".")
        plt.title(f"{qb1.name}")
        plt.ylabel("I quadrature [V]")
        plt.subplot(223)
        plt.cla()
        plt.plot(t_delay * 4, Q1, ".")
        plt.xlabel("Wait time [ns]")
        plt.ylabel("Q quadrature [V]")
        plt.subplot(222)
        plt.cla()
        plt.plot(t_delay * 4, I2, ".")
        plt.title(f"{qb2.name}")
        plt.subplot(224)
        plt.cla()
        plt.plot(t_delay * 4, Q2, ".")
        plt.xlabel("Wait time [ns]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
# Fit the data to extract T1
try:
    from qualang_tools.plot.fitting import Fit

    fit = Fit()
    plt.figure()
    plt.suptitle("T1")
    plt.subplot(121)
    fit_1 = fit.T1(4 * t_delay, I1, plot=True)
    plt.xlabel("Wait time [ns]")
    plt.ylabel("I quadrature [V]")
    plt.title(f"{qb1.name}")
    plt.legend((f"T1 = {np.round(np.abs(fit_1['T1'][0]) / 4) * 4:.0f} ns",))
    plt.subplot(122)
    fit_2 = fit.T1(4 * t_delay, I2, plot=True)
    plt.xlabel("Wait time [ns]")
    plt.ylabel("I quadrature [V]")
    plt.title(f"{qb2.name}")
    plt.legend((f"T1 = {np.round(np.abs(fit_2['T1'][0]) / 4) * 4:.0f} ns",))
    plt.tight_layout()

    # Update the state
    qb1.T1 = int(np.round(np.abs(fit_1["T1"][0]) / 4) * 4)
    qb2.T1 = int(np.round(np.abs(fit_2["T1"][0]) / 4) * 4)
except (Exception,):
    pass

# machine._save("current_state.json")

"""
        TIME RABI
The sequence consists in playing the qubit pulse (x180 or square_pi or else) and measuring the state of the resonator
for different qubit pulse durations.
The results are then post-processed to find the qubit pulse duration for the chosen amplitude.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and pi pulse amplitude (rabi_chevron_amplitude or power_rabi).
    - Set the qubit frequency and desired pi pulse amplitude (pi_amp) in the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the qubit pulse duration (pi_len) in the state.
    - Save the current state by calling machine._save("current_state.json")
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
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
n_avg = 100  # The number of averages
cooldown_time = 5 * max(qb1.T1, qb2.T1)
# Pulse duration sweep (in clock cycles = 4ns)
# must be larger than 4 clock cycles and larger than the pi_len defined in the state
times = np.arange(4, 200, 2)

with program() as rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)  # QUA variable for the qubit pulse duration

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, times)):
            # Play the qubit drives with varying durations
            play("x180", qb1.name + "_xy", duration=t)
            play("x180", qb2.name + "_xy", duration=t)
            # Align all elements to measure after playing the qubit pulse.
            align()
            # Start using Rotated-Readout:
            multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, weights="rotated_")
            # Wait for the qubit to decay to the ground state
            wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(times)).average().save("I1")
        Q_st[0].buffer(len(times)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(times)).average().save("I2")
        Q_st[1].buffer(len(times)).average().save("Q2")

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
    job = qmm.simulate(config, rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rabi)
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
        progress_counter(n, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle("Time Rabi")
        plt.subplot(221)
        plt.cla()
        plt.plot(times * 4, I1)
        plt.title(f"{qb1.name}")
        plt.ylabel("I quadrature [V]")
        plt.subplot(223)
        plt.cla()
        plt.plot(times * 4, Q1)
        plt.xlabel("qubit pulse duration [ns]")
        plt.ylabel("Q quadrature [V]")
        plt.subplot(222)
        plt.cla()
        plt.plot(times * 4, I2)
        plt.title(f"{qb2.name}")
        plt.subplot(224)
        plt.cla()
        plt.plot(times * 4, Q2)
        plt.xlabel("qubit pulse duration [ns]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Fit the results to extract the x180 length
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        plt.suptitle("Time Rabi")
        plt.subplot(121)
        rabi_fit1 = fit.rabi(4 * times, I1, plot=True)
        plt.title(f"{qb1.name}")
        plt.xlabel("Rabi pulse duration [ns]")
        plt.ylabel("I quadrature [V]")
        plt.subplot(122)
        rabi_fit2 = fit.rabi(4 * times, I2, plot=True)
        plt.title(f"{qb2.name}")
        plt.xlabel("Rabi pulse duration [ns]")
        plt.ylabel("I quadrature [V]")
        plt.tight_layout()
        print(f"Optimal x180_len for {qb1.name} = {round(1 / rabi_fit1['f'][0] / 2 / 4) * 4} ns for {qb1.xy.pi_amp:} V")
        print(f"Optimal x180_len for {qb2.name} = {round(1 / rabi_fit2['f'][0] / 2 / 4) * 4} ns for {qb2.xy.pi_amp:} V")
        qb1.xy.pi_length = round(1 / rabi_fit1["f"][0] / 2 / 4) * 4
        qb2.xy.pi_length = round(1 / rabi_fit2["f"][0] / 2 / 4) * 4
    except (Exception,):
        pass

# machine._save("current_state.json")

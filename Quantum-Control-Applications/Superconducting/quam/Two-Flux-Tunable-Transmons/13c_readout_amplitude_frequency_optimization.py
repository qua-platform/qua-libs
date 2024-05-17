"""
        READOUT OPTIMISATION: AMPLITUDE VS FREQUENCY
The sequence consists in measuring the state of the resonator after thermalization (qubit in |g>) and after
playing a pi pulse to the qubit (qubit in |e>) successively while sweeping the readout amplitude and frequency.
The 'I' & 'Q' quadratures when the qubit is in |g> and |e> are extracted to derive the readout fidelity.
The optimal readout amplitude is chosen as to maximize the readout fidelity.

This version can be particularly useful when the resonator is driven in a regime where its frequency depends on the readout amplitude.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.

Next steps before going to the next node:
    - Update the readout amplitude and frequency in the state.
    - Update the readout fidelity in the state.
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.units import unit
from qualang_tools.analysis.discriminator import two_state_discriminator

import matplotlib.pyplot as plt
import numpy as np

from components import QuAM
from macros import qua_declaration, multiplexed_readout, node_save


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("state.json")
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
q1 = machine.active_qubits[0]
q2 = machine.active_qubits[1]
rr1 = machine.active_qubits[0].resonator
rr2 = machine.active_qubits[1].resonator


###################
# The QUA program #
###################
n_runs = 100  # The number of averages

# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
amplitudes = np.arange(0.5, 1.5, 0.05)
# The frequency sweep parameters with respect to the resonators resonance frequencies
dfs = np.arange(-5e6, 5e6, 0.1e6)


with program() as ro_amp_freq_opt:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(nb_of_qubits=2)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)  # QUA variable for the readout amplitude
    df = declare(int)  # QUA variable for the readout frequency detuning
    counter = declare(int, value=0)  # Counter for the progress bar

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(*from_array(df, dfs)):
        save(counter, n_st)
        # Update the resonators frequency
        update_frequency(rr1.name, df + rr1.intermediate_frequency)
        update_frequency(rr2.name, df + rr2.intermediate_frequency)
        with for_(*from_array(a, amplitudes)):
            with for_(n, 0, n < n_runs, n + 1):
                # ground iq blobs for both qubits
                wait(machine.get_thermalization_time * u.ns)
                align()
                multiplexed_readout(machine, I_g, I_g_st, Q_g, Q_g_st, amplitude_scale=a)

                # excited iq blobs for both qubits
                align()
                # Wait for thermalization again in case of measurement induced transitions
                wait(machine.get_thermalization_time * u.ns)
                q1.xy.play("x180")
                q2.xy.play("x180")
                align()
                multiplexed_readout(machine, I_e, I_e_st, Q_e, Q_e_st, amplitude_scale=a)
        # Save the counter to get the progress bar
        assign(counter, counter + 1)

    with stream_processing():
        # Save all streamed points for plotting the IQ blobs
        for i in range(2):
            I_g_st[i].buffer(n_runs).buffer(len(amplitudes)).buffer(len(dfs)).save(f"I_g_q{i}")
            Q_g_st[i].buffer(n_runs).buffer(len(amplitudes)).buffer(len(dfs)).save(f"Q_g_q{i}")
            I_e_st[i].buffer(n_runs).buffer(len(amplitudes)).buffer(len(dfs)).save(f"I_e_q{i}")
            Q_e_st[i].buffer(n_runs).buffer(len(amplitudes)).buffer(len(dfs)).save(f"Q_e_q{i}")
        n_st.save("n")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ro_amp_freq_opt, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_active_qubits(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ro_amp_freq_opt)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["n"], mode="live")
    # Get progress counter to monitor runtime of the program
    while results.is_processing():
        # Fetch results
        iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration[0], len(dfs), start_time=results.get_start_time())

    # fetch data
    results = fetching_tool(job, ["I_g_q0", "Q_g_q0", "I_e_q0", "Q_e_q0", "I_g_q1", "Q_g_q1", "I_e_q1", "Q_e_q1"])
    I_g_q1, Q_g_q1, I_e_q1, Q_e_q1, I_g_q2, Q_g_q2, I_e_q2, Q_e_q2 = results.fetch_all()
    # Process the data
    fidelity_vec = [np.zeros((len(amplitudes), len(dfs))), np.zeros((len(amplitudes), len(dfs)))]
    for j in range(len(dfs)):
        for i in range(len(amplitudes)):
            _, _, fidelity_q1, _, _, _, _ = two_state_discriminator(
                I_g_q1[j][i], Q_g_q1[j][i], I_e_q1[j][i], Q_e_q1[j][i], b_print=False, b_plot=False
            )
            _, _, fidelity_q2, _, _, _, _ = two_state_discriminator(
                I_g_q2[j][i], Q_g_q2[j][i], I_e_q2[j][i], Q_e_q2[j][i], b_print=False, b_plot=False
            )
            fidelity_vec[0][i][j] = fidelity_q1
            fidelity_vec[1][i][j] = fidelity_q2

    # Plot the data
    fig = plt.figure()
    plt.suptitle("Readout amplitude optimization")
    plt.subplot(121)
    plt.pcolor(
        (dfs + rr1.intermediate_frequency) / u.MHz, amplitudes * rr1.operations["readout"].amplitude, fidelity_vec[0]
    )
    plt.title(f"{rr1.name}")
    plt.colorbar()
    plt.ylabel("Readout amplitude [V]")
    plt.xlabel("Readout IF [MHz]")
    plt.subplot(122)
    plt.pcolor(
        (dfs + rr2.intermediate_frequency) / u.MHz, amplitudes * rr2.operations["readout"].amplitude, fidelity_vec[1]
    )
    plt.title(f"{rr2.name}")
    plt.colorbar()
    plt.ylabel("Readout amplitude [V]")
    plt.xlabel("Readout IF [MHz]")
    plt.tight_layout()

    # Update the state
    rr1.readout_pulse_amp *= amplitudes[np.where(fidelity_vec[0] == np.amax(fidelity_vec[0]))[0][0]]

    rr2.readout_pulse_amp *= amplitudes[np.where(fidelity_vec[1] == np.amax(fidelity_vec[1]))[0][0]]
    rr1.readout_fidelity = np.amax(fidelity_vec[0])
    rr2.readout_fidelity = np.amax(fidelity_vec[1])

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {
        f"{rr1.name}_amplitude": amplitudes * rr1.operations["readout"].amplitude,
        f"{rr1.name}_frequency": dfs + rr1.intermediate_frequency,
        f"{rr1.name}_fidelity": fidelity_vec[0],
        f"{rr1.name}_amp_opt": rr1.operations["readout"].amplitude,
        f"{rr1.name}_if_opt": rr1.intermediate_frequency
        + dfs[np.where(fidelity_vec[0] == np.amax(fidelity_vec[0]))[1][0]],
        f"{rr2.name}_amplitude": amplitudes * rr2.operations["readout"].amplitude,
        f"{rr2.name}_frequency": dfs + rr2.intermediate_frequency,
        f"{rr2.name}_fidelity": fidelity_vec[1],
        f"{rr2.name}_amp_opt": rr2.operations["readout"].amplitude,
        f"{rr2.name}_if_opt": rr2.intermediate_frequency
        + dfs[np.where(fidelity_vec[1] == np.amax(fidelity_vec[1]))[1][0]],
        "figure": fig,
    }
    rr1.intermediate_frequency += dfs[np.where(fidelity_vec[0] == np.amax(fidelity_vec[0]))[1][0]]
    rr2.intermediate_frequency += dfs[np.where(fidelity_vec[1] == np.amax(fidelity_vec[1]))[1][0]]
    node_save("readout_amplitude_frequency_optimization", data, machine)

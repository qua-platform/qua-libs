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
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np
import os

from quam_components import QuAM
from macros import qua_declaration, multiplexed_readout, node_save


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load(os.path.join('..', 'configuration', 'quam_state'))
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
qubits = machine.active_qubits
num_qubits = len(qubits)

###################
# The QUA program #
###################

operation = "x180"  # The qubit operation to play
n_avg = 100  # The number of averages

# Pulse duration sweep (in clock cycles = 4ns)
# must be larger than 4 clock cycles and larger than the pi_len defined in the state
times = np.arange(4, 200, 2)

with program() as time_rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=num_qubits)
    t = declare(int)  # QUA variable for the qubit pulse duration

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, times)):
            # Play the qubit drives with varying durations
            for qubit in qubits:
                qubit.xy.play(operation, duration=t)
            # Align all elements to measure after playing the qubit pulse.
            align()
            # QUA macro the readout the state of the active resonators (defined in macros.py)
            multiplexed_readout(qubits, I, I_st, Q, Q_st)
            # Wait for the qubit to decay to the ground state
            wait(machine.thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            I_st[i].buffer(len(times)).average().save(f"I{i+1}")
            Q_st[i].buffer(len(times)).average().save(f"Q{i+1}")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, time_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(time_rabi)
    # Get results from QUA program
    data_list = ["n"] + sum([[f"I{i+1}", f"Q{i+1}"] for i in range(num_qubits)], [])
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    while results.is_processing():
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I = fetched_data[1::2]
        Q = fetched_data[2::2]
        # Progress bar
        progress_counter(n, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle("Time Rabi")
        I_volts, Q_volts = [], []
        for i, qubit in enumerate(qubits):
            I_volts.append(u.demod2volts(I[i], qubit.resonator.operations["readout"].length))
            Q_volts.append(u.demod2volts(Q[i], qubit.resonator.operations["readout"].length))
            plt.subplot(2, num_qubits, i+1)
            plt.cla()
            plt.plot(times * 4, I_volts[i])
            plt.title(f"{qubit.name}")
            plt.ylabel("I quadrature [V]")
            plt.subplot(2, num_qubits, i+num_qubits+1)
            plt.cla()
            plt.plot(times * 4, Q_volts[i])
            plt.xlabel("qubit pulse duration [ns]")
            plt.ylabel("Q quadrature [V]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    data = {}
    for i, qubit in enumerate(qubits):
        data[f"{qubit.name}_time"] = times * 4
        data[f"{qubit.name}_I"] = np.abs(I_volts[i])
        data[f"{qubit.name}_Q"] = np.angle(Q_volts[i])
    data["figure"] = fig

    # Fit the results to extract the x180 length
    for i, qubit in enumerate(qubits):
        try:
            from qualang_tools.plot.fitting import Fit

            fit = Fit()
            plt.figure()
            plt.suptitle("Time Rabi")
            plt.subplot(1, num_qubits, i+1)
            rabi_fit1 = fit.rabi(4 * times, I_volts[i], plot=True)
            plt.title(f"{qubit.name}")
            plt.xlabel("Rabi pulse duration [ns]")
            plt.ylabel("I quadrature [V]")
            qubit.xy.operations[operation].length = int(round(1 / rabi_fit1["f"][0] / 2 / 4) * 4)
            data[f"{qubit.name}"] = {"x180_length": qubit.xy.operations[operation].length, "successful_fit": True}
            print(
                f"Optimal x180_len for {qubit.name} = {qubit.xy.operations[operation].length} ns "
                f"for {qubit.xy.operations[operation].amplitude:} V"
            )

        except (Exception,):
            data[f"{qubit.name}"] = {"successful_fit": False}

    # Save data from the node
    node_save("time_rabi", data, machine)

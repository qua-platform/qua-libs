"""
T1 MEASUREMENT
The sequence consists in putting the qubits in the excited state by playing the x180 pulse and measuring the resonator
after a varying time. The qubit T1 is extracted by fitting the exponential decay of the measured quadratures.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubit T1 in the state.
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np
import os

from quam_components import QuAM
from macros import qua_declaration, multiplexed_readout, node_save


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handle units and conversions.
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
n_avg = 1000

# The wait time sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
# Linear sweep
t_delay = np.arange(4, 10000, 40)
# Log sweep
# t_delay = np.logspace(np.log10(4), np.log10(12 * u.us), 29)

with program() as T1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=num_qubits)
    t = declare(int)  # QUA variable for the wait time

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(t, t_delay)):
            for qubit in qubits:
                qubit.xy.play("x180")
                qubit.xy.wait(t)
            align()
            multiplexed_readout(qubits, I, I_st, Q, Q_st)
            # Wait for the qubits to decay to the ground state
            wait(machine.thermalization_time * u.ns)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        # If log sweep, then the swept values will be slightly different from np.logspace because of integer rounding in QUA.
        # get_equivalent_log_array() is used to get the exact values used in the QUA program.
        if np.isclose(np.std(t_delay[1:] / t_delay[:-1]), 0, atol=1e-3):
            t_delay = get_equivalent_log_array(t_delay)

        for i in range(len(machine.active_qubits)):
            I_st[i].buffer(len(t_delay)).average().save(f"I{i+1}")
            Q_st[i].buffer(len(t_delay)).average().save(f"Q{i+1}")
        n_st.save("n")


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
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(T1)
    # Get results from QUA program
    data_list = sum([[f"I{i+1}", f"Q{i+1}"] for i in range(num_qubits)], ["n"])
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    fig, axes = plt.subplots(2, num_qubits, figsize=(4*num_qubits, 8))
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I_data = fetched_data[1::2]
        Q_data = fetched_data[2::2]
        # Convert the results into Volts
        I_volts = [u.demod2volts(I, qubit.resonator.operations["readout"].length) for I, qubit in zip(I_data, qubits)]
        Q_volts = [u.demod2volts(Q, qubit.resonator.operations["readout"].length) for Q, qubit in zip(Q_data, qubits)]
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot results
        plt.suptitle("T1")
        for i, qubit in enumerate(qubits):
            axes[i, 0].cla()
            axes[i, 0].plot(t_delay * 4, I_volts[i], ".")
            axes[i, 0].set_title(f"{qubit.name}")
            axes[i, 0].set_ylabel("I quadrature [V]")
            axes[i, 1].cla()
            axes[i, 1].plot(t_delay * 4, Q_volts[i], ".")
            axes[i, 1].set_xlabel("Wait time [ns]")
            axes[i, 1].set_ylabel("Q quadrature [V]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {}
    for i, qubit in enumerate(qubits):
        data[f"{qubit.name}_time"] = t_delay * 4
        data[f"{qubit.name}_I"] = I_volts[i]
        data[f"{qubit.name}_Q"] = Q_volts[i]
    data["figure"] = fig

    # Fit the data to extract T1
    for i, qubit in enumerate(qubits):
        try:
            from qualang_tools.plot.fitting import Fit

            fit = Fit()
            plt.figure()
            plt.suptitle("T1")
            plt.subplot(num_qubits, 1, i+1)
            fit_res = fit.T1(4 * t_delay, I_volts[i], plot=True)
            plt.xlabel("Wait time [ns]")
            plt.ylabel("I quadrature [V]")
            plt.title(f"{qubit.name}")
            plt.legend((f"T1 = {np.round(np.abs(fit_res['T1'][0]) / 4) * 4:.0f} ns",))
            qubit.T1 = int(np.round(np.abs(fit_res["T1"][0]) / 4) * 4)
            data[f"{qubit.name}"] = {"T1": qubit.T1, "successful_fit": True}
            plt.tight_layout()
        except (Exception,):
            data[f"{qubit.name}"] = {"successful_fit": False}
            pass

    # Save data from the node
    node_save("T1", data, machine)

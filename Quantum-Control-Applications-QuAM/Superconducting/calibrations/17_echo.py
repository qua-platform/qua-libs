# %%
"""
ECHO MEASUREMENT
The program consists in playing a Ramsey sequence with an echo pulse in the middle to compensate for dephasing and
enhance the coherence time (x90 - idle_time - x180 - idle_time - x90 - measurement) for different idle times.
Here the gates are on resonance so no oscillation is expected.

From the results, one can fit the exponential decay and extract T2.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubits T2 echo in the state.
    - Save the current state by calling machine.save("quam")
"""

from pathlib import Path

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save

import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
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
n_avg = 2

# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.arange(4, 2000, 5)  # Linear sweep
# taus = np.logspace(np.log10(4), np.log10(10_000), 21)  # Log sweep


with program() as echo:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(t, idle_times)):
            # Echo sequence
            for qubit in qubits:
                qubit.xy.play("x90")
                qubit.xy.wait(t)
                qubit.xy.play("x180")
                qubit.xy.wait(t)
                qubit.xy.play("x90")

            # Align the elements to measure after playing the qubit pulse.
            align()
            # Measure the state of the resonators
            multiplexed_readout(qubits, I, I_st, Q, Q_st)
            # Wait for the qubits to decay to the ground state
            wait(machine.thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        # If log sweep, then the swept values will be slightly different from np.logspace because of integer rounding in QUA.
        # get_equivalent_log_array() is used to get the exact values used in the QUA program.
        if np.isclose(np.std(idle_times[1:] / idle_times[:-1]), 0, atol=1e-3):
            idle_times = get_equivalent_log_array(idle_times)
        for i in range(num_qubits):
            I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, echo, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(echo)
    # Get results from QUA program
    data_list = sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_qubits)], ["n"])
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    fig, axes = plt.subplots(2, num_qubits, figsize=(4 * num_qubits, 8))
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
        plt.suptitle("Echo")
        for i, (ax, qubit) in enumerate(zip(axes, qubits)):
            ax[0].cla()
            ax[0].plot(8 * idle_times, I_volts[i])
            ax[0].set_ylabel("I [V]")
            ax[0].set_title(f"{qubit.name}")
            ax[1].cla()
            ax[1].plot(8 * idle_times, Q_volts[i])
            ax[1].set_xlabel("Idle time [ns]")
            ax[1].set_ylabel("Q [V]")
            ax[1].set_title(f"{qubit.name}")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {}
    for i, qubit in enumerate(qubits):
        data[f"{qubit.name}_amplitude"] = 8 * idle_times
        data[f"{qubit.name}_I"] = np.abs(I_volts[i])
        data[f"{qubit.name}_Q"] = np.angle(Q_volts[i])
    data["figure"] = fig

    fig_analysis = plt.figure()
    plt.suptitle("Echo")
    # Fit the data to extract T2 echo
    for i, qubit in enumerate(qubits):
        try:
            from qualang_tools.plot.fitting import Fit

            fit = Fit()
            plt.subplot(num_qubits, 1, i + 1)
            fit_I = fit.T1(8 * idle_times, I_volts[i], plot=True)
            plt.xlabel("Idle time [ns]")
            plt.ylabel("I [V]")
            plt.title(f"{qubit.name}")
            plt.legend((f"T2 = {int(fit_I['T1'][0])} ns",))

            # Update the state
            qubit.T2echo = int(fit_I["T1"][0])
            data[f"{qubit.name}"] = {"T2": qubit.T2echo, "successful_fit": True}
            plt.tight_layout()
            data["figure_analysis"] = fig_analysis
        except (Exception,):
            data[f"{qubit.name}"] = {"successful_fit": True}
            pass
    plt.show()

    # Save data from the node
    node_save(machine, "ramsey", data, additional_files=True)

# %%

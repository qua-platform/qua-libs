"""
RAMSEY CHEVRON (IDLE TIME VS FREQUENCY)
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different qubit intermediate
frequencies and idle times.
From the results, one can estimate the qubit frequency more precisely than by doing Rabi and also gets a rough estimate
of the qubit coherence time.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubit frequency (f_01) in the state.
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
n_avg = 100  # Number of averaging loops

# Frequency detuning sweep in Hz
dfs = np.arange(-10e6, 10e6, 0.1e6)
# Idle time sweep (Must be a list of integers) - in clock cycles (4ns)
t_delay = np.arange(4, 300, 4)


with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    df = declare(int)  # QUA variable for the qubit detuning

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            # Update the qubits frequency
            for qubit in qubits:
                update_frequency(qubit.xy.name, df + qubit.xy.intermediate_frequency)

            with for_(*from_array(t, t_delay)):
                for qubit in qubits:
                    qubit.xy.play("x90")
                    qubit.xy.wait(t)
                    qubit.xy.play("x90")

                align()
                # QUA macro the readout the state of the active resonators (defined in macros.py)
                multiplexed_readout(qubits, I, I_st, Q, Q_st)
                # Wait for the qubits to decay to the ground state
                wait(machine.thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        for i in range(len(machine.active_qubits)):
            I_st[i].buffer(len(t_delay)).buffer(len(dfs)).average().save(f"I{i+1}")
            Q_st[i].buffer(len(t_delay)).buffer(len(dfs)).average().save(f"Q{i+1}")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ramsey)
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
        plt.suptitle("Ramsey chevron")
        for i, qubit in enumerate(qubits):
            axes[i, 0].cla()
            axes[i, 0].pcolor(4 * t_delay, dfs / u.MHz, I_volts[i])
            axes[i, 0].set_title(f"{qubit.name} - I, f_01={int(qubit.f_01 / u.MHz)} MHz")
            axes[i, 0].set_ylabel("detuning [MHz]")
            axes[i, 1].cla()
            axes[i, 1].pcolor(4 * t_delay, dfs / u.MHz, Q_volts[i])
            axes[i, 1].set_title(f"{qubit.name} - Q")
            axes[i, 1].set_xlabel("Idle time [ns]")
            axes[i, 1].set_ylabel("detuning [MHz]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {}
    for i, qubit in enumerate(qubits):
        data[f"{qubit.name}_time"] = t_delay * 4
        data[f"{qubit.name}_frequency"] = dfs + qubit.xy.intermediate_frequency
        data[f"{qubit.name}_I"] = np.abs(I_volts[i])
        data[f"{qubit.name}_Q"] = np.angle(Q_volts[i])
    data["figure"] = fig

    node_save("ramsey_chevron", data, machine)

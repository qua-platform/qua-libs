"""
RABI CHEVRON (AMPLITUDE VS FREQUENCY)
This sequence involves executing the qubit pulse (such as x180, square_pi, or other types) and measuring the state
of the resonator across various qubit intermediate frequencies and pulse amplitudes.
By analyzing the results, one can determine the qubit and estimate the x180 pulse amplitude for a specified duration.

Prerequisites:
    - Determination of the resonator's resonance frequency when coupled to the qubit of interest (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (be it an external mixer or an Octave port).
    - Identification of the approximate qubit frequency (referred to as "qubit_spectroscopy").
    - Configuration of the qubit frequency and the desired pi pulse duration (labeled as "pi_len").
    - Set the desired flux bias

Before proceeding to the next node:
    - Adjust the qubit frequency setting, labeled as "f_01", in the state.
    - Modify the qubit pulse amplitude setting, labeled as "pi_amp", in the state.
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
qubits = machine.qubits
num_qubits = len(qubits)

###################
# The QUA program #
###################

operation = "x180"  # The qubit operation to play
n_avg = 100  # The number of averages

# The frequency sweep with respect to the qubits resonance frequencies
dfs = np.arange(-100e6, +100e6, 1e6)
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(0.0, 1.9, 0.02)

with program() as rabi_chevron:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit detuning
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            # Update the qubit frequencies
            for qubit in qubits:
                update_frequency(qubit.xy.name, df + qubit.xy.intermediate_frequency)

            with for_(*from_array(a, amps)):
                # Play the qubit drives
                for qubit in qubits:
                    qubit.xy.play(operation, amplitude_scale=a)
                # Align all elements to measure after playing the qubit pulse.
                align()
                # QUA macro the readout the state of the active resonators (defined in macros.py)
                multiplexed_readout(qubits, I, I_st, Q, Q_st)
                # Wait for the qubit to decay to the ground state
                wait(machine.thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i+1}")
            Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i+1}")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, rabi_chevron, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rabi_chevron)
    # Get results from QUA program
    data_list = ["n"] + sum([[f"I{i+1}", f"Q{i+1}"] for i in range(num_qubits)], [])
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I = fetched_data[1::2]
        Q = fetched_data[2::2]
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot results
        plt.suptitle("Rabi chevron")
        I_volts, Q_volts = [], []
        for i, qubit in enumerate(qubits):
            # Convert results into Volts
            I_volts.append(u.demod2volts(I[i], qubit.resonator.operations["readout"].length))
            Q_volts.append(u.demod2volts(Q[i], qubit.resonator.operations["readout"].length))
            plt.subplot(2, num_qubits, i+1)
            plt.cla()
            plt.pcolor(amps * qubit.xy.operations[operation].amplitude, dfs / u.MHz, I_volts[i])
            plt.plot(qubit.xy.operations[operation].amplitude, 0, "r*")
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.ylabel("Qubit detuning [MHz]")
            plt.title(f"{qubit.name} (f_01: {int(qubit.f_01 / u.MHz)} MHz)")
            plt.subplot(2, num_qubits, i+num_qubits+1)
            plt.cla()
            plt.pcolor(amps * qubit.xy.operations[operation].amplitude, dfs / u.MHz, Q_volts[i])
            plt.plot(qubit.xy.operations[operation].amplitude, 0, "r*")
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.ylabel("Qubit detuning [MHz]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end to put all flux biases to 0 and prevent the fridge from heating up
    qm.close()

    # Save data from the node
    data = {}
    for i, qubit in enumerate(qubits):
        data[f"{qubit.name}_amplitude"] = amps * qubit.xy.operations[operation].amplitude
        data[f"{qubit.name}_frequency"] = dfs + qubit.xy.intermediate_frequency
        data[f"{qubit.name}_I"] = np.abs(I_volts[i])
        data[f"{qubit.name}_Q"] = np.angle(Q_volts[i])
    data["figure"] = fig

    node_save("rabi_chevron_amplitude", data, machine)

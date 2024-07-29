# %%
"""
        READOUT OPTIMISATION: FREQUENCY
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). This is done while varying the readout frequency.
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
optimal choice.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the readout frequency (f_opt) in the state.
    - Save the current state by calling machine.save("quam")
"""

from pathlib import Path

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
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
# Class containing tools to help handling units and conversions.
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
n_avg = 2  # The number of averages

# The frequency sweep parameters with respect to the resonators resonance frequencies
dfs = np.arange(-2e6, 2e6, 0.02e6)

with program() as ro_freq_opt:
    n = declare(int)
    I_g = [declare(fixed) for _ in range(num_qubits)]
    Q_g = [declare(fixed) for _ in range(num_qubits)]
    I_e = [declare(fixed) for _ in range(num_qubits)]
    Q_e = [declare(fixed) for _ in range(num_qubits)]
    DI = declare(fixed)
    DQ = declare(fixed)
    D = [declare(fixed) for _ in range(num_qubits)]
    df = declare(int)
    D_st = [declare_stream() for _ in range(num_qubits)]

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the resonator frequencies
            for qubit in qubits:
                update_frequency(qubit.resonator.name, df + qubit.resonator.intermediate_frequency)

            # Wait for the qubits to decay to the ground state
            wait(machine.thermalization_time * u.ns)
            align()
            # Measure the state of the resonators
            multiplexed_readout(qubits, I_g, None, Q_g, None)

            align()
            # Wait for thermalization again in case of measurement induced transitions
            wait(machine.thermalization_time * u.ns)
            # Play the x180 gate to put the qubits in the excited state
            for qubit in qubits:
                qubit.xy.play("x180")
            # Align the elements to measure after playing the qubit pulses.
            align()
            # Measure the state of the resonators
            multiplexed_readout(qubits, I_e, None, Q_e, None)

            # Derive the distance between the blobs for |g> and |e>
            for i in range(num_qubits):
                assign(DI, (I_e[i] - I_g[i]) * 100)
                assign(DQ, (Q_e[i] - Q_g[i]) * 100)
                assign(D[i], DI * DI + DQ * DQ)
                save(D[i], D_st[i])

    with stream_processing():
        for i in range(num_qubits):
            D_st[i].buffer(len(dfs)).average().save(f"D{i + 1}")

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ro_freq_opt, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ro_freq_opt)
    # Get results from QUA program

    result_keys = [f"D{i + 1}" for i in range(num_qubits)]
    results = fetching_tool(job, result_keys)
    D_data = results.fetch_all()

    # Plot the results
    fig, axes = plt.subplots(num_qubits, 1, figsize=(10, 4 * num_qubits))
    if num_qubits == 1:
        axes = [axes]

    for i, qubit in enumerate(qubits):
        axes[i].plot(dfs, D_data[i])
        axes[i].set_xlabel("Readout detuning [MHz]")
        axes[i].set_ylabel("Distance between IQ blobs [a.u.]")
        # axes[i].set_title(f"{qubit.name} - f_opt = {int(qubit.resonator.f_01 / u.MHz)} MHz")
        print(f"{qubit.resonator.name}: Shifting readout frequency by {dfs[np.argmax(D_data[i])]} Hz")

    plt.tight_layout()
    plt.show()

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {}
    for i, qubit in enumerate(qubits):
        data[f"{qubit.resonator.name}_frequency"] = dfs + qubit.resonator.intermediate_frequency
        data[f"{qubit.resonator.name}_D"] = D_data[i]
        data[f"{qubit.resonator.name}_if_opt"] = qubit.resonator.intermediate_frequency + dfs[np.argmax(D_data[i])]
        # Update the state
        qubit.resonator.intermediate_frequency += dfs[np.argmax(D_data[i])]

    data["figure"] = fig
    node_save(machine, "readout_frequency_optimization", data, additional_files=True)

# %%

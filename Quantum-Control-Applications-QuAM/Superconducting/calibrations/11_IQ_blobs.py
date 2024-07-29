# %%
"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e threshold (ge_threshold) in the state.
    - Save the current state by calling machine.save("quam")
"""

from pathlib import Path

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.analysis.discriminator import two_state_discriminator
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
n_runs = 10000  # Number of runs

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, _ = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_runs, n + 1):
        # ground iq blobs for all qubits
        wait(machine.thermalization_time * u.ns)
        align()
        multiplexed_readout(qubits, I_g, I_g_st, Q_g, Q_g_st)

        align()
        wait(machine.thermalization_time * u.ns)
        for qubit in qubits:
            # excited iq blobs for all qubits
            qubit.xy.play("x180")
        align()
        multiplexed_readout(qubits, I_e, I_e_st, Q_e, Q_e_st)

    with stream_processing():
        for i in range(num_qubits):
            I_g_st[i].save_all(f"I_g_q{i}")
            Q_g_st[i].save_all(f"Q_g_q{i}")
            I_e_st[i].save_all(f"I_e_q{i}")
            Q_e_st[i].save_all(f"Q_e_q{i}")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(iq_blobs)
    # Fetch data
    data_list = sum(
        [[f"I_g_q{i}", f"Q_g_q{i}", f"I_e_q{i}", f"Q_e_q{i}"] for i in range(num_qubits)],
        [],
    )
    results = fetching_tool(job, data_list)
    fetched_data = results.fetch_all()
    I_g_data = fetched_data[1::2]
    Q_g_data = fetched_data[2::2]
    I_e_data = fetched_data[3::2]
    Q_e_data = fetched_data[4::2]
    # Prepare for save data
    data = {}
    # Plot the results
    figs = []
    for i, qubit in enumerate(qubits):
        I_g = I_g_data[i]
        Q_g = Q_g_data[i]
        I_e = I_e_data[i]
        Q_e = Q_e_data[i]

        hist = np.histogram(I_g, bins=100)
        rus_threshold = hist[1][1:][np.argmax(hist[0])]
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(I_g, Q_g, I_e, Q_e, True, b_plot=True)

        plt.suptitle(f"{qubit.name} - IQ Blobs")
        plt.axvline(rus_threshold, color="k", linestyle="--", label="Threshold")
        figs.append(plt.gcf())

        data[f"{qubit.name}_I_g"] = I_g
        data[f"{qubit.name}_Q_g"] = Q_g
        data[f"{qubit.name}_I_e"] = I_e
        data[f"{qubit.name}_Q_e"] = Q_e
        data[f"{qubit.name}"] = {
            "angle": angle,
            "threshold": threshold,
            "rus_exit_threshold": rus_threshold,
            "fidelity": fidelity,
            "confusion_matrix": [[gg, ge], [eg, ee]],
        }
        data[f"{qubit.name}_figure"] = figs[i]

        qubit.resonator.operations["readout"].integration_weights_angle -= angle
        qubit.resonator.operations["readout"].threshold = threshold
        qubit.resonator.operations["readout"].rus_exit_threshold = rus_threshold

    qm.close()

    node_save(machine, "iq_blobs", data, additional_files=True)

# %%

# %%
"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly without having to modify the configuration.

The data is post-processed to determine the qubit resonance frequency, which can then be used to adjust
the qubit intermediate frequency in the configuration under "center".

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

This step can be repeated using the "x180" operation instead of "saturation" to adjust the pulse parameters (amplitude,
duration, frequency) before performing the next calibration steps.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Set the flux bias to the minimum frequency point, labeled as "max_frequency_point", in the state.
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.
    - Specification of the expected qubit T1 in the state.

Before proceeding to the next node:
    - Update the qubit frequency, labeled as f_01, in the state.
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

operation = "saturation"  # The qubit operation to play, can be switched to "x180" when the qubits are found.
n_avg = 600  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
saturation_len = 20 * u.us  # In ns
saturation_amp = 0.01  # pre-factor to the value defined in the config - restricted to [-2; 2)
# Qubit detuning sweep with respect to their resonance frequencies
dfs = np.arange(-50e6, +50e6, 0.1e6)

with program() as multi_qubit_spec:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            # Update the qubit frequencies for all qubits
            for q in qubits:
                update_frequency(q.xy.name, df + q.xy.intermediate_frequency)

            for q in qubits:
                # Play the saturation pulse
                q.xy.play(
                    operation,
                    amplitude_scale=saturation_amp,
                    duration=saturation_len * u.ns,
                )
                align(q.xy.name, q.resonator.name)

            # QUA macro the readout the state of the active resonators (defined in macros.py)
            multiplexed_readout(qubits, I, I_st, Q, Q_st, sequential=False)
            # Wait for the qubit to decay to the ground state
            wait(machine.thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(multi_qubit_spec)
    # Get results from QUA program
    data_list = ["n"] + sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_qubits)], [])
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I = fetched_data[1::2]
        Q = fetched_data[2::2]

        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.suptitle("Qubit spectroscopy")
        s_data = []
        for i, q in enumerate(qubits):
            s = u.demod2volts(I[i] + 1j * Q[i], q.resonator.operations["readout"].length)
            s_data.append(s)
            plt.subplot(2, num_qubits, i + 1)
            plt.cla()
            plt.plot(
                (q.xy.LO_frequency + q.xy.intermediate_frequency + dfs) / u.MHz,
                np.abs(s),
            )
            plt.grid(True)
            plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
            plt.title(f"{q.name} (f_01: {q.xy.rf_frequency / u.MHz} MHz)")
            plt.subplot(2, num_qubits, num_qubits + i + 1)
            plt.cla()
            plt.plot(
                (q.xy.LO_frequency + q.xy.intermediate_frequency + dfs) / u.MHz,
                np.unwrap(np.angle(s)),
            )
            plt.grid(True)
            plt.ylabel("Phase [rad]")
            plt.xlabel(f"{q.name} detuning [MHz]")
            plt.plot((q.xy.LO_frequency + q.xy.intermediate_frequency) / u.MHz, 0.0, "r*")

        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {}
    for i, q in enumerate(qubits):
        data[f"{q.name}_frequency"] = dfs + q.xy.intermediate_frequency
        data[f"{q.name}_R"] = np.abs(s_data[i])
        data[f"{q.name}_phase"] = np.angle(s_data[i])
    data["figure"] = fig

    fig_analysis = plt.figure()
    plt.suptitle("Qubit spectroscopy")
    # Fit the results to extract the resonance frequency
    for i, q in enumerate(qubits):
        try:
            from qualang_tools.plot.fitting import Fit

            fit = Fit()
            plt.subplot(1, num_qubits, i + 1)
            res = fit.reflection_resonator_spectroscopy(
                (q.xy.LO_frequency + q.xy.intermediate_frequency + dfs) / u.MHz,
                -np.unwrap(np.angle(s_data[i])),
                plot=True,
            )
            plt.legend((f"f = {res['f'][0]:.3f} MHz",))
            plt.xlabel(f"{q.name} IF [MHz]")
            plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
            plt.title(f"{q.name}")

            # q.xy.intermediate_frequency = int(res["f"][0] * u.MHz)
            data[f"{q.name}"] = {
                "res_if": q.xy.intermediate_frequency,
                "fit_successful": True,
            }

            plt.tight_layout()
            data["fit_figure"] = fig_analysis

        except Exception:
            data[f"{q.name}"] = {"successful_fit": False}
            pass

    plt.show()
    # additional files
    # Save data from the node
    node_save(machine, "qubit_spectroscopy", data, additional_files=True)

# %%

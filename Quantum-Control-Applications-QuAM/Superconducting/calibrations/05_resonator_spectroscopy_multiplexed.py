# %%
"""
        RESONATOR SPECTROSCOPY MULTIPLEXED
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies for all resonators simultaneously.
The data is then post-processed to determine the resonator resonance frequency.
This frequency can be used to update the readout frequency in the state.

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the IQ mixer connected to the readout line (whether it's an external mixer or an Octave port).
    - Define the readout pulse amplitude and duration in the state.
    - Specify the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, labeled as f_res and f_opt, in the state for all resonators.
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
from scipy import signal

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
resonators = [qubit.resonator for qubit in machine.active_qubits]
num_qubits = len(qubits)
num_resonators = len(resonators)

###################
# The QUA program #
###################

n_avg = 400  # The number of averages
# The frequency sweep around the resonator resonance frequency f_opt
dfs = np.arange(-5e6, +5e6, 0.1e6)
# You can adjust the IF frequency here to manually adjust the resonator frequencies instead of updating the state
# rr1.intermediate_frequency = -50 * u.MHz
# rr2.intermediate_frequency = 50 * u.MHz


with program() as multi_res_spec:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the readout frequency

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            for i, rr in enumerate(resonators):
                # Update the resonator frequencies for all resonators
                update_frequency(rr.name, df + rr.intermediate_frequency)

                rr.measure("readout", qua_vars=(I[i], Q[i]))

                # wait for the resonator to relax
                rr.wait(machine.depletion_time * u.ns)

                # save data
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])

    with stream_processing():
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, multi_res_spec, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(multi_res_spec)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    data_list = sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_qubits)], [])
    results = fetching_tool(job, data_list, mode="live")
    # Prepare the figures for live plotting
    fig, axss = plt.subplots(2, num_qubits, figsize=(4 * num_qubits, 5))
    interrupt_on_close(fig, job)
    # Live plotting
    s_data = []
    while results.is_processing():
        # Fetch results
        data = results.fetch_all()
        for i in range(num_qubits):
            I, Q = data[2 * i : 2 * i + 2]
            rr = resonators[i]
            # Data analysis
            s_data.append(u.demod2volts(I + 1j * Q, rr.operations["readout"].length))
            # Plot
            plt.sca(axss[0, i])
            plt.suptitle("Multiplexed resonator spectroscopy")
            plt.cla()
            plt.plot(
                (rr.LO_frequency + rr.intermediate_frequency) / u.MHz + dfs / u.MHz,
                np.abs(s_data[-1]),
                ".",
            )
            plt.title(f"{rr.name}")
            plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
            plt.sca(axss[1, i])
            plt.cla()
            plt.plot(
                (rr.LO_frequency + rr.intermediate_frequency) / u.MHz + dfs / u.MHz,
                signal.detrend(np.unwrap(np.angle(s_data[-1]))),
                ".",
            )
            plt.ylabel("Phase [rad]")
            plt.xlabel("Readout frequency [MHz]")
            plt.tight_layout()
            plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {"figure_raw": fig}
    for rr, s in zip(resonators, s_data):
        data[f"{rr.name}_frequencies"] = rr.intermediate_frequency + dfs
        data[f"{rr.name}_S"] = s
        data[f"{rr.name}_R"] = np.abs(s)
        data[f"{rr.name}_phase"] = signal.detrend(np.unwrap(np.angle(s)))

    fig_analysis, axs = plt.subplots(1, num_qubits, figsize=(4 * num_qubits, 5))
    plt.suptitle("Multiplexed resonator spectroscopy")
    for i, (ax, rr, s) in enumerate(zip(axs, resonators, s_data)):
        try:
            from qualang_tools.plot.fitting import Fit

            fit = Fit()
            plt.sca(ax)
            res_1 = fit.reflection_resonator_spectroscopy(
                (rr.intermediate_frequency + dfs) / u.MHz, np.abs(s), plot=True
            )
            plt.xlabel(f"{rr.name} IF [MHz]")
            plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
            plt.title(f"{rr.name}")
            intermediate_frequency = int(res_1["f"][0] * u.MHz)
            rr.RF_frequency = rr.LO_frequency + intermediate_frequency
            plt.legend((f"fr = {rr.RF_frequency:.3f} MHz",))

            data[f"{rr.name}"] = {
                "resonator_frequency": int(rr.RF_frequency),
                "successful_fit": True,
            }
            data[f"figure_fit_{rr.name}"] = fig_analysis
        except (Exception,):
            data[f"{rr.name}"] = {"successful_fit": False}
            pass

    plt.show()
    # Save data from the node
    node_save(machine, "resonator_spectroscopy_multiplexed", data, additional_files=True)

# %%

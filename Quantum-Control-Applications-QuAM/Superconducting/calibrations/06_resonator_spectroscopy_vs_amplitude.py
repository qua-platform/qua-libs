"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures for all resonators simultaneously.
This is done across various readout intermediate dfs and amplitudes.
Based on the results, one can determine if a qubit is coupled to the resonator by noting the resonator frequency
splitting. This information can then be used to adjust the readout amplitude, choosing a readout amplitude value
just before the observed frequency splitting.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude (the pulse processor will sweep up to twice this value) and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "f_res" and "f_opt", in the state.
    - Adjust the readout amplitude, labeled as "readout_pulse_amp", in the state.
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig

from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np
import os

from quam_components import QuAM
from macros import qua_declaration, node_save

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
resonators = [qb.resonator for qb in machine.active_qubits]
num_resonators = len(resonators)
prev_amps = [rr.operations["readout"].amplitude for rr in resonators]

###################
# The QUA program #
###################

n_avg = 100  # The number of averages

# Initial readout amplitude for all resonators
for rr in resonators:
    rr.operations["readout"].amplitude = 0.01

# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
amps = np.arange(0.05, 1.99, 0.01)
# The frequency sweep around the resonator resonance frequencies f_opt
dfs = np.arange(-10e6, +10e6, 0.1e6)

with program() as multi_res_spec_vs_amp:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=num_resonators)
    a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
    df = declare(int)  # QUA variable for the readout frequency

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        save(n, n_st)

        with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
            # Update the resonator frequencies for all resonators
            for i, rr in enumerate(resonators):
                update_frequency(rr.name, df + rr.intermediate_frequency)

            with for_(*from_array(a, amps)):  # QUA for_ loop for sweeping the readout amplitude
                for i, rr in enumerate(resonators):
                    rr.wait(machine.depletion_time * u.ns)  # wait for the resonator to relax
                    rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

    with stream_processing():
        n_st.save("n")
        for i in range(num_resonators):
            I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i + 1}")

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_amp, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(multi_res_spec_vs_amp)
    # Prepare the figures for live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    res_list = ["n"] + sum([[f"I{i}", f"Q{i}"] for i in range(num_resonators)], [])
    results = fetching_tool(job, res_list, mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I_data = fetched_data[1::2]
        Q_data = fetched_data[2::2]

        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.suptitle("Resonator spectroscopy vs amplitude")
        A_data = []
        for i, rr in enumerate(resonators):
            s = u.demod2volts(I_data[i] + 1j * Q_data[i], rr.operations["readout"].length)
            A = np.abs(s)
            # Normalize data
            row_sums = A.sum(axis=0)
            A = A / row_sums[np.newaxis, :]
            A_data.append(A)
            # Plot
            plt.subplot(1, num_resonators, i + 1)
            plt.cla()
            plt.title(f"{rr.name} - f_cent: {int(rr.rf_frequency / u.MHz)} MHz")
            plt.xlabel("Readout amplitude [V]")
            plt.ylabel("Readout detuning [MHz]")
            plt.pcolor(amps * rr.operations["readout"].amplitude, dfs / u.MHz, A)
            plt.axhline(0, color="k", linestyle="--")
            plt.axvline(prev_amps[i], color="k", linestyle="--")

        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {}
    for i, rr in enumerate(resonators):
        data[f"{rr.name}_amplitude"] = amps * rr.operations["readout"].amplitude
        data[f"{rr.name}_frequency"] = dfs + rr.intermediate_frequency
        data[f"{rr.name}_R"] = A_data[i]
        data[f"{rr.name}_readout_amplitude"] = prev_amps[i]
    data["figure"] = fig

    node_save("resonator_spectroscopy_vs_amplitude", data, machine)

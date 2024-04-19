"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures for both resonators simultaneously.
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

from components import QuAM
from macros import qua_declaration, node_save


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("state.json")
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
rr1 = machine.active_qubits[0].resonator
rr2 = machine.active_qubits[1].resonator
prev_amp1 = rr1.operations["readout"].amplitude
prev_amp2 = rr2.operations["readout"].amplitude

###################
# The QUA program #
###################

n_avg = 100  # The number of averages

rr1.operations["readout"].amplitude = 0.01
rr2.operations["readout"].amplitude = 0.01

# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
amps = np.arange(0.05, 1.99, 0.01)
# The frequency sweep around the resonator resonance frequencies f_opt
dfs = np.arange(-10e6, +10e6, 0.1e6)


with program() as multi_res_spec_vs_amp:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
    df = declare(int)  # QUA variable for the readout frequency

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        save(n, n_st)

        with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
            # Update the resonator frequencies
            update_frequency(rr1.name, df + rr1.intermediate_frequency)
            update_frequency(rr2.name, df + rr2.intermediate_frequency)

            with for_(*from_array(a, amps)):  # QUA for_ loop for sweeping the readout amplitude
                # resonator 1
                rr1.wait(machine.get_depletion_time * u.ns)  # wait for the resonator to relax
                rr1.measure("readout", qua_vars=(I[0], Q[0]), amplitude_scale=a)
                save(I[0], I_st[0])
                save(Q[0], Q_st[0])

                ## rr2.align(rr1.name)  # Uncomment to measure sequentially and avoid overflow

                # resonator 2
                rr2.wait(machine.get_depletion_time * u.ns)  # wait for the resonator to relax
                rr2.measure("readout", qua_vars=(I[1], Q[1]), amplitude_scale=a)
                save(I[1], I_st[1])
                save(Q[1], Q_st[1])

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(amps)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(amps)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amps)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(amps)).buffer(len(dfs)).average().save("Q2")

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
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Data analysis
        s1 = u.demod2volts(I1 + 1j * Q1, rr1.operations["readout"].length)
        s2 = u.demod2volts(I2 + 1j * Q2, rr2.operations["readout"].length)

        A1 = np.abs(s1)
        A2 = np.abs(s2)
        # Normalize data
        row_sums = A1.sum(axis=0)
        A1 = A1 / row_sums[np.newaxis, :]
        row_sums = A2.sum(axis=0)
        A2 = A2 / row_sums[np.newaxis, :]
        # Plot
        plt.suptitle("Resonator spectroscopy vs amplitude")
        plt.subplot(121)
        plt.cla()
        plt.title(f"{rr1.name} - f_cent: {int(rr1.rf_frequency / u.MHz)} MHz")
        plt.xlabel("Readout amplitude [V]")
        plt.ylabel("Readout detuning [MHz]")
        plt.pcolor(amps * rr1.operations["readout"].amplitude, dfs / u.MHz, A1)
        plt.axhline(0, color="k", linestyle="--")
        plt.axvline(prev_amp1, color="k", linestyle="--")
        plt.subplot(122)
        plt.cla()
        plt.title(f"{rr2.name} - f_cent: {int(rr2.rf_frequency / u.MHz)} MHz")
        plt.xlabel("Readout amplitude [V]")
        plt.ylabel("Readout detuning [MHz]")
        plt.pcolor(amps * rr2.operations["readout"].amplitude, dfs / u.MHz, A2)
        plt.axhline(0, color="k", linestyle="--")
        plt.axvline(prev_amp2, color="k", linestyle="--")
        plt.tight_layout()

        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # rr1.operations["readout"].amplitude =
    # rr2.operations["readout"].amplitude =
    # Save data from the node
    data = {
        f"{rr1.name}_amplitude": amps * rr1.operations["readout"].amplitude,
        f"{rr1.name}_frequency": dfs + rr1.intermediate_frequency,
        f"{rr1.name}_R": A1,
        f"{rr1.name}_readout_amplitude": prev_amp1,
        f"{rr2.name}_amplitude": amps * rr2.operations["readout"].amplitude,
        f"{rr2.name}_frequency": dfs + rr2.intermediate_frequency,
        f"{rr2.name}_R": A2,
        f"{rr2.name}_readout_amplitude": prev_amp2,
        "figure": fig,
    }
    node_save("resonator_spectroscopy_vs_amplitude", data, machine)

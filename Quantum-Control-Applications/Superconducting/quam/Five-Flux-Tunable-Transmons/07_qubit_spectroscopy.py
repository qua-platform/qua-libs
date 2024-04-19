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

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np

from components import QuAM
from macros import qua_declaration, multiplexed_readout

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("quam")
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
q1 = machine.active_qubits[0]
q2 = machine.active_qubits[1]

###################
# The QUA program #
###################

operation = "saturation"  # The qubit operation to play, can be switched to "x180" when the qubits are found.
n_avg = 100  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
saturation_len = 10 * u.us  # In ns
saturation_amp = 0.5  # pre-factor to the value defined in the config - restricted to [-2; 2)
# Qubit detuning sweep with respect to their resonance frequencies
dfs = np.arange(-60e6, +80e6, 1e6)

with program() as multi_qubit_spec:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the qubit frequency

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            # Update the qubit frequencies
            update_frequency(q1.xy.name, df + q1.xy.intermediate_frequency)
            update_frequency(q2.xy.name, df + q2.xy.intermediate_frequency)

            # qubit 1
            q1.xy.play(operation, amplitude_scale=saturation_amp, duration=saturation_len * u.ns)
            align(q1.xy.name, q1.resonator.name)  # TODO: use q1.align() instead?
            # qubit 2
            q2.xy.play(operation, amplitude_scale=saturation_amp, duration=saturation_len * u.ns)
            align(q2.xy.name, q2.resonator.name)

            # QUA macro the readout the state of the active resonators (defined in macros.py)
            multiplexed_readout(machine, I, I_st, Q, Q_st, sequential=False)
            # Wait for the qubit to decay to the ground state
            wait(machine.get_thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(dfs)).average().save("Q2")


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
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(multi_qubit_spec)
    # Get results from QUA program
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Convert results into Volts
        s1 = u.demod2volts(I1 + 1j * Q1, q1.resonator.operations["readout"].length)
        s2 = u.demod2volts(I2 + 1j * Q2, q2.resonator.operations["readout"].length)
        # Plot results
        plt.suptitle("Qubit spectroscopy")
        plt.subplot(221)
        plt.cla()
        plt.plot(dfs / u.MHz, np.abs(s1))
        plt.grid(True)
        plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
        plt.title(f"{q1.name} (f_01: {q1.xy.rf_frequency / u.MHz} MHz)")
        plt.subplot(223)
        plt.cla()
        plt.plot(dfs / u.MHz, np.angle(s1))
        plt.grid(True)
        plt.ylabel("Phase [rad]")
        plt.xlabel(f"{q1.name} detuning [MHz]")
        plt.subplot(222)
        plt.cla()
        plt.plot(dfs / u.MHz, np.abs(s2))
        plt.grid(True)
        plt.title(f"{q2.name} (f_01: {q2.xy.rf_frequency / u.MHz} MHz)")
        plt.subplot(224)
        plt.cla()
        plt.plot(dfs / u.MHz, np.angle(s2))
        plt.grid(True)
        plt.xlabel(f"{q2.name} detuning [MHz]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Fit the results to extract the resonance frequency
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        plt.suptitle("Qubit spectroscopy")
        plt.subplot(121)
        res_1 = fit.reflection_resonator_spectroscopy(
            (q1.xy.intermediate_frequency + dfs) / u.MHz, -np.angle(s1), plot=True
        )
        plt.legend((f"f = {res_1['f'][0]:.3f} MHz",))
        plt.xlabel(f"{q1.name} IF [MHz]")
        plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
        plt.title(f"{q1.name}")
        plt.subplot(122)
        res_2 = fit.reflection_resonator_spectroscopy(
            (q2.xy.intermediate_frequency + dfs) / u.MHz, np.abs(s2), plot=True
        )
        plt.legend((f"f = {res_2['f'][0]:.3f} MHz",))
        plt.xlabel(f"{q2.name} IF [MHz]")
        plt.title(f"{q2.name}")
        plt.tight_layout()

        q1.xy.intermediate_frequency = res_1["f"][0] * u.MHz
        q2.xy.intermediate_frequency = res_2["f"][0] * u.MHz
        # machine.save("quam")

    except (Exception,):
        pass

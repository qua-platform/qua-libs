"""
        QUBIT SPECTROSCOPY VERSUS FLUX
This sequence involves doing a qubit spectroscopy for several flux biases in order to exhibit the qubit frequency
versus flux response.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Identification of the approximate qubit frequency ("qubit_spectroscopy").

Before proceeding to the next node:
    - Update the qubit frequency, labeled as "f_01", in the state.
    - Update the relevant flux points in the state.
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

from components import QuAM
from macros import qua_declaration, multiplexed_readout, node_save

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
q1 = machine.active_qubits[0]
q2 = machine.active_qubits[1]

###################
# The QUA program #
###################

operation = "saturation"  # The qubit operation to play, can be switched to "x180" when the qubits are found.
n_avg = 100  # Number of averaging loops
cooldown_time = max(q1.thermalization_time, q2.thermalization_time)

# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
saturation_len = 10 * u.us  # In ns
saturation_amp = 0.5  # pre-factor to the value defined in the config - restricted to [-2; 2)

# Qubit detuning sweep with respect to their resonance frequencies
dfs = np.arange(-50e6, 100e6, 0.1e6)
# Flux bias sweep
dcs = np.linspace(-0.05, 0.05, 40)

# Adjust the qubits IFs locally to help find the qubits
# q1.xy.intermediate_frequency = 340e6
# q2.xy.intermediate_frequency = 0

with program() as multi_qubit_spec_vs_flux:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the qubit frequency
    dc = declare(fixed)  # QUA variable for the flux dc level

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            # Update the qubit frequencies
            update_frequency(q1.xy.name, df + q1.xy.intermediate_frequency)
            update_frequency(q2.xy.name, df + q2.xy.intermediate_frequency)
            with for_(*from_array(dc, dcs)):
                # Flux sweeping
                q1.z.set_dc_offset(dc)
                q2.z.set_dc_offset(dc)
                wait(100)  # Wait for the flux to settle

                # Saturate qubit
                q1.xy.play(operation, amplitude_scale=saturation_amp, duration=saturation_len * u.ns)
                q2.xy.play(operation, amplitude_scale=saturation_amp, duration=saturation_len * u.ns)

                # QUA macro the readout the state of the active resonators (defined in macros.py)
                multiplexed_readout(machine, I, I_st, Q, Q_st)
                # Wait for the qubit to decay to the ground state
                wait(machine.get_thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(dcs)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(dcs)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dcs)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(dcs)).buffer(len(dfs)).average().save("Q2")


#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_qubit_spec_vs_flux, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_active_qubits(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(multi_qubit_spec_vs_flux)
    # Get results from QUA program
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Convert results into Volts
        s1 = u.demod2volts(I1 + 1j * Q1, q1.resonator.operations["readout"].length)
        s2 = u.demod2volts(I2 + 1j * Q2, q2.resonator.operations["readout"].length)
        # 2D spectroscopy plot
        plt.suptitle("Qubit spectroscopy vs flux")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(dcs, (q1.xy.intermediate_frequency + dfs) / u.MHz, np.abs(s1))
        plt.plot(q1.z.min_offset, q1.xy.intermediate_frequency / u.MHz, "r*")
        plt.xlabel("Flux [V]")
        plt.ylabel(f"{q1.name} IF [MHz]")
        plt.title(f"{q1.name} (f_01: {int(q1.f_01 / u.MHz)} MHz)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(dcs, (q1.xy.intermediate_frequency + dfs) / u.MHz, np.unwrap(np.angle(s1)))
        plt.plot(q1.z.min_offset, q1.xy.intermediate_frequency / u.MHz, "r*")
        plt.xlabel("Flux [V]")
        plt.ylabel(f"{q1.name} IF [MHz]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(dcs, (q2.xy.intermediate_frequency + dfs) / u.MHz, np.abs(s2))
        plt.plot(q2.z.min_offset, q2.xy.intermediate_frequency / u.MHz, "r*")
        plt.title(f"{q2.name} (f_01: {int(q2.f_01 / u.MHz)} MHz)")
        plt.ylabel(f"{q2.name} IF [MHz]")
        plt.xlabel("flux [V]")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(dcs, (q2.xy.intermediate_frequency + dfs) / u.MHz, np.unwrap(np.angle(s2)))
        plt.plot(q2.z.min_offset, q2.xy.intermediate_frequency / u.MHz, "r*")
        plt.xlabel("Flux [V]")
        plt.ylabel(f"{q2.name} IF [MHz]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Set the relevant flux points
    # q1.z.min_offset =
    # q1.z.min_offset =

    # Save data from the node
    data = {
        f"{q1.name}_flux_bias": dcs,
        f"{q1.name}_frequency": dfs + q1.xy.intermediate_frequency,
        f"{q1.name}_R": np.abs(s1),
        f"{q1.name}_phase": np.angle(s1),
        f"{q1.name}_min_offset": q1.z.min_offset,
        f"{q2.name}_flux_bias": dcs,
        f"{q2.name}_frequency": dfs + q2.xy.intermediate_frequency,
        f"{q2.name}_R": np.abs(s2),
        f"{q2.name}_phase": np.angle(s2),
        f"{q2.name}_min_offset": q2.z.min_offset,
        "figure": fig,
    }
    node_save("qubit_spectroscopy_vs_flux", data, machine)

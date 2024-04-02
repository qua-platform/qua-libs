"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the configuration.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Adjust the flux bias to the maximum frequency point, labeled as "max_frequency_point", in the state.
    - Adjust the flux bias to the minimum frequency point, labeled as "min_frequency_point", in the state.
    - Save the current state by calling machine._save("current_state.json")
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

from quam.examples.superconducting_qubits.components import QuAM
from macros import qua_declaration, apply_all_flux_to_min, multiplexed_readout


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class t handle unit and conversion functions
u = unit(coerce_to_integer=True)
# Instantiate the abstract machine
machine = QuAM.load("quam")
# Load the config
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
# Open the Quantum Machine Manager
qmm = QuantumMachinesManager(host="172.16.33.101", cluster_name="Cluster_81", octave=octave_config)


###################
# The QUA program #
###################
q1 = machine.active_qubits[0]
q2 = machine.active_qubits[1]

n_avg = 200  # Number of averaging loops
depletion_time = 5 * max(q1.resonator.depletion_time, q2.resonator.depletion_time)
# Flux bias sweep in V
dcs = np.linspace(-0.49, 0.49, 50)
# The frequency sweep around the resonator resonance frequency f_opt
dfs = np.arange(-50e6, 5e6, 0.1e6)
# You can adjust the IF frequency here to manually adjust the resonator frequencies instead of updating the state
# res_if_1 = 244e6
# res_if_2 = 205e6

with program() as multi_res_spec_vs_flux:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    dc = declare(fixed)  # QUA variable for the flux bias
    df = declare(int)  # QUA variable for the readout frequency

    # Bring the active qubits to the maximum frequency point
    apply_all_flux_to_min(machine)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            # Update the resonator frequencies
            update_frequency(q1.resonator.name, df + q1.resonator.intermediate_frequency)
            update_frequency(q2.resonator.name, df + q2.resonator.intermediate_frequency)

            with for_(*from_array(dc, dcs)):
                # Flux sweeping by tuning the OPX dc offset associated to the flux_line element
                set_dc_offset(q1.z.name, "single", dc)
                set_dc_offset(q2.z.name, "single", dc)
                wait(100)  # Wait for the flux to settle
                # QUA macro the readout the state of the active resonators (defined in macros.py)
                multiplexed_readout(machine, I, I_st, Q, Q_st, sequential=False)
                # wait for the resonators to relax
                wait(depletion_time * u.ns, q1.resonator.name, q2.resonator.name)

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
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(multi_res_spec_vs_flux)
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
        # Convert results into Volts and normalize
        s1 = u.demod2volts(I1 + 1j * Q1, q1.resonator.operations["readout"].length)
        s2 = u.demod2volts(I2 + 1j * Q2, q2.resonator.operations["readout"].length)

        A1 = np.abs(s1)
        A2 = np.abs(s2)

        plt.suptitle("Resonator specrtoscopy vs flux")
        plt.subplot(121)
        plt.cla()
        plt.title(f"{q1.resonator.name} (LO: {q1.resonator.frequency_converter_up.LO_frequency / u.MHz} MHz)")
        plt.xlabel("flux [V]")
        plt.ylabel(f"{q1.resonator.name} IF [MHz]")
        plt.pcolor(dcs, q1.resonator.intermediate_frequency / u.MHz + dfs / u.MHz, A1)
        plt.plot(q1.z.min_offset, q1.resonator.intermediate_frequency / u.MHz, "r*")
        plt.subplot(122)
        plt.cla()
        plt.title(f"{q2.resonator.name} (LO: {q2.resonator.frequency_converter_up.LO_frequency / u.MHz} MHz)")
        plt.xlabel("flux [V]")
        plt.ylabel(f"{q2.resonator.name} IF [MHz]")
        plt.pcolor(dcs, q2.resonator.intermediate_frequency / u.MHz + dfs / u.MHz, A2)
        plt.plot(q2.z.min_offset, q2.resonator.intermediate_frequency / u.MHz, "r*")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

# Update machine with max frequency point for both resonator and qubit
# qb1.z.max_frequency_point =
# qb2.z.max_frequency_point =
# machine._save("current_state.json")

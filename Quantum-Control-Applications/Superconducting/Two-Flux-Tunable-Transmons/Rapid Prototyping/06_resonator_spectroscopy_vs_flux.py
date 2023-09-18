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
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from macros import qua_declaration, multiplexed_readout
import warnings

warnings.filterwarnings("ignore")

#######################################################
# Get the config from the machine in configuration.py #
#######################################################
# Build the config
config = build_config(machine)

# Get the resonators frequencies
res_if_1 = rr1.f_res - machine.local_oscillators.readout[rr1.LO_index].freq
res_if_2 = rr2.f_res - machine.local_oscillators.readout[rr2.LO_index].freq

###################
# The QUA program #
###################
n_avg = 200  # Number of averaging loops
depletion_time = 5 * max(rr1.depletion_time, rr2.depletion_time)
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
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            # Update the resonator frequencies
            update_frequency(rr1.name, df + res_if_1)
            update_frequency(rr2.name, df + res_if_2)
            with for_(*from_array(dc, dcs)):
                # Flux sweeping by tuning the OPX dc offset associated to the flux_line element
                set_dc_offset(q1_z, "single", dc)
                set_dc_offset(q2_z, "single", dc)
                wait(100)  # Wait for the flux to settle
                # QUA macro the readout the state of the active resonators (defined in macros.py)
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, sequential=False)
                # wait for the resonators to relax
                wait(depletion_time * u.ns, rr1.name, rr2.name)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(dcs)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(dcs)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dcs)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(dcs)).buffer(len(dfs)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

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
        s1 = u.demod2volts(I1 + 1j * Q1, machine.resonators[0].readout_pulse_length)
        s2 = u.demod2volts(I2 + 1j * Q2, machine.resonators[0].readout_pulse_length)

        A1 = np.abs(s1)
        A2 = np.abs(s2)

        plt.suptitle("Resonator specrtoscopy vs flux")
        plt.subplot(121)
        plt.cla()
        plt.title(f"{rr1.name} (LO: {machine.local_oscillators.readout[0].freq / u.MHz} MHz)")
        plt.xlabel("flux [V]")
        plt.ylabel(f"{rr1.name} IF [MHz]")
        plt.pcolor(dcs, res_if_1 / u.MHz + dfs / u.MHz, A1)
        plt.plot(machine.qubits[active_qubits[0]].z.max_frequency_point, res_if_1 / u.MHz, "r*")
        plt.subplot(122)
        plt.cla()
        plt.title(f"{rr2.name} (LO: {machine.local_oscillators.readout[0].freq / u.MHz} MHz)")
        plt.xlabel("flux [V]")
        plt.ylabel(f"{rr2.name} IF [MHz]")
        plt.pcolor(dcs, res_if_2 / u.MHz + dfs / u.MHz, A2)
        plt.plot(machine.qubits[active_qubits[1]].z.max_frequency_point, res_if_2 / u.MHz, "r*")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

# Update machine with max frequency point for both resonator and qubit
# qb1.z.max_frequency_point =
# qb2.z.max_frequency_point =
# machine._save("current_state.json")

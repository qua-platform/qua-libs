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
    - Save the current state by calling machine._save("current_state.json")
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from macros import qua_declaration, multiplexed_readout
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

#######################################################
# Get the config from the machine in configuration.py #
#######################################################

# Get the qubit frequencies (IFs and LOs)
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq
qb_if_1 = qb1.xy.f_01 - lo1
qb_if_2 = qb2.xy.f_01 - lo2

# Build the config
config = build_config(machine)

###################
# The QUA program #
###################
n_avg = 100  # Number of averaging loops
cooldown_time = 5 * max(qb1.T1, qb2.T1)

# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
saturation_len = 10 * u.us  # In ns
saturation_amp = 0.5  # pre-factor to the value defined in the config - restricted to [-2; 2)

# Qubit detuning sweep with respect to their resonance frequencies
dfs = np.arange(-50e6, 100e6, 0.1e6)
# Flux bias sweep
dcs = np.linspace(-0.05, 0.05, 40)

# Adjust the qubits IFs locally to help find the qubits
# qb_if_1 = 340e6
# qb_if_2 = 0

with program() as multi_qubit_spec_vs_flux:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the qubit frequency
    dc = declare(fixed)  # QUA variable for the flux dc level

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            # Update the qubit frequencies
            update_frequency(qb1.name + "_xy", df + qb_if_1)
            update_frequency(qb2.name + "_xy", df + qb_if_2)
            with for_(*from_array(dc, dcs)):
                # Flux sweeping
                set_dc_offset(qb1.name + "_z", "single", dc)
                set_dc_offset(qb2.name + "_z", "single", dc)
                wait(100)  # Wait for the flux to settle

                # Saturate qubit
                play("cw" * amp(saturation_amp), qb1.name + "_xy", duration=saturation_len * u.ns)
                play("cw" * amp(saturation_amp), qb2.name + "_xy", duration=saturation_len * u.ns)
                # Play x180 once the qubits are found
                # play("x180", qb1.name + "_xy")
                # play("x180", qb2.name + "_xy")

                # QUA macro the readout the state of the active resonators (defined in macros.py)
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits)
                # Wait for the qubit to decay to the ground state
                wait(cooldown_time * u.ns)

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
    job = qmm.simulate(config, multi_qubit_spec_vs_flux, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
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
        s1 = u.demod2volts(I1 + 1j * Q1, rr1.readout_pulse_length)
        s2 = u.demod2volts(I2 + 1j * Q2, rr2.readout_pulse_length)
        # 2D spectroscopy plot
        plt.suptitle("Qubit spectroscopy vs flux")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(dcs, (qb_if_1 + dfs) / u.MHz, np.abs(s1))
        plt.plot(qb1.z.max_frequency_point, qb_if_1 / u.MHz, "r*")
        plt.xlabel("Flux [V]")
        plt.ylabel(f"{qb1.name} IF [MHz]")
        plt.title(f"{qb1.name} (f_01: {int(qb1.xy.f_01 / u.MHz)} MHz)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(dcs, (qb_if_1 + dfs) / u.MHz, np.unwrap(np.angle(s1)))
        plt.plot(qb1.z.max_frequency_point, qb_if_1 / u.MHz, "r*")
        plt.xlabel("Flux [V]")
        plt.ylabel(f"{qb1.name} IF [MHz]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(dcs, (qb_if_2 + dfs) / u.MHz, np.abs(s2))
        plt.plot(qb2.z.max_frequency_point, qb_if_2 / u.MHz, "r*")
        plt.title(f"{qb2.name} (f_01: {int(qb2.xy.f_01 / u.MHz)} MHz)")
        plt.ylabel(f"{qb2.name} IF [MHz]")
        plt.xlabel("flux [V]")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(dcs, (qb_if_2 + dfs) / u.MHz, np.unwrap(np.angle(s2)))
        plt.plot(qb2.z.max_frequency_point, qb_if_2 / u.MHz, "r*")
        plt.xlabel("Flux [V]")
        plt.ylabel(f"{qb2.name} IF [MHz]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    # qm.close()

# Set the relevant flux points
# qb1.z.max_frequency_point =
# qb1.z.min_frequency_point =
# machine._save("quam_bootstrap_state.json")

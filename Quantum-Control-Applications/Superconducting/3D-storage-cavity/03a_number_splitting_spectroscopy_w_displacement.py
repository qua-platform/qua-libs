"""
        NUMBER SPLITTING SPECTROSCOPY WITH DISPLACEMENT
This sequence involves sending a cw pulse to the storage cavity (displacing the storage)
followed by a selective pi-pulse (x180_long) to qubit and measure across various qubit drive intermediate dfs.

The data is post-processed to determine the distance between different Fock states in the frequency domain.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the resonator drive line (whether it's an external mixer or an Octave port).
    - Identification of the qubit's resonance frequency (referred to as "qubit_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Having calibrated qubit pi pulse (x180_len) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Specification of the expected storage_thermalization_time of the storage in the configuration.

Before proceeding to the next node:
    - Update the resonance frequency of the qubit when the storage cavity is at Fock state n=1, labeled as "qubit_IF_n1".
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
import macros as macros
import numpy as np


###################
# The QUA program #
###################
n_avg = 500  # The number of averages
# Qubit detuning sweep
center = 177 * u.MHz
top = 3 * u.MHz
bottom = -3 * u.MHz
df = 10 * u.kHz
dfs = np.arange(bottom, top, df)


with program() as number_splitting_spectroscopy:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the qubit frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    state = declare(bool)
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the frequency of the digital oscillator linked to the qubit element
            update_frequency("qubit", df + center)
            # Play a displacement pulse to the storage
            play("cw", "storage")
            align("qubit", "storage")
            # Align the two elements to measure after playing the storage pulse.
            # Measure the storage state by applying a selective pi-pulse to the qubit and measure the qubit state
            play("x180_long", "qubit")
            align("qubit", "resonator")
            # Measure the state of the resonator
            state, I, Q = macros.readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)

            # Wait for the storage to decay to the ground state
            align("storage", "resonator")
            wait(storage_thermalization_time * u.ns, "storage")
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I, I_st)
            save(Q, Q_st)
            save(state, state_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        I_st.buffer(len(dfs)).average().save("I")
        Q_st.buffer(len(dfs)).average().save("Q")
        state_st.boolean_to_int().buffer(len(dfs)).average().save("state")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, number_splitting_spectroscopy, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(number_splitting_spectroscopy)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    # Live plotting
    fig1, ax1 = plt.subplots(2, 1)
    fig2, ax2 = plt.subplots(1, 1)
    interrupt_on_close(fig1, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, state, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        fig1.suptitle(f"Number Splitting Spectroscopy - LO = {storage_LO / u.GHz} GHz")
        ax1[0].clear()
        ax1[1].clear()
        ax1[0].cla()
        ax1[0].plot((dfs + center) / u.MHz, I, ".")
        ax1[0].set_xlabel("Qubit intermediate frequency [MHz]")
        ax1[0].set_ylabel(r"I [V]")
        ax1[1].cla()
        ax1[1].plot((dfs + center) / u.MHz, Q, ".")
        ax1[1].set_xlabel("Qubit intermediate frequency [MHz]")
        ax1[1].set_ylabel("Q [V]")
        plt.pause(1)
        plt.tight_layout()

        ax2.clear()
        ax2.plot((dfs + center) / u.MHz, state, ".")
        ax2.set_ylabel(r"$P_e$")
        ax2.set_xlabel("Qubit intermediate frequency [MHz]")
        ax2.set_ylim(0, 1)

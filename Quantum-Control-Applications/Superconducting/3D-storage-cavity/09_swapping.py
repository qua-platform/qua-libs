"""
        SWAPPING
This sequence involves preparing the storage cavity in Fock state n=1, then playing two detuned pump pulses to the
storage and the resonator simultaneously, followed by a selective pi-pulse (x180_long) to the qubit (with frequency that
corresponds to cavity is a Fock state n=1) and measure across various off pump pulses durations.

The data is post-processed to determine the swapping operation time.

Note that the pi-pulse should be long enough such that it will apply a pi-pulse only when the storage is at Fock state n=1.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the resonator drive line (whether it's an external mixer or an Octave port).
    - Identification of the qubit's resonance frequency (referred to as "qubit_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Having calibrated qubit pi pulse (x180_len) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having calibrated qubit's frequency that corresponds to Fock state n=1 by running number_splitting_spectroscopy.
    - Specification of the expected storage_thermalization_time of the storage in the configuration.
    - Specification of the expected beta3 pulse amplitude and duration (from "storage_displacement").
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
import scipy.optimize as spo


###################
# The QUA program #
###################
n_avg = 500  # The number of averages

# Duration time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
t_min = 16 // 4
t_max = 100000 // 4
dt = 400 // 4
# Detuning of the off pump pulses
durations = np.arange(t_min, t_max, dt)

detuning = 30 * u.MHz
with program() as swap:
    n = declare(int)  # QUA variable for the averaging loop
    t = declare(int)  # QUA variable for the qubit pulse duration
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    state = declare(bool)
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, durations)):
            update_frequency("storage", storage_IF)
            update_frequency("qubit", qubit_IF)
            # Prepare the Storage Cavity in Fock state n=1
            play("beta1", "storage")
            align()
            play("x360_long", "qubit")
            align()
            play("beta2", "storage")

            align()
            # Play two off resonance pulses. One pulse to the storage cavity and another pulse to the resonator
            update_frequency("storage", storage_IF - detuning)
            update_frequency("resonator", resonator_IF - detuning)
            play("off_pump", "storage", duration=t)
            play("off_pump", "resonator", duration=t)

            align()
            # Measure the storage state by applying a selective pi-pulse to the qubit (for storage state Fock state n=1)
            # and measure the qubit state
            update_frequency("resonator", resonator_IF)
            update_frequency("qubit", qubit_IF_n1)
            play("x180_long", "qubit")
            align()
            # Measure the state of the resonator
            state, I, Q = macros.readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)

            # Wait for the storage to decay to the ground state
            align()
            wait(storage_thermalization_time * u.ns, "storage")
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I, I_st)
            save(Q, Q_st)
            save(state, state_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        I_st.buffer(len(durations)).average().save("I")
        Q_st.buffer(len(durations)).average().save("Q")
        state_st.boolean_to_int().buffer(len(durations)).average().save("state")
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
    job = qmm.simulate(config, swap, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(swap)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "state", "Q", "iteration"], mode="live")
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
        fig1.suptitle("Swapping operation")
        ax1[0].clear()
        ax1[1].clear()
        ax1[0].cla()
        ax1[0].plot(4 * durations, R, ".")
        ax1[0].set_xlabel("Swapping duration [ms]")
        ax1[0].set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        ax1[1].cla()
        ax1[1].plot(4 * durations, phase, ".")
        ax1[1].set_xlabel("Swapping duration [ms]")
        ax1[1].set_ylabel("Phase [rad]")
        plt.pause(1)
        plt.tight_layout()

        ax2.clear()
        ax2.plot(4 * durations, state, ".")
        ax2.set_ylabel(r"$P_e$")
        ax2.set_xlabel("Swapping duration [ms]")
        ax2.set_ylim(0, 1)


def func(t, A, alpha, kappa, offset, n=0):
    return A * np.exp(-np.abs(alpha) ** 2 * np.exp(-kappa * t)) + offset


x0 = [-max(state) + min(state), 3, 6, max(state)]
popt, pcov = spo.curve_fit(func, durations * 4 / u.ms, state, p0=x0)
print(popt)

fig3, ax3 = plt.subplots(1, 1)

x = 4 * np.linspace(4e-3, np.max(durations)) / u.ms
ax3.plot(4 * durations / u.ms, state, ".")
ax3.plot(x, func(x, *popt))
ax3.plot(x, func(x, *x0))
ax3.set_ylabel(r"$P_e$")
ax3.set_xlabel("Pulse duration [ns]")
plt.show()

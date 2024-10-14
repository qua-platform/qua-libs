"""
        CAVITY T2
This sequence involves initiating the storage in the Fock state n=1 using SNAP, wait varying time,
applying a beta3 pulse that displace the storage back to the ground state, and measure by applying a selective pi-pulse (x180_long) to qubit (with frequency that corresponds to Fock state n=1) and measure the resonator.

The data is post-processed to determine the storage T2* parameter and the storage's frequency.


Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the resonator drive line (whether it's an external mixer or an Octave port).
    - Identification of the qubit's resonance frequency (referred to as "qubit_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Having calibrated qubit pi pulse (x180_len) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Specification of the expected storage_thermalization_time of the storage in the configuration.
    - Specification of the expected beta3 pulse amplitude and duration (from "storage_displacement").

Before proceeding to the next node:
    - Update the storage frequency, labeled as storage_IF.
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
detuning = 20 * u.kHz

# Duration time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
t_min = 10000 // 4
t_max = 100000 // 4
dt = 1000 // 4
durations = np.arange(t_min, t_max, dt)


with program() as qubit_spec:
    n = declare(int)  # QUA variable for the averaging loop
    t = declare(int)  # QUA variable for the qubit frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    phase = declare(fixed)
    state = declare(bool)
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, durations)):
            assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
            # Prepare the storage cavity in Fock state n=1
            update_frequency("qubit", qubit_IF)
            play("beta1", "storage")
            align()
            align("qubit", "storage")
            play("x360_long", "qubit")  # Play a selective pi-pulse for n=0
            align()
            play("beta2", "storage")

            # Wait t time before the next cavity displacement
            wait(t, "storage")
            # Rotate the frame of the beta3 pulse to implement a virtual Z-rotation
            frame_rotation_2pi(phase, "storage")
            play("beta3", "storage")

            # Align the two elements to measure after playing the qubit pulse.
            # Measure the storage state by applying a selective pi-pulse to the qubit (for storage state Fock state n=1)
            # and measure the qubit state
            align()
            update_frequency("qubit", qubit_IF_n1)
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
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(qubit_spec)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    # Live plotting
    fig1, ax1 = plt.subplots(2, 1)
    fig2, ax2 = plt.subplots(1, 1)
    interrupt_on_close(fig1, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, state, Q, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        fig1.suptitle(f"Storage Cavity T2 for Fock state n=1 - LO = {storage_LO / u.GHz} GHz")
        ax1[0].clear()
        ax1[1].clear()
        ax1[0].cla()
        ax1[0].plot(4 * durations, I, ".")
        ax1[0].set_xlabel("Idle time [ns]")
        ax1[0].set_ylabel(r"I [V]")
        ax1[1].cla()
        ax1[1].plot(4 * durations, Q, ".")
        ax1[1].set_xlabel("Idle time [ns]")
        ax1[1].set_ylabel("Q [V]")
        plt.pause(1)
        plt.tight_layout()

        ax2.clear()
        ax2.plot(4 * durations, state, ".")
        ax2.set_ylabel(r"$P_e$")
        ax2.set_xlabel("Idle time [ns]")
        ax2.set_ylim(0, 1)

    # Fit the results to extract the qubit frequency and T2*
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        ramsey_fit = fit.ramsey(4 * durations, state, plot=True)
        qubit_T2 = np.abs(ramsey_fit["T2"][0])
        qubit_detuning = ramsey_fit["f"][0] * u.GHz - detuning
        plt.xlabel("Idle time [ns]")
        plt.ylabel("I quadrature [V]")
        print(f"Qubit detuning to update in the config: qubit_IF += {-qubit_detuning:.0f} Hz")
        print(f"T2* = {qubit_T2:.0f} ns")
        plt.legend((f"detuning = {-qubit_detuning / u.kHz:.3f} kHz", f"T2* = {qubit_T2:.0f} ns"))
        plt.title("Ramsey measurement with detuned gates")
        print(f"Detuning to add: {-qubit_detuning / u.kHz:.3f} kHz")
    except (Exception,):
        pass

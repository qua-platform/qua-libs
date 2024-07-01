"""
        T1 MEASUREMENT
The sequence consists in putting the qubit in the excited stated by playing the x180 pulse and measuring the resonator
after a varying time. The qubit T1 is extracted by fitting the exponential decay of the measured quadratures.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubit T1 (qubit_T1) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
import matplotlib.pyplot as plt


###################
# The QUA program #
###################
n_avg = 1000
# The wait time sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
tau_min = 16 // 4
tau_max = 40_000 // 4
d_tau = 100 // 4
taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # Linear sweep
# taus = np.logspace(np.log10(tau_min), np.log10(tau_max), 29)  # Log sweep

with program() as T1:
    n = declare(int)  # QUA variable for the averaging loop
    t = declare(int)  # QUA variable for the wait time
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, taus)):
            # Play the x180 gate to put the qubit in the excited state
            play("x180", "qubit")
            # Wait a varying time after putting the qubit in the excited state
            wait(t, "qubit")
            # Align the two elements to measure after having waited a time "tau" after the qubit pulse.
            align("qubit", "resonator")
            # Measure the state of the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "rotated_sin", I),
                dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
            )
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, "resonator")
            # Save the 'I_e' & 'Q_e' quadratures to their respective streams
            save(I, I_st)
            save(Q, Q_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        # If log sweep, then the swept values will be slightly different from np.logspace because of integer rounding in QUA.
        # get_equivalent_log_array() is used to get the exact values used in the QUA program.
        if np.isclose(np.std(taus[1:] / taus[:-1]), 0, atol=1e-3):
            taus = get_equivalent_log_array(taus)
            I_st.buffer(len(taus)).average().save("I")
            Q_st.buffer(len(taus)).average().save("Q")
        else:
            I_st.buffer(len(taus)).average().save("I")
            Q_st.buffer(len(taus)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, T1, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(T1)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert the results into Volts
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle("T1 measurement")
        plt.subplot(211)
        plt.cla()
        plt.plot(4 * taus, I, ".")
        plt.ylabel("I quadrature [V]")
        plt.subplot(212)
        plt.cla()
        plt.plot(4 * taus, Q, ".")
        plt.xlabel("Qubit decay time [ns]")
        plt.ylabel("Q quadrature [V]")
        plt.pause(0.1)
        plt.tight_layout()

    # Fit the results to extract the qubit decay time T1
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        decay_fit = fit.T1(4 * taus, I, plot=True)
        qubit_T1 = np.round(np.abs(decay_fit["T1"][0]) / 4) * 4
        plt.xlabel("Delay [ns]")
        plt.ylabel("I quadrature [V]")
        print(f"Qubit decay time to update in the config: qubit_T1 = {qubit_T1:.0f} ns")
        plt.legend((f"Relaxation time T1 = {qubit_T1:.0f} ns",))
        plt.title("T1 measurement")
    except (Exception,):
        pass

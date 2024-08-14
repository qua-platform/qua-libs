"""
        RAMSEY CHEVRON (IDLE TIME VS FREQUENCY)
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different qubit intermediate
frequencies and idle times.
From the results, one can estimate the qubit frequency more precisely than by doing Rabi and also gets a rough estimate
of the qubit coherence time.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubit frequency (qubit_IF) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt


###################
# The QUA program #
###################
n_avg = 100  # Number of averaging loops

# Frequency sweep in Hz
freq_span = 10 * u.MHz
df = 100 * u.kHz
dfs = np.arange(-freq_span, freq_span, df)

# Idle time sweep (Needs to be a list of integers) - in clock cycles (4ns)
tau_max = 2000 // 4
d_tau = 40 // 4
taus = np.arange(0, tau_max, d_tau)
if len(np.where((taus > 0) & (taus < 4))[0]) > 0:
    raise Exception("Delay must be either 0 or an integer larger than 4.")

with program() as ramsey_freq_duration:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the qubit detuning
    delay = declare(int)  # QUA variable for the idle time
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(delay, taus)):  # QUA for_ loop for sweeping the idle time
            with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the qubit frequency
                # Update the frequency of the digital oscillator linked to the qubit element
                update_frequency("qubit", df + qubit_IF)
                # Adjust the idle time
                with if_(delay >= 4):
                    play("x90", "qubit")
                    wait(delay, "qubit")
                    play("x90", "qubit")
                with else_():
                    play("x90", "qubit")
                    play("x90", "qubit")
                align("qubit", "resonator")
                # Measure the state of the resonator.
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("rotated_cos", "rotated_sin", I),
                    dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
                )
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns, "resonator")
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(dfs)).buffer(len(taus)).average().save("I")
        Q_st.buffer(len(dfs)).buffer(len(taus)).average().save("Q")
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
    job = qmm.simulate(config, ramsey_freq_duration, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ramsey_freq_duration)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert the results into Volts
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle("Ramsey chevron")
        plt.subplot(211)
        plt.cla()
        plt.title("I quadrature [V]")
        plt.pcolor(dfs / u.MHz, taus * 4, I)
        plt.xlabel("Qubit detuning [MHz]")
        plt.ylabel("Idle time [ns]")
        plt.subplot(212)
        plt.cla()
        plt.title("Q quadrature [V]")
        plt.pcolor(dfs / u.MHz, taus * 4, Q)
        plt.xlabel("Qubit detuning [MHz]")
        plt.ylabel("Idle time [ns]")
        plt.tight_layout()
        plt.pause(0.01)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

"""
        RABI CHEVRON (AMPLITUDE VS FREQUENCY)
This sequence involves executing the qubit pulse (such as x180, square_pi, or other types) and measuring the state
of the resonator across various qubit intermediate frequencies and pulse amplitudes.
By analyzing the results, one can determine the qubit and estimate the x180 pulse amplitude for a specified duration.

Prerequisites:
    - Determination of the resonator's resonance frequency when coupled to the qubit of interest (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (be it an external mixer or an Octave port).
    - Identification of the approximate qubit frequency (referred to as "qubit_spectroscopy").
    - Configuration of the qubit frequency and the desired pi pulse duration (labeled as "x180_len").
    - Set the desired flux bias

Before proceeding to the next node:
    - Adjust the qubit frequency setting, labeled as "qubit_IF", in the configuration.
    - Modify the qubit pulse amplitude setting, labeled as "x180_amp", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100  # The number of averages
# The frequency sweep parameters
span = 5 * u.MHz
df = 100 * u.kHz
dfs = np.arange(-span, +span + 0.1, df)  # The frequency vector
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
a_min = 0
a_max = 1.0
n_a = 51
amplitudes = np.linspace(a_min, a_max, n_a)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "dfs": dfs,
    "amplitudes": amplitudes,
    "config": config,
}

###################
# The QUA program #
###################
with program() as rabi_amp_freq:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the qubit frequency
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(a, amplitudes)):  # QUA for_ loop for sweeping the pulse amplitude pre-factor
            with for_(*from_array(f, dfs)):  # QUA for_ loop for sweeping the frequency
                # Update the frequency of the digital oscillator linked to the qubit element
                update_frequency("qubit", f + qubit_IF)
                # Adjust the qubit pulse amplitude
                play("x180" * amp(a), "qubit")
                # Align the two elements to measure after playing the qubit pulse.
                align("qubit", "resonator")
                # Measure the state of the resonator
                # The integration weights have changed to maximize the SNR after having calibrated the IQ blobs.
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
        I_st.buffer(len(dfs)).buffer(n_a).average().save("I")
        Q_st.buffer(len(dfs)).buffer(n_a).average().save("Q")
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
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, rabi_amp_freq, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rabi_amp_freq)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.subplot(211)
        plt.suptitle(f"Rabi chevron with LO={qubit_LO / u.GHz}GHz and IF={qubit_IF / u.MHz}MHz")
        plt.cla()
        plt.title(r"$R=\sqrt{I^2 + Q^2}$")
        plt.pcolor(dfs / u.MHz, amplitudes * x180_amp, R)
        plt.xlabel("Frequency detuning [MHz]")
        plt.ylabel("Pulse amplitude [V]")
        plt.subplot(212)
        plt.cla()
        plt.title("Phase")
        plt.pcolor(dfs / u.MHz, amplitudes * x180_amp, np.unwrap(phase))
        plt.xlabel("Frequency detuning [MHz]")
        plt.ylabel("Pulse amplitude [V]")
        plt.tight_layout()
        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

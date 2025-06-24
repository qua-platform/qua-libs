"""
        TIME OF FLIGHT
This sequence involves sending a readout pulse and capturing the raw ADC traces.
The data undergoes post-processing to calibrate three distinct parameters:
    - Time of Flight: This represents the internal processing time and the propagation delay of the readout pulse.
    Its value can be adjusted in the configuration under "time_of_flight".
    This value is utilized to offset the acquisition window relative to when the readout pulse is dispatched.

    - Analog Inputs Offset: Due to minor impedance mismatches, the signals captured by the OPX might exhibit slight offsets.
    These can be rectified in the configuration at: config/controllers/"con1"/analog_inputs, enhancing the demodulation process.

    - Analog Inputs Gain: If a signal is constrained by digitization or if it saturates the ADC,
    the variable gain of the OPX analog input can be modified to fit the signal within the ADC range of +/-0.5V.
    This gain, ranging from -12 dB to 20 dB, can also be adjusted in the configuration at: config/controllers/"con1"/analog_inputs.
"""

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from qm import SimulationConfig

from configuration_mw_fem import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 5000  # The number of averages

###################
# The QUA program #
###################
with program() as PROGRAM:
    n = declare(int)  # QUA variable for the averaging loop
    adc_st = declare_stream(adc_trace=True)  # The stream to store the raw ADC trace

    # # OPTIONAL to check time-of-flight at arbitrary frequency
    # update_frequency('q1_rr', 100e6)

    with for_(n, 0, n < n_avg, n + 1):
        # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
        reset_if_phase("rr1")
        reset_if_phase("rr2")
        # Sends the readout pulse and stores the raw ADC traces in the stream called "adc_st"
        measure("readout", "rr1", adc_st)
        measure("readout", "rr2", None)
        # Wait for the resonators to empty
        wait(depletion_time * u.ns, "rr1")
        wait(depletion_time * u.ns, "rr2")

    with stream_processing():
        # Will save average:
        adc_st.input1().average().save("adc1")
        # # Will save only last run:
        adc_st.input1().save("adc1_single_run")


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
    simulation_config = SimulationConfig(duration=1_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, PROGRAM, simulation_config)
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
    try:
        # Open a quantum machine to execute the QUA program
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(PROGRAM)
        # Creates a result handle to fetch data from the OPX
        res_handles = job.result_handles
        # Waits (blocks the Python console) until all results have been acquired
        res_handles.wait_for_all_values()
        # Fetch the raw ADC traces and convert them into Volts
        adc1 = u.raw2volts(res_handles.get("adc1").fetch_all())
        adc1_single_run = u.raw2volts(res_handles.get("adc1_single_run").fetch_all())

        # Derive the average values
        adc1_mean = np.mean(adc1)
        # Remove the average values
        adc1_unbiased = adc1 - np.mean(adc1)
        # Filter the data to get the pulse arrival time
        signal = savgol_filter(np.abs(adc1_unbiased), 11, 3)
        # Detect the arrival of the readout signal
        th = (np.mean(signal[:100]) + np.mean(signal[:-100])) / 2
        delay = np.where(signal > th)[0][0]

        # Plot data for each rl
        fig = plt.figure(figsize=(12, 6))

        # Plot for single run
        plt.subplot(121)
        plt.title("Single run")
        plt.plot(adc1_single_run.real, label="Input 1 real")
        plt.plot(adc1_single_run.imag, label="Input 1 image")
        plt.axhline(y=0)
        plt.xlabel("Time [ns]")
        plt.ylabel("Signal amplitude [V]")
        plt.legend()

        # Plot for averaged run
        plt.subplot(122)
        plt.title("Averaged run")
        plt.plot(adc1.real, label="Input 1 real")
        plt.plot(adc1.imag, label="Input 1 imag")
        plt.axhline(y=0)
        plt.xlabel("Time [ns]")
        plt.legend()
        plt.tight_layout()

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show()

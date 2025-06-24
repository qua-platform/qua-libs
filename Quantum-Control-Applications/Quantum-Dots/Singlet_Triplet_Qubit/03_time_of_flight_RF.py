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
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100  # Number of averaging loops

###################
# The QUA program #
###################
with program() as tof_prog:
    n = declare(int)  # QUA variable for the averaging loop
    adc_st = declare_stream(adc_trace=True)  # The stream to store the raw ADC trace

    with for_(n, 0, n < n_avg, n + 1):
        # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
        reset_phase("tank_circuit")
        # Sends the readout pulse and stores the raw ADC traces in the stream called "adc_st"
        # measure("readout", "TIA", adc_st)
        measure("readout", "tank_circuit", adc_st)
        # Wait for the resonator to deplete
        wait(1_000 * u.ns, "tank_circuit")

    with stream_processing():
        # Please adjust the analog inputs according to the connectivity (input1/2 -> rf)
        # Will save average:
        adc_st.input1().average().save("adc1")
        # Will save only last run:
        adc_st.input1().save("adc1_single_run")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False
if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, tof_prog, simulation_config)
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
    job = qm.execute(tof_prog)
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
    signal = savgol_filter(np.abs(1j * adc1_unbiased), 11, 3)
    # Detect the arrival of the readout signal
    th = (np.mean(signal[:100]) + np.mean(signal[:-100])) / 2
    delay = np.where(signal > th)[0][0]
    delay = np.round(delay / 4) * 4  # Find the closest multiple integer of 4ns

    # Plot data
    fig = plt.figure()
    plt.subplot(121)
    plt.title("Single run")
    plt.plot(adc1_single_run, "r", label="Input 2")
    xl = plt.xlim()
    yl = plt.ylim()
    plt.axhline(y=0.5)
    plt.axhline(y=-0.5)
    plt.plot(xl, adc1_mean * np.ones(2), "k--")
    plt.plot(delay * np.ones(2), yl, "k--")
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal amplitude [V]")
    plt.legend()
    plt.subplot(122)
    plt.title("Averaged run")
    plt.plot(adc1, "r", label="Input 2")
    xl = plt.xlim()
    yl = plt.ylim()
    plt.plot(xl, adc1_mean * np.ones(2), "k--")
    plt.plot(delay * np.ones(2), yl, "k--")
    plt.xlabel("Time [ns]")
    plt.legend()
    plt.grid("all")
    plt.tight_layout()
    plt.show()

    # Update the config
    print(f"DC offset to add to Q in the config: {-adc1_mean:.6f} V")
    print(f"Time Of Flight to add in the config: {delay} ns")

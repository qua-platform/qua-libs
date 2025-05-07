"""
        RAW ADC TRACES
The goal of this script is to measure the raw ADC traces without demodulation or integration.
It can be used to check the signals before demodulation, make sure that the ADCs are not saturated and estimate the SNR.
It also allows to calibrate several parameters:
    - The time of flight: it corresponds to some internal processing time and propagation delay of the readout pulse.
      Its value can be updated in the configuration (time_of_flight) and is used to delay the acquisition window with
      respect to the time at which the readout pulse is sent.
    - The analog inputs offset: due to some small impedance mismatch, the signals acquired by the OPX can have small
      offsets that can be removed in the configuration (config/controllers/"con1"/analog_inputs) to improve demodulation.
    - The analog inputs gain: if the signal is limited by digitization or saturates the ADC, the variable gain of the
      OPX analog input can be set to adjust the signal within the ADC range +/-0.5V.
      The gain (-12 dB to 20 dB) can also be set in the configuration (config/controllers/"con1"/analog_inputs).
    - The threshold for time-tagging: it corresponds to the ADC value above or below which the signal is considered to
      be an event that can be detected and time-tagged in the subsequent scripts.
"""

from qm import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from configuration import *
from qm import SimulationConfig

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 1_000

###################
# The QUA program #
###################
with program() as TimeTagging_calibration:
    n = declare(int)  # QUA variable for the averaging loop
    adc_st = declare_stream(adc_trace=True)  # The stream to store the raw ADC trace
    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        # Drive the AOM to play the readout laser pulse
        play("laser_ON", "AOM1")
        # Record the raw ADC traces in the stream called "adc_st"
        measure("long_readout", "SPCM1", adc_st)
        # Waits for the
        wait(1000, "SPCM1")

    with stream_processing():
        # Will save average:
        adc_st.input1().average().save("adc1")
        # Will save only last run:
        adc_st.input1().save("adc1_single_run")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, TimeTagging_calibration, simulation_config)
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
    # Open Quantum Machine
    qm = qmm.open_qm(config)
    # Execute program
    job = qm.execute(TimeTagging_calibration)
    # create a handle to get results
    res_handles = job.result_handles
    # Wait until the program is done
    res_handles.wait_for_all_values()
    # Fetch results and convert traces to volts
    adc1 = u.raw2volts(res_handles.get("adc1").fetch_all())
    adc1_single_run = u.raw2volts(res_handles.get("adc1_single_run").fetch_all())
    # Plot data
    plt.figure()
    plt.subplot(121)
    plt.title("Single run")
    plt.plot(adc1_single_run, label="Input 1")
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal amplitude [V]")

    plt.subplot(122)
    plt.title("Averaged run")
    plt.plot(adc1, label="Input 1")
    plt.xlabel("Time [ns]")
    plt.tight_layout()

    print(f"\nInput1 mean: {np.mean(adc1)} V")

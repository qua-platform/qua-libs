"""
02_raw_adc_traces.py: a script used to look at the raw ADC data from inputs 1 and 2,
this allows checking that the ADC is not saturated, correct for DC offsets and define the time of flight and
threshold for time-tagging.
"""

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
import matplotlib.pyplot as plt
from configuration import *

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 1000

###################
# The QUA program #
###################
with program() as TimeTagging_calibration:
    n = declare(int)
    adc_st = declare_stream(adc_trace=True)
    with for_(n, 0, n < n_avg, n + 1):
        play("laser_ON", "AOM")
        measure("long_readout", "SNSPD", adc_st)
        wait(1000, "SNSPD")

    with stream_processing():
        # Will save average:
        adc_st.input1().average().save("adc1")
        # Will save only last run:
        adc_st.input1().save("adc1_single_run")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################
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
    # Wait untill the program is done
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
    plt.legend()

    plt.subplot(122)
    plt.title("Averaged run")
    plt.plot(adc1, label="Input 1")
    plt.xlabel("Time [ns]")
    plt.legend()
    plt.tight_layout()

    print(f"\nInput1 mean: {np.mean(adc1)} V")

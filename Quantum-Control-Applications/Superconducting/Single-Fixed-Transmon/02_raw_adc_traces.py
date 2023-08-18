"""
        RAW ADC TRACES
The goal of this script is to measure the raw ADC traces without demodulation or integration.
It can be used to check the signals before demodulation, make sure that the ADCs are not saturated and estimate the SNR.
"""
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np

##############################
# Program-specific variables #
##############################
n_avg = 100  # The number of averages

###################
# The QUA program #
###################
with program() as raw_trace_prog:
    n = declare(int)  # QUA variable for the averaging loop
    adc_st = declare_stream(adc_trace=True)  # The stream to store the raw ADC trace

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        # Make sure that the readout pulse is sent with the same phase so that the acquired signal does not average out
        reset_phase("resonator")
        # Measure the resonator (send a readout pulse and record the raw ADC trace)
        measure("readout", "resonator", adc_st)
        # Wait for the resonator to deplete
        wait(depletion_time * u.ns, "resonator")

    with stream_processing():
        # Will save average:
        adc_st.input1().average().save("adc1")
        adc_st.input2().average().save("adc2")
        # Will save only last run:
        adc_st.input1().save("adc1_single_run")
        adc_st.input2().save("adc2_single_run")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(raw_trace_prog)
    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    adc1 = u.raw2volts(res_handles.get("adc1").fetch_all())
    adc2 = u.raw2volts(res_handles.get("adc2").fetch_all())
    adc1_single_run = u.raw2volts(res_handles.get("adc1_single_run").fetch_all())
    adc2_single_run = u.raw2volts(res_handles.get("adc2_single_run").fetch_all())

    plt.figure()
    plt.subplot(121)
    plt.title("Single run")
    plt.plot(adc1_single_run, label="Input 1")
    plt.plot(adc2_single_run, label="Input 2")
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal amplitude [V]")
    plt.legend()

    plt.subplot(122)
    plt.title("Averaged run")
    plt.plot(adc1, label="Input 1")
    plt.plot(adc2, label="Input 2")
    plt.xlabel("Time [ns]")
    plt.legend()
    plt.tight_layout()

    print(f"\nInput1 mean: {np.mean(adc1)} V\n" f"Input2 mean: {np.mean(adc2)} V")

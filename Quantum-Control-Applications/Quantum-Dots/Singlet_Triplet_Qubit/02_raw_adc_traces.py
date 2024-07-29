"""
        RAW ADC TRACES
This script aims to measure data captured within a specific window defined by the measure() function.
We term the digitized, unprocessed data as "raw ADC traces" because they represent the acquired waveforms without any
real-time processing by the pulse processor, such as demodulation, integration, or time-tagging.

The script is useful for inspecting signals prior to demodulation, ensuring the ADCs are not saturated,
correcting any non-zero DC offsets, and estimating the SNR.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt


###################
# The QUA program #
###################
n_avg = 100  # The number of averages

with program() as raw_trace_prog:
    n = declare(int)  # QUA variable for the averaging loop
    adc_dc_st = declare_stream(adc_trace=True)  # The stream to store the raw ADC trace for the DC line
    adc_rf_st = declare_stream(adc_trace=True)  # The stream to store the raw ADC trace for the RF line

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        # Make sure that the readout pulse is sent with the same phase so that the acquired signal does not average out
        reset_phase("tank_circuit")
        # *amp(0) sets the amplitude to zero, thus the collected trace is the digitization of the analog input
        # in the majority of cases the DC bias to the sensor is applied with an external DC source.
        measure("readout" * amp(0), "TIA", adc_dc_st)
        # Measure the tank circuit (send a readout pulse and record the raw ADC trace)
        # Wait for the resonator to deplete
        wait(1_000 * u.ns, "TIA", "tank_circuit")

    with stream_processing():
        # Please adjust the analog inputs according to the connectivity (input1/2 -> dc or rf)
        # Will save average:
        adc_dc_st.input1().average().save("adc_dc")
        adc_rf_st.input2().average().save("adc_rf")
        # Will save only last run:
        adc_dc_st.input1().save("adc_dc_single_run")
        adc_rf_st.input2().save("adc_rf_single_run")

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
    job = qmm.simulate(config, raw_trace_prog, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(raw_trace_prog)
    # Creates a result handle to fetch data from the OPX
    res_handles = job.result_handles
    # Waits (blocks the Python console) until all results have been acquired
    res_handles.wait_for_all_values()
    # Fetch the raw ADC traces and convert them into Volts
    adc_dc = u.raw2volts(res_handles.get("adc_dc").fetch_all())
    adc_rf = u.raw2volts(res_handles.get("adc_rf").fetch_all())
    adc_dc_single_run = u.raw2volts(res_handles.get("adc_dc_single_run").fetch_all())
    adc_rf_single_run = u.raw2volts(res_handles.get("adc_rf_single_run").fetch_all())
    # Plot data
    plt.figure()
    plt.subplot(121)
    plt.title("Single run")
    plt.plot(adc_dc_single_run, label="DC input")
    plt.plot(adc_rf_single_run, label="RF input")
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal amplitude [V]")
    plt.legend()

    plt.subplot(122)
    plt.title("Averaged run")
    plt.plot(adc_dc, label="DC input")
    plt.plot(adc_rf, label="RF input")
    plt.xlabel("Time [ns]")
    plt.legend()
    plt.tight_layout()

    print(f"\nDC input mean: {np.mean(adc_dc)} V")
    print(f"RF input mean: {np.mean(adc_rf)} V\n")

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
from qm import SimulationConfig
from qualang_tools.units import unit
from quam_components import QuAM
from macros import node_save
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load(os.path.join('..', 'configuration', 'quam_state'))
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
resonator = machine.active_qubits[0].resonator  # The resonator element

###################
# The QUA program #
###################
n_avg = 100  # Number of averaging loops

with program() as raw_trace_prog:
    n = declare(int)  # QUA variable for the averaging loop
    adc_st = declare_stream(adc_trace=True)  # The stream to store the raw ADC trace

    with for_(n, 0, n < n_avg, n + 1):
        # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
        reset_phase(resonator.name)
        # Measure the resonator (send a readout pulse and record the raw ADC trace)
        resonator.measure("readout", stream=adc_st)
        # Wait for the resonator to deplete
        wait(machine.depletion_time * u.ns, resonator.name)

    with stream_processing():
        # Will save average:
        adc_st.input1().average().save("adc1")
        adc_st.input2().average().save("adc2")
        # # Will save only last run:
        adc_st.input1().save("adc1_single_run")
        adc_st.input2().save("adc2_single_run")


#######################
# Simulate or execute #
#######################
simulate = False
if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, raw_trace_prog, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(raw_trace_prog)
    # Creates a result handle to fetch data from the OPX
    res_handles = job.result_handles
    # Waits (blocks the Python console) until all results have been acquired
    res_handles.wait_for_all_values()
    # Fetch the raw ADC traces and convert them into Volts
    adc1 = u.raw2volts(res_handles.get("adc1").fetch_all())
    adc2 = u.raw2volts(res_handles.get("adc2").fetch_all())
    adc1_single_run = u.raw2volts(res_handles.get("adc1_single_run").fetch_all())
    adc2_single_run = u.raw2volts(res_handles.get("adc2_single_run").fetch_all())
    # Derive the average values
    adc1_mean = np.mean(adc1)
    adc2_mean = np.mean(adc2)
    # Remove the average values
    adc1_unbiased = adc1 - np.mean(adc1)
    adc2_unbiased = adc2 - np.mean(adc2)
    # Filter the data to get the pulse arrival time
    signal = savgol_filter(np.abs(adc1_unbiased + 1j * adc2_unbiased), 11, 3)
    # Detect the arrival of the readout signal
    th = (np.mean(signal[:100]) + np.mean(signal[:-100])) / 2
    delay = np.where(signal > th)[0][0]
    delay = int(np.round(delay / 4) * 4)  # Find the closest multiple integer of 4ns

    # Plot data
    fig = plt.figure()
    plt.subplot(121)
    plt.title("Single run")
    plt.plot(adc1_single_run, "b", label="Input 1")
    plt.plot(adc2_single_run, "r", label="Input 2")
    xl = plt.xlim()
    yl = plt.ylim()
    plt.axhline(y=0.5)
    plt.axhline(y=-0.5)
    plt.plot(xl, adc1_mean * np.ones(2), "k--")
    plt.plot(xl, adc2_mean * np.ones(2), "k--")
    plt.plot(delay * np.ones(2), yl, "k--")
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal amplitude [V]")
    plt.legend()
    plt.subplot(122)
    plt.title("Averaged run")
    plt.plot(adc1, "b", label="Input 1")
    plt.plot(adc2, "r", label="Input 2")
    xl = plt.xlim()
    yl = plt.ylim()
    plt.plot(xl, adc1_mean * np.ones(2), "k--")
    plt.plot(xl, adc2_mean * np.ones(2), "k--")
    plt.plot(delay * np.ones(2), yl, "k--")
    plt.xlabel("Time [ns]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"DC offset to add to I in the config: {-adc1_mean:.6f} V")
    print(f"DC offset to add to Q in the config: {-adc2_mean:.6f} V")
    print(f"Time Of Flight to add in the config: {delay} ns")

    # Update QUAM
    for q in machine.active_qubits:
        q.resonator.opx_input_offset_I -= np.mean(adc1)
        q.resonator.opx_input_offset_Q -= np.mean(adc2)
        q.resonator.time_of_flight += delay

    # Save data from the node
    data = {
        "offset_1": np.mean(adc1),
        "offset_2": np.mean(adc2),
        "delay": delay,
        "raw_adc_1": adc1,
        "raw_adc_2": adc2,
        "raw_adc_1_single_shot": adc1_single_run,
        "raw_adc_2_single_shot": adc2_single_run,
        "figure": fig,
    }
    node_save("time_of_flight", data, machine)

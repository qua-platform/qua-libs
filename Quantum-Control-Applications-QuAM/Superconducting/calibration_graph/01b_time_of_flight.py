# %%
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
from typing import Optional
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.trackable_object import tracked_updates

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubit: str = "q1"
    num_averages: int = 100
    time_of_flight_in_ns: Optional[int] = None
    intermediate_frequency_in_mhz: Optional[float] = None
    readout_amplitude_in_v: Optional[float] = None
    readout_length_in_ns: Optional[int] = None
    simulate: bool = False
    timeout: int = 100

node = QualibrationNode(
    name="01_Time_of_Flight",
    parameters=Parameters()
)



from qm.qua import *
from qm import SimulationConfig
from qualang_tools.units import unit
from quam_libs.components import QuAM
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter



# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Get the relevant QuAM components
resonators = [q.resonator for q in machine.active_qubits]

tracked_resonators = []
for resonator in resonators:
    # make temporary updates before running the program and revert at the end.
    with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
        resonator.time_of_flight = node.parameters.time_of_flight_in_ns
        resonator.operations["readout"].length = node.parameters.readout_length_in_ns
        resonator.operations["readout"].amplitude = node.parameters.readout_amplitude_in_v
        if node.parameters.intermediate_frequency_in_mhz is not None:
            resonator.intermediate_frequency = node.parameters.intermediate_frequency_in_mhz * u.MHz
        tracked_resonators.append(resonator)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
resonator = machine.qubits[node.parameters.qubit].resonator  # The resonator element



with program() as raw_trace_prog:
    n = declare(int)  # QUA variable for the averaging loop
    adc_st = declare_stream(adc_trace=True)  # The stream to store the raw ADC trace

    with for_(n, 0, n < node.parameters.num_averages, n + 1):
        # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
        reset_phase(resonator.name)
        # Measure the resonator (send a readout pulse and record the raw ADC trace)
        resonator.measure("readout", stream=adc_st)
        # Wait for the resonator to deplete
        wait(machine.depletion_time * u.ns, resonator.name)

    with stream_processing():
        # Will save average:
        adc_st.input1().average().save("adc")
        # Will save only last run:
        adc_st.input1().save("adc_single_run")



if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, raw_trace_prog, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    # save the figure
    node.results = {"figure": plt.gcf()}
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
    adc = u.raw2volts(res_handles.get("adc").fetch_all())
    adc_single_run = u.raw2volts(res_handles.get("adc_single_run").fetch_all())
    # Filter the data to get the pulse arrival time
    signal = savgol_filter(np.abs(adc), 11, 3)
    # Detect the arrival of the readout signal
    th = (np.mean(signal[:100]) + np.mean(signal[:-100])) / 2
    delay = np.where(signal > th)[0][0]
    delay = np.round(delay / 4) * 4  # Find the closest multiple integer of 4ns

    # Plot data
    fig = plt.figure()
    plt.subplot(121)
    plt.title("Single run")
    plt.plot(adc_single_run.real, "b", label="I")
    plt.plot(adc_single_run.imag, "r", label="Q")
    xl = plt.xlim()
    yl = plt.ylim()
    plt.axvline(delay, color="k", linestyle="--", label="TOF")
    plt.fill_between(range(len(adc_single_run)), -0.5, 0.5, color="grey", alpha=0.2, label="ADC Range")
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal amplitude [V]")
    plt.legend()
    plt.subplot(122)
    plt.title("Averaged run")
    plt.plot(adc.real, "b", label="I")
    plt.plot(adc.imag, "r", label="Q")
    plt.axvline(delay, color="k", linestyle="--", label="TOF")
    plt.xlabel("Time [ns]")
    plt.legend()
    plt.grid("all")
    plt.tight_layout()
    plt.show()

    # Update the config
    print(f"Time Of Flight to add in the config: {delay} ns")

    for resonator in tracked_resonators:
        resonator.revert_changes()

    with node.record_state_updates():
        for resonator in tracked_resonators:
            resonator.reapply_changes()
        for resonator in resonators:
            if node.parameters.time_of_flight_in_ns is not None:
                resonator.time_of_flight = node.parameters.time_of_flight_in_ns + int(delay)
            else:
                resonator.time_of_flight = resonator.time_of_flight + int(delay)


    node.results = {
        "run_parameters": node.parameters.model_dump(),
        "delay": delay,
        "raw_adc": adc,
        "raw_adc_single_shot": adc_single_run,
        "figure": fig,
    }

node.machine = machine
node.save()

# %%

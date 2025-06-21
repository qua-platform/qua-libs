# %% {Imports}
import matplotlib.pyplot as plt

from configuration.configuration_with_lf_fem_and_mw_fem import *

from qm import QuantumMachinesManager
from qm.qua import *

from qualang_tools.results import progress_counter, fetching_tool

from qualibrate import QualibrationNode
from calibration_utils.time_of_flight import (
    Parameters,
    process_raw_data,
    fit_raw_data,
    plot_single_run_with_fit,
    plot_averaged_run_with_fit,
)
from qualibration_libs.runtime import simulate_and_plot

# %% {Initialisation}
description = """
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


node = QualibrationNode[Parameters, None](name="time_of_flight", description=description, parameters=Parameters())


# %% {Node_parameters}
# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, None]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.simulate = False
    node.parameters.resonators = ["q1_resonator", "q2_resonator"]
    node.parameters.multiplexed = True
    node.parameters.num_shots = 10
    node.parameters.depletion_time = 10 * u.us


# %% {QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, None]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Get the active qubits from the node and organize them by batches
    resonators = node.parameters.resonators
    with program() as node.namespace["qua_program"]:
        n = declare(int)  # QUA variable for the averaging loop
        n_st = declare_stream()
        adc_st = [
            declare_stream(adc_trace=True) for _ in range(len(resonators))
        ]  # The stream to store the raw ADC trace

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)
            for i, resonator in enumerate(resonators):
                # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
                reset_if_phase(resonator)
                # Measure the resonator (send a readout pulse and record the raw ADC trace)
                measure(
                    "readout",
                    resonator,
                    adc_stream=adc_st[i],
                )
                # Wait for the resonator to deplete
                wait(node.parameters.depletion_time * u.ns, resonator)
                if not node.parameters.multiplexed:
                    align()
            align()

        with stream_processing():
            n_st.save("n")
            for i, resonator in enumerate(resonators):
                # Will save average:
                adc_st[i].input1().real().average().save(f"adcI{i + 1}")
                adc_st[i].input1().image().average().save(f"adcQ{i + 1}")
                # Will save only last run:
                adc_st[i].input1().real().save(f"adc_single_runI{i + 1}")
                adc_st[i].input1().image().save(f"adc_single_runQ{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, None]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, None]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    qm = qmm.open_qm(config)
    # The job is stored in the node namespace to be reused in the fetching_data run_action
    node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
    # Names and values mapping
    keys = []

    for i in range(1, len(node.parameters.resonators) + 1):
        keys.extend([f"adcI{i}", f"adcQ{i}", f"adc_single_runI{i}", f"adc_single_runQ{i}"])
    data_fetcher = fetching_tool(job, data_list=keys, mode="wait_for_all")
    values = data_fetcher.fetch_all()
    # Display the execution report to expose possible runtime errors
    node.log(job.execution_report())
    # # Register the raw dataset
    for key, value in zip(keys, values):
        node.results[key] = value


# %% {Data_loading_and_dataset_creation}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, None]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id


# %% {Data_analysis}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, None]):
    """Analyse the raw data and store the fitted data in node.results."""
    process_raw_data(node.results)
    fit_raw_data(node.results, node)


# %% {Plotting}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, None]):
    """Plot the raw and fitted data."""
    num_resonators = len(node.parameters.resonators)
    fig_single_run_fit = plot_single_run_with_fit(num_resonators, node.results)
    fig_averaged_run_fit = plot_averaged_run_with_fit(num_resonators, node.results)
    node.results["figures"] = {
        "single_run": fig_single_run_fit,
        "averaged_run": fig_averaged_run_fit,
    }
    plt.show()


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, None]):
    node.save()

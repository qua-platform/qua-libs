# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibrate.utils.logger_m import logger
from quam_config import QuAM
from quam_experiments.experiments.time_of_flight_mw import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_single_run_with_fit,
    plot_averaged_run_with_fit,
)
from quam_experiments.parameters.qubits_experiment import get_qubits
from quam_experiments.workflow import simulate_and_plot
from qualibration_libs.xarray_data_fetcher import XarrayDataFetcher
from qualibration_libs.trackable_object import tracked_updates

description = """
        TIME OF FLIGHT
This sequence involves sending a readout pulse and capturing the raw ADC traces.
The data undergoes post-processing to calibrate three distinct parameters:
    - Time of Flight: This represents the internal processing time and the propagation
      delay of the readout pulse. Its value can be adjusted in the configuration under
      "time_of_flight". This value is utilized to offset the acquisition window relative
      to when the readout pulse is dispatched.

    - Analog Inputs Offset: Due to minor impedance mismatches, the signals captured by
      the OPX might exhibit slight offsets.

    - Analog Inputs Gain: If a signal is constrained by digitization or if it saturates
      the ADC, the variable gain of the OPX analog input, ranging from -12 dB to 20 dB,
      can be modified to fit the signal within the ADC range of +/-0.5V.
"""


node = QualibrationNode[Parameters, QuAM](
    name="01b_time_of_flight_mw_fem", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1", "q2", "q3", "q4"]
    pass


# Instantiate the QuAM class from the state file
node.machine = QuAM.load()


# %% {QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    node.namespace["tracked_resonators"] = [] = []
    for q in qubits:
        resonator = q.resonator
        # make temporary updates before running the program and revert at the end.
        with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            if node.parameters.time_of_flight_in_ns is not None:
                resonator.time_of_flight = node.parameters.time_of_flight_in_ns
            resonator.operations["readout"].length = node.parameters.readout_length_in_ns
            resonator.set_output_power(node.parameters.readout_amplitude_in_dBm, operation="readout")
            node.namespace["tracked_resonators"].append(resonator)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "readout_time": xr.DataArray(
            np.arange(0, node.parameters.readout_length_in_ns, 1),
            attrs={"long_name": "readout time", "units": "ns"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)  # QUA variable for the averaging loop
        n_st = declare_stream()
        adc_st = [declare_stream(adc_trace=True) for _ in range(num_qubits)]  # The stream to store the raw ADC trace

        for multiplexed_qubits in qubits.batch():
            with for_(n, 0, n < node.parameters.num_averages, n + 1):
                save(n, n_st)
                for i, qubit in multiplexed_qubits.items():
                    # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
                    reset_if_phase(qubit.resonator.name)
                    # Measure the resonator (send a readout pulse and record the raw ADC trace)
                    qubit.resonator.measure("readout", stream=adc_st[i])
                    # Wait for the resonator to deplete
                    qubit.resonator.wait(node.machine.depletion_time * u.ns)

        with stream_processing():
            n_st.save("n")
            for i, qubit in enumerate(node.namespace["qubits"]):
                if qubit.resonator.opx_input.port_id == 1:
                    stream = adc_st[i].input1()
                else:
                    stream = adc_st[i].input2()
                # Will save average:
                stream.real().average().save(f"adcI{i + 1}")
                stream.image().average().save(f"adcQ{i + 1}")
                # Will save only last run:
                stream.real().save(f"adc_single_runI{i + 1}")
                stream.image().save(f"adc_single_runQ{i + 1}")


# %% {Simulate_or_execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            # print_progress_bar(job, iteration_variable="n", total_number_of_iterations=node.parameters.num_averages)
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_averages,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        print(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset
    node.save()


# %% {Data_loading_and_dataset_creation}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, QuAM]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node = node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Data_analysis}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, QuAM]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], logger)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plotting}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, QuAM]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_single_run_fit = plot_single_run_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    fig_averaged_run_fit = plot_averaged_run_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    plt.show()
    # Store the generated figures
    node.results["figure_single_run"] = fig_single_run_fit
    node.results["figure_averaged_run"] = fig_averaged_run_fit


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, QuAM]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    # Revert the change done at the beginning of the node
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if not node.results["fit_results"][q.name]["success"]:
                continue

            fit_result = node.results["fit_results"][q.name]
            if node.parameters.time_of_flight_in_ns is not None:
                q.resonator.time_of_flight = node.parameters.time_of_flight_in_ns + fit_result["tof_to_add"]
            else:
                q.resonator.time_of_flight += fit_result["tof_to_add"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

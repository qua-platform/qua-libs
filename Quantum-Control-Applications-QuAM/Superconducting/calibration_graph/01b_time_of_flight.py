# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibrate.utils.logger_m import logger
from quam_config import QuAM
from quam_experiments.experiments.time_of_flight import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from quam_experiments.parameters.qubits_experiment import get_qubits
from quam_experiments.workflow import simulate_and_plot
from qualibration_libs.xarray_data_fetcher import XarrayDataFetcher
from qualibration_libs.trackable_object import tracked_updates


# %% {Initialisation}
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


node = QualibrationNode[Parameters, QuAM](name="01b_time_of_flight", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1", "q3"]
    pass


# Instantiate the QuAM class from the state file
node.machine = QuAM.load()


# %% {Create_QUA_program}
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
            resonator.time_of_flight = node.parameters.time_of_flight_in_ns
            resonator.operations["readout"].length = node.parameters.readout_length_in_ns
            resonator.operations["readout"].amplitude = node.parameters.readout_amplitude_in_v
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
        adc_st = [declare_stream(adc_trace=True) for _ in range(num_qubits)]  # The stream to store the raw ADC trace

        for multiplexed_qubits in qubits.batch():
            with for_(n, 0, n < node.parameters.num_averages, n + 1):
                for i, qubit in multiplexed_qubits.items():
                    # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
                    reset_phase(qubit.resonator.name)
                    # Measure the resonator (send a readout pulse and record the raw ADC trace)
                    qubit.resonator.measure("readout", stream=adc_st[i])
                    # Wait for the resonator to deplete
                    qubit.resonator.wait(node.machine.depletion_time * u.ns)

        with stream_processing():
            for i in range(num_qubits):
                # Will save average:
                adc_st[i].input1().average().save(f"adcI{i + 1}")
                adc_st[i].input2().average().save(f"adcQ{i + 1}")
                # Will save only last run:
                adc_st[i].input1().save(f"adc_single_runI{i + 1}")
                adc_st[i].input2().save(f"adc_single_runQ{i + 1}")


# %% {Simulate}
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


# %% {Execute}
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


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, QuAM]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node = node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
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


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, QuAM]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figure_amplitude"] = fig_raw_fit


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, QuAM]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    ds = node.results["ds"]

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            # if node.parameters.time_of_flight_in_ns is not None:
            #     q.resonator.time_of_flight = node.parameters.time_of_flight_in_ns + int(ds.sel(qubit=q.name).delays)
            # else:
            #     q.resonator.time_of_flight += int(ds.sel(qubit=q.name).delays)

    # Revert the change done at the beginning of the node
    for resonator in node.namespace["tracked_resonators"]:
        resonator.revert_changes()

    # # Update the offsets per controller for each qubit
    # for con in np.unique(ds.con.values):
    #     for i, q in enumerate(ds.where(ds.con == con).qubit.values):
    #         resonator = node.machine.qubits[q].resonator
    #         # Only add the offsets once,
    #         if i == 0:
    #             if resonator.opx_input_I.offset is not None:
    #                 resonator.opx_input_I.offset += float(ds.where(ds.con == con).offsets_I.mean(dim="qubit").values)
    #             else:
    #                 resonator.opx_input_I.offset = float(ds.where(ds.con == con).offsets_I.mean(dim="qubit").values)
    #             if resonator.opx_input_Q.offset is not None:
    #                 resonator.opx_input_Q.offset += float(ds.where(ds.con == con).offsets_Q.mean(dim="qubit").values)
    #             else:
    #                 resonator.opx_input_Q.offset = float(ds.where(ds.con == con).offsets_Q.mean(dim="qubit").values)
    #         # else copy the values from the updated qubit
    #         else:
    #             resonator.opx_input_I.offset = node.machine.qubits[
    #                 ds.where(ds.con == con).qubit.values[0]
    #             ].resonator.opx_input_I.offset
    #             resonator.opx_input_Q.offset = node.machine.qubits[
    #                 ds.where(ds.con == con).qubit.values[0]
    #             ].resonator.opx_input_Q.offset


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

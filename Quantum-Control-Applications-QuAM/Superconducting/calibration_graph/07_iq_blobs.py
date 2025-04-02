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
from quam_experiments.experiments.iq_blobs import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_iq_blobs,
    plot_confusion_matrices,
)
from quam_experiments.parameters.qubits_experiment import get_qubits
from quam_experiments.workflow import simulate_and_plot
from qualibration_libs.xarray_data_fetcher import XarrayDataFetcher


# %% {Description}
description = """
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after
thermalization (with the qubit in the |g> state) and then after applying a pi pulse
to the qubit (bringing the qubit to the |e> state) successively. The resulting IQ blobs
are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the 
      separation between |g> and |e> states aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having calibrated a pi pulse (node 04b_power_rabi.py).

State update:
    - The integration weight angle
    - the ge discrimination threshold
    - the Repeat Until Success threshold
    - The confusion matrix
"""

# Be sure to include [Parameters, QuAM] so the node has proper type hinting
node = QualibrationNode[Parameters, QuAM](
    name="07_iq_blobs",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the Qualibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    """
    Allow the user to locally set the node parameters for debugging purposes, or
    execution in the Python IDE.
    """
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1", "q3"]
    pass


# Instantiate the QuAM class from the state file
node.machine = QuAM.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """
    Create the sweep axes and generate the QUA program from the pulse sequence and the
    node parameters.
    """
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_runs = node.parameters.num_runs  # Number of runs
    operation = node.parameters.operation
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}),
    }

    with program() as node.namespace["qua_program"]:
        I_g, I_g_st, Q_g, Q_g_st, n, n_st = node.machine.qua_declaration()
        I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.qua_declaration()

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)

                # Ground state iq blobs for all qubits
                # Qubit initialization
                for i, qubit in multiplexed_qubits.items():
                    qubit.reset_qubit(node.parameters.reset_type, node.parameters.simulate)
                align()
                # Qubit readout
                for i, qubit in multiplexed_qubits.items():
                    qubit.resonator.measure(operation, qua_vars=(I_g[i], Q_g[i]))
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    # save data
                    save(I_g[i], I_g_st[i])
                    save(Q_g[i], Q_g_st[i])
                align()

                # Excited state iq blobs for all qubits
                # Qubit initialization
                for i, qubit in multiplexed_qubits.items():
                    qubit.reset_qubit(node.parameters.reset_type, node.parameters.simulate)
                align()

                # Qubit readout
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    qubit.resonator.measure(operation, qua_vars=(I_e[i], Q_e[i]))
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    # save data
                    save(I_e[i], I_e_st[i])
                    save(Q_e[i], Q_e_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_g_st[i].save_all(f"Ig{i + 1}")
                Q_g_st[i].save_all(f"Qg{i + 1}")
                I_e_st[i].save_all(f"Ie{i + 1}")
                Q_e_st[i].save_all(f"Qe{i + 1}")


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
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in
    an xarray dataset called "ds_raw".
    """
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
                node.parameters.num_runs,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        print(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)


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
    """
    Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary.
    """
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
    """
    Plot the raw and fitted data in specific figures whose shape is given by
    qubit.grid_location.
    """
    fig_iq = plot_iq_blobs(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_confusion = plot_confusion_matrices(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figure_iq_blobs"] = fig_iq
    node.results["figure_confusion_matrix"] = fig_confusion


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, QuAM]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_result = node.results["fit_results"][q.name]
            operation = q.resonator.operations[node.parameters.operation]
            operation.integration_weights_angle -= float(fit_result["iw_angle"])
            # Convert the thresholds back in demod units
            # todo: what about when calibrating qnd_readout, would state discrimination be wrong?
            operation.threshold = float(fit_result["ge_threshold"]) * operation.length / 2**12
            operation.rus_exit_threshold = float(fit_result["rus_threshold"]) * operation.length / 2**12
            # todo: add conf matrix to the readout operation rather than the resonator
            if node.parameters.operation == "readout":
                q.resonator.confusion_matrix = fit_result["confusion_matrix"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

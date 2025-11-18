# # -------------------------------------------
# # -------------------------------------------
# # -------------------------------------------
from quam_qd_generator_example import config, config_path
# # -------------------------------------------
# # -------------------------------------------
# # -------------------------------------------


# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
from calibration_utils.iq_blobs.plotting import plot_historams, plot_visibility_curves
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.iq_blobs import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_iq_blobs,
    plot_confusion_matrices,
    simulate_quantum_dot_readout_from_node,
)
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """

"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="xx_iq_blobs",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """
    Allow the user to locally set the node parameters for debugging purposes, or
    execution in the Python IDE.
    """
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    node.parameters.qubit_pairs = ['Q0_Q1']
    node.parameters.simulate = True


# Instantiate the QUAM class from the state file

node.machine = Quam.load(config_path)


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Create the sweep axes and generate the QUA program from the pulse sequence and the
    node parameters.
    """
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    n_runs = node.parameters.num_shots  # Number of runs
    operation = node.parameters.operation
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "n_runs": xr.DataArray(np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}),
    }

    with program() as node.namespace["qua_program"]:
        I_g, I_g_st, Q_g, Q_g_st, n, n_st = node.machine.declare_qua_variables()
        I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit_pair in multiplexed_qubit_pairs.values():
                # -----------------------------------------
                # Initialise qpu
                # -----------------------------------------
                # node.machine.initialize_qpu(target='idle')
                pass

            align()

            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)

                # Qubit readout
                for i, qubit_pair in multiplexed_qubit_pairs.items():
                    # -----------------------------------------
                    # Initialise quantum dot pair - mixed or GS
                    # -----------------------------------------
                    qubit_pair.quantum_dot_pair.run_sequence('initialise')
                    # -----------------------------------------
                    # Readout quantum dot pair
                    # -----------------------------------------
                    # Measure the state of the resonators
                    qubit_pair.quantum_dot_pair.run_sequence('measure')
                    if len(qubit_pair.quantum_dot_pair.sensor_dots)!=1:
                        raise NotImplementedError(
                            f'Handling of multiple sensor dots is not implemented yet.'
                        )
                    qubit_pair.quantum_dot_pair.sensor_dots[0].measure(
                        operation, qua_vars=(I_g[i], Q_g[i])
                    )
                    save(I_g[i], I_g_st[i])
                    save(Q_g[i], Q_g_st[i])

                align()

                # Qubit readout
                for i, qubit_pair in multiplexed_qubit_pairs.items():
                    # -----------------------------------------
                    # Initialise quantum dot pair - ES or GS then xy pulse
                    # -----------------------------------------
                    qubit_pair.quantum_dot_pair.run_sequence('initialise')
                    # qubit_pair.quantum_dot[0].xy.play("x180")
                    # -----------------------------------------
                    # Readout quantum dot pair
                    # -----------------------------------------
                    # Measure the state of the resonators
                    qubit_pair.quantum_dot_pair.run_sequence('measure')
                    if len(qubit_pair.quantum_dot_pair.sensor_dots)!=1:
                        raise NotImplementedError(
                            f'Handling of multiple sensor dots is not implemented yet.'
                        )
                    qubit_pair.quantum_dot_pair.sensor_dots[0].measure(
                        operation, qua_vars=(I_g[i], Q_g[i])
                    )
                    save(I_g[i], I_g_st[i])
                    save(Q_g[i], Q_g_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                I_g_st[i].buffer(n_runs).save(f"Ig{i + 1}")
                Q_g_st[i].buffer(n_runs).save(f"Qg{i + 1}")
                I_e_st[i].buffer(n_runs).save(f"Ie{i + 1}")
                Q_e_st[i].buffer(n_runs).save(f"Qe{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Simulate the QUA program for waveform visualization AND generate realistic
    quantum dot readout data for testing the analysis and plotting.
    """
    # Connect to the QOP
    # qmm = node.machine.connect()
    # # Get the config from the machine
    # config = node.machine.generate_config()
    # # Simulate the QUA program, generate the waveform report and plot the simulated samples
    # samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # # Store the figure, waveform report and simulated samples
    # node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}

    # Generate realistic quantum dot IQ blob data using Barthel model
    node.log("Generating realistic simulated IQ data using Barthel model...")
    ds_simulated = simulate_quantum_dot_readout_from_node(node, add_noise_variation=True, seed=42)
    # Process the simulated data (convert to voltage units)
    node.results["ds_raw"] = process_raw_dataset(ds_simulated, node)
    node.log(f"Simulated {len(node.namespace['qubit_pairs'])} quantum dot pairs with {node.parameters.num_shots} shots each")


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw".
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
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action()
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """
    Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary.
    """
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action()
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """
    Plot the raw and fitted data in specific figures whose shape is given by
    qubit.grid_location.
    """
    fig_iq = plot_iq_blobs(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"])
    fig_confusion = plot_confusion_matrices(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"])
    fig_histogram = plot_historams(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"])
    fig_visibility = plot_visibility_curves(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "iq_blobs": fig_iq,
        "confusion_matrix": fig_confusion,
        "histograms": fig_histogram,
        "visibility_curves": fig_visibility,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubit_pairs"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_result = node.results["fit_results"][q.name]
            operation = q.resonator.operations[node.parameters.operation]
            operation.integration_weights_angle -= float(fit_result["iw_angle"])
            # Convert the thresholds back to demod units
            operation.threshold = float(fit_result["ge_threshold"]) * operation.length / 2**12
            operation.rus_exit_threshold = float(fit_result["rus_threshold"]) * operation.length / 2**12
            if node.parameters.operation == "readout":
                q.resonator.confusion_matrix = fit_result["confusion_matrix"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

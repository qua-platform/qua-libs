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
from quam_config import Quam
from calibration_utils.confusion_matrix_2q import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.data_process_utils import *
from qualibration_libs.parameters import get_qubit_pairs, get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Description}
description = """
Two-Qubit Readout Confusion Matrix Measurement

This sequence measures the readout error when simultaneously measuring the state of two qubits. The process involves:

1. Preparing the two qubits in all possible combinations of computational basis states (|00⟩, |01⟩, |10⟩, |11⟩)
2. Performing simultaneous readout on both qubits
3. Calculating the confusion matrix based on the measurement results

For each prepared state, we measure:
1. The readout result of the first qubit
2. The readout result of the second qubit

The measurement process involves:
1. Initializing both qubits to the ground state
2. Applying single-qubit gates to prepare the desired input state
3. Performing simultaneous readout on both qubits
4. Repeating the process multiple times to gather statistics

The outcome of this measurement will be used to:
1. Quantify the readout fidelity for two-qubit states
2. Identify and characterize crosstalk effects in the readout process
3. Provide data for readout error mitigation in two-qubit experiments

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits

Outcomes:
- 4x4 confusion matrix representing the probabilities of measuring each two-qubit state given a prepared input state
- Readout fidelity metrics for simultaneous two-qubit measurement
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="51a_2Q_confusion_matrix",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubit_pairs = ["q1-2", "q3-4"]
    # # node.parameters.simulate = True
    # # node.parameters.simulation_duration_ns = 6000
    # node.parameters.num_shots = 100
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Update the readout power to match the desired range, this change will be reverted at the end of the node.
    node.namespace["tracked_qubit_pairs"] = []
    for qp in qubit_pairs:
        with tracked_updates(qp, auto_revert=False) as qp:
            pass
        node.namespace["tracked_qubit_pairs"].append(qp)

    n_shots = node.parameters.num_shots  # The number of averages
    state_discrimination = node.parameters.use_state_discrimination

    assert node.parameters.use_state_discrimination, "this protocol requires to use state discriminations"
    # Control states
    prepared_states = ["00", "01", "10", "11"]

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "n_shots": xr.DataArray(np.arange(n_shots), attrs={"long_name": "number of shots"}),
        "prepared_state": xr.DataArray(prepared_states, attrs={"long_name": "prepared initial state"}),
    }

    with program() as node.namespace["qua_program"]:
        _, _, _, _, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        state_c = [declare(int) for _ in range(num_qubit_pairs)]
        state_t = [declare(int) for _ in range(num_qubit_pairs)]
        state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
        state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
        s_c = declare(int)
        s_t = declare(int)

        # Reset explicitly
        reset_global_phase()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_shots, n + 1):
                save(n, n_st)

                with for_(s_c, 0, s_c < 2, s_c + 1):
                    with for_(s_t, 0, s_t < 2, s_t + 1):
                        # Reset the qubits to the ground state
                        for i, qp in multiplexed_qubit_pairs.items():
                            qc = qp.qubit_control
                            qt = qp.qubit_target
                            zz = qp.zz_drive
                            zz_elems = [zz.name, qc.xy.name, qt.xy.name, qt.xy_detuned.name]

                            # Reset the qubits to the ground state
                            qc.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                log_callable=node.log,
                            )
                            qt.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                log_callable=node.log,
                            )
                            align(*zz_elems)

                            # Prepare initial states
                            with if_(s_c == 1):  # states 0:g or 1:e
                                qc.xy.play("x180")
                            with else_():  # states 0:g or 1:e
                                qc.xy.wait(qc.xy.operations["x180"].length // 4)

                            with if_(s_t == 1):  # states 0:g or 1:e
                                qt.xy.play("x180")
                            with else_():  # states 0:g or 1:e
                                qt.xy.wait(qt.xy.operations["x180"].length // 4)
                            align(*zz_elems)

                            # Measure the state of the resonators
                            qc.readout_state(state_c[i])
                            qt.readout_state(state_t[i])
                            save(state_c[i], state_c_st[i])
                            save(state_t[i], state_t_st[i])

                            # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                            qc.resonator.wait(qc.resonator.depletion_time // 4)
                            qt.resonator.wait(qt.resonator.depletion_time // 4)

        with stream_processing():
            n_st.save("n")
            for i, qp in enumerate(qubit_pairs):
                state_c_st[i].buffer(4).buffer(n_shots).save(f"state_c{i + 1}")
                state_t_st[i].buffer(4).buffer(n_shots).save(f"state_t{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()

    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
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
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())

    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
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
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    figs_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {f"IQ_{qp.name}": fig for fig, qp in zip(figs_raw_fit, node.namespace["qubit_pairs"])}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    # Revert the change done at the beginning of the node
    for tracked_qubit_pair in node.namespace.get("tracked_qubit_pairs", []):
        tracked_qubit_pair.zz_drive.revert_changes()
        tracked_qubit_pair.qubit_control.revert_changes()
        tracked_qubit_pair.qubit_target.revert_changes()

    with node.record_state_updates():
        for i, qp in enumerate(node.namespace["qubit_pairs"]):
            if node.outcomes[qp.name] == "failed":
                continue

            qp.confusion = node.results["ds_fit"].sel(qubit_pair=qp.name).prob.values.tolist()


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%

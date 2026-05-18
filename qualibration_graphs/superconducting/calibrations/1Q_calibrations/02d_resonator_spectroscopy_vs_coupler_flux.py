"""Resonator spectroscopy versus coupler flux calibration."""

# %% {Imports}
from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.resonator_spectroscopy_vs_coupler_flux import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Node initialisation}
description = """
        RESONATOR SPECTROSCOPY VERSUS COUPLER FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout frequencies and coupler flux biases.
The resonator frequency as a function of coupler flux bias is then extracted and fitted in order to extract the relevant
coupler flux biases and corresponding readout frequency.

This information can then be used to adjust the readout frequency for different coupler flux points.

Prerequisites:
    - Having calibrated the resonator frequency (nodes 02a, 02b and/or 02c).
    - Having specified the desired flux point (qubit.z.flux_point).

State update:
    - update coupler flux biases
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="02d_resonator_spectroscopy_vs_coupler_flux",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubit_pairs = ["coupler_q1_q2", "coupler_q2_q3"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Extract the measured qubits based on measure_qubit parameter
    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    # Also set in qubits for compatibility with utility functions
    node.namespace["qubits"] = measured_qubits

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    # Coupler flux bias sweep in V
    dcs = np.linspace(
        node.parameters.min_flux_offset_in_v,
        node.parameters.max_flux_offset_in_v,
        node.parameters.num_flux_points,
    )
    # The frequency sweep around the resonator resonance frequency
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)

    # Buffer time for flux to wait (in unit of clock cycle)
    settle_time = node.parameters.settle_time_in_ns // 4

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "flux_bias": xr.DataArray(dcs, attrs={"long_name": "coupler flux bias", "units": "V"}),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_qubit_pairs)
        dc = declare(fixed)  # QUA variable for the coupler flux bias
        df = declare(int)  # QUA variable for the readout frequency detuning

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU for all qubits in the batch
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(dc, dcs)):
                    # Apply coupler flux bias via DC offset for EACH qubit pair's coupler
                    for ii, qp in multiplexed_qubit_pairs.items():
                        qp.coupler.set_dc_offset(dc)
                        # Wait for coupler to settle
                        qp.coupler.wait(settle_time)
                    # align all the couplers
                    align()
                    # Measure all qubits in the batch
                    with for_(*from_array(df, dfs)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            measured_qubit = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            rr = measured_qubit.resonator
                            # Update the resonator frequency
                            rr.update_frequency(df + rr.intermediate_frequency)
                            # Readout the resonator
                            rr.measure("readout", qua_vars=(I[ii], Q[ii]))
                            # Wait for the resonator to deplete
                            rr.wait(rr.depletion_time * u.ns)
                            # Save data
                            save(I[ii], I_st[ii])
                            save(Q[ii], Q_st[ii])
                        align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                I_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"Q{i + 1}")


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
    """Connect to the QOP, execute the QUA program and fetch the raw data,
    storing it in a xarray dataset called "ds_raw"."""
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
    # Rename qubit_pair dimension to qubit for compatibility with analysis functions
    if "qubit_pair" in dataset.dims:
        qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
        measured_qubit_names = [q.name for q in node.namespace["measured_qubits"]]
        dataset = dataset.rename({"qubit_pair": "qubit"})
        dataset = dataset.assign_coords(qubit=qubit_pair_names)
        dataset = dataset.assign_coords(measured_qubit_name=("qubit", measured_qubit_names))
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id

    # Get the active qubit pairs from the loaded node parameters
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    # Extract measured qubits based on measure_qubit parameter
    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    node.namespace["qubits"] = measured_qubits

    # Rename qubit_pair dimension to qubit, keeping the qubit-pair names as
    # the coordinate (they are unique even when two pairs share a measured qubit).
    if "qubit_pair" in node.results["ds_raw"].dims:
        qubit_pair_names = [qp.name for qp in qubit_pairs]
        measured_qubit_names = [q.name for q in measured_qubits]
        node.results["ds_raw"] = node.results["ds_raw"].rename({"qubit_pair": "qubit"})
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(qubit=qubit_pair_names)
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(
            measured_qubit_name=("qubit", measured_qubit_names)
        )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary."""
    # Ensure qubits namespace is set for utility functions
    if "qubits" not in node.namespace:
        node.namespace["qubits"] = node.namespace["measured_qubits"]

    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    # fit_results is keyed by the qubit coordinate, which is the qubit-pair name.
    qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
    node.outcomes = {
        qp_name: node.results["fit_results"].get(qp_name, {}).get("success", False) for qp_name in qubit_pair_names
    }
    # Convert boolean outcomes to "successful"/"failed" strings
    node.outcomes = {k: ("successful" if v else "failed") for k, v in node.outcomes.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data, laid out by coupler chip position using QubitPairGrid."""
    # Ensure qubits namespace is set for plotting function
    if "qubits" not in node.namespace:
        node.namespace["qubits"] = node.namespace["measured_qubits"]

    fig_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        node.namespace["measured_qubits"],
        node.results["ds_fit"],
    )
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue

            # Access the measured qubit-pair if needed:
            # fit_results = node.results["fit_results"][qp.name]
            # Update manually the state according to the data


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()

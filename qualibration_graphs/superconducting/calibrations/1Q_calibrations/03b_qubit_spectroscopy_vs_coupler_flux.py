# %% {Imports}
import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.qubit_spectroscopy_vs_coupler import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
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

# %% {Description}
description = """
        QUBIT SPECTROSCOPY VERSUS FLUX
This sequence involves doing a qubit spectroscopy for several flux biases in order to exhibit the qubit frequency
versus flux response.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit frequency (node 03a_qubit_spectroscopy.py).
    - Having specified the desired flux point (qubit.z.flux_point).

State update:
    - The qubit 0->1 frequency at the set flux point: qubit.f_01 & qubit.xy.RF_frequency
    - The flux bias corresponding to the set flux point: q.z.independent_offset or q.z.joint_offset.
"""


node = QualibrationNode[Parameters, Quam](
    name="qubit_spectroscopy_vs_coupler",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubit_pairs = ["qB2-B3"]
    node.parameters.measure_qubit = "target"
    node.parameters.operation_amplitude_factor = 0.2
    node.parameters.operation = "x180"
    node.parameters.operation_len_in_ns = 200
    node.parameters.num_shots = 300
    node.parameters.frequency_span_in_mhz = 70
    node.parameters.frequency_step_in_mhz = 1
    node.parameters.min_flux = -2.2
    node.parameters.max_flux = 0.5
    node.parameters.num_flux_points = 101

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

    operation = node.parameters.operation  # The qubit operation to play
    n_avg = node.parameters.num_shots
    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    operation_len = node.parameters.operation_len_in_ns
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)
    # Flux bias sweep in V
    min_flux = node.parameters.min_flux * u.V
    max_flux = node.parameters.max_flux * u.V
    num = node.parameters.num_flux_points
    dcs = np.linspace(min_flux, max_flux, num)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "flux_bias": xr.DataArray(dcs, attrs={"long_name": "flux bias", "units": "V"}),
    }

    with program() as node.namespace["qua_program"]:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit pairs
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the qubit frequency
        dc = declare(fixed)  # QUA variable for the flux dc level

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    with for_(*from_array(dc, dcs)):
                        # Select the measured qubit based on measure_qubit parameter
                        for ii, qp in multiplexed_qubit_pairs.items():
                            if node.parameters.measure_qubit == "control":
                                protagonist_qubit = qp.qubit_control
                            else:
                                protagonist_qubit = qp.qubit_target

                            # Update the qubit frequency
                            protagonist_qubit.xy.update_frequency(df + protagonist_qubit.xy.intermediate_frequency)
                            # Wait for the qubits to decay to the ground state
                            protagonist_qubit.reset_qubit_thermal()
                            # Flux sweeping for a qubit
                            duration = (
                                operation_len * u.ns
                                if operation_len is not None
                                else protagonist_qubit.xy.operations[operation].length * u.ns
                            )
                        align()

                        # Qubit manipulation
                        for ii, qp in multiplexed_qubit_pairs.items():
                            if node.parameters.measure_qubit == "control":
                                protagonist_qubit = qp.qubit_control
                            else:
                                protagonist_qubit = qp.qubit_target

                            # Bring the qubit to the desired point during the saturation pulse
                            qp.coupler.play(
                                "const",
                                amplitude_scale=dc / qp.coupler.operations["const"].amplitude,
                                duration=duration + 200,
                            )
                            protagonist_qubit.xy.wait(50)
                            # Apply saturation pulse to the measured qubit
                            protagonist_qubit.xy.play(
                                operation,
                                amplitude_scale=operation_amp,
                                duration=duration,
                            )
                        align()

                        # Qubit readout - only measure the selected qubit
                        for ii, qp in multiplexed_qubit_pairs.items():
                            if node.parameters.measure_qubit == "control":
                                protagonist_qubit = qp.qubit_control
                            else:
                                protagonist_qubit = qp.qubit_target

                            protagonist_qubit.resonator.wait(50)
                            protagonist_qubit.resonator.measure("readout", qua_vars=(I[ii], Q[ii]))
                            # save data
                            save(I[ii], I_st[ii])
                            save(Q[ii], Q_st[ii])

            # Measure sequentially
            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                I_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"Q{i + 1}")


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
    """
    Connect to the QOP, execute the QUA program and fetch the raw data
    and store it in a xarray dataset called "ds_raw".
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
    # Rename qubit_pair dimension to qubit for compatibility with analysis functions
    if "qubit_pair" in dataset.dims:
        qubit_names = [q.name for q in node.namespace["measured_qubits"]]
        dataset = dataset.rename({"qubit_pair": "qubit"})
        dataset = dataset.assign_coords(qubit=qubit_names)
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
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    # Extract measured qubits
    measured_qubits = []
    for qp in node.namespace["qubit_pairs"]:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    node.namespace["qubits"] = measured_qubits
    # Rename qubit_pair dimension to qubit for compatibility with analysis functions
    if "qubit_pair" in node.results["ds_raw"].dims:
        qubit_names = [q.name for q in measured_qubits]
        node.results["ds_raw"] = node.results["ds_raw"].rename({"qubit_pair": "qubit"})
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(qubit=qubit_names)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """
    Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary.
    """
    # Ensure qubits namespace is set for utility functions
    if "qubits" not in node.namespace:
        node.namespace["qubits"] = node.namespace["measured_qubits"]

    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    # Map outcomes to qubit_pair names for state update
    qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
    measured_qubit_names = [q.name for q in node.namespace["measured_qubits"]]
    node.outcomes = {
        qubit_pair_name: node.results["fit_results"].get(measured_qubit_name, {}).get("success", False)
        for qubit_pair_name, measured_qubit_name in zip(qubit_pair_names, measured_qubit_names)
    }
    # Convert boolean outcomes to "successful"/"failed" strings
    node.outcomes = {k: ("successful" if v else "failed") for k, v in node.outcomes.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    # Ensure qubits namespace is set for plotting function
    if "qubits" not in node.namespace:
        node.namespace["qubits"] = node.namespace["measured_qubits"]

    fig_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"], node.namespace["measured_qubits"], node.results["ds_fit"]
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
            else:
                # fit_results = node.results["fit_results"][measured_qubit_name]
                # Store avoided crossing positions for reference
                # The avoided crossings are recorded but not automatically used to update state
                # This allows the user to manually inspect and decide how to use them
                # Example: could store in qubit metadata or coupler configuration
                pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%

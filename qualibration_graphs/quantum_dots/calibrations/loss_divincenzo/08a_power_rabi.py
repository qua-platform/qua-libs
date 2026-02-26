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
from calibration_utils.common_utils.experiment import get_qubits
from calibration_utils.power_rabi import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        POWER RABI
This sequence involves parking the qubit at the manipulation bias point, playing the qubit pulse (such as x180) and
measuring the state of the resonator across different qubit pulse amplitudes, exhibit Rabi oscillations.
The results are then analyzed to determine the qubit pulse amplitude suitable for the selected gate duration.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the qubit frequency.
    - Having set the qubit gate duration.

State update:
    - The qubit pulse amplitude corresponding to the specified operation (x180, x90...).
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="08a_power_rabi",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # The number of averages
    operation = node.parameters.operation  # The qubit operation to play
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
    }

    with program() as node.namespace["qua_program"]:
        # Declare QUA variables
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        n = declare(int)

        # Additional variables for pre/post measurement comparison
        # Use int instead of bool so we can average in stream processing
        p1 = declare(int, size=num_qubits)
        p2 = declare(int, size=num_qubits)
        # Streams keyed by qubit name; each qubit appears in exactly one batch.
        p1_st = {qubit.name: declare_stream() for qubit in qubits}
        p2_st = {qubit.name: declare_stream() for qubit in qubits}
        pdiff_st = {qubit.name: declare_stream() for qubit in qubits}
        n_st = declare_stream()

        # Main experiment loop
        for batched_qubits in qubits.batch():
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(a, amps)):
                    # ---------------------------------------------------------
                    # Step 1: Empty - step to empty point (fixed duration)
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        qubit.empty()

                    align()
                    for i, qubit in batched_qubits.items():
                        # Measure initial state (includes step_to('readout'))
                        # Cast bool to int for stream averaging
                        assign(p1[i], Cast.to_int(qubit.measure()))

                    # ---------------------------------------------------------
                    # Step 2: Initialize - load electron into dot
                    # ---------------------------------------------------------
                    align()

                    for i, qubit in batched_qubits.items():
                        # Get operation duration in ns from QUAM state
                        op_length_ns = qubit.macros["x180"].duration
                        qubit.initialize(duration=op_length_ns + node.parameters.gap_wait_time_in_ns + 1000)

                    # ---------------------------------------------------------
                    # Step 3: Apply qubit pulse with variable amplitude
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        qubit.x180(amplitude_scale=a)

                    # Synchronize before measurement
                    align()

                    # ---------------------------------------------------------
                    # Step 4: Measure - move to PSB and measure
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        # Measure macro handles step_to('readout') + measurement
                        # Cast bool to int for stream averaging
                        assign(p2[i], Cast.to_int(qubit.measure()))

                    # Synchronize before compensation
                    align()

                    # ---------------------------------------------------------
                    # Step 5: Apply compensation pulse to reset DC bias
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        qubit.voltage_sequence.apply_compensation_pulse()

                    # ---------------------------------------------------------
                    # Save results
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        save(p1[i], p1_st[qubit.name])
                        save(p2[i], p2_st[qubit.name])

                        # Calculate state difference
                        with if_(p1[i] == p2[i]):
                            save(0, pdiff_st[qubit.name])
                        with else_():
                            save(1, pdiff_st[qubit.name])

        # Stream processing
        with stream_processing():
            n_st.save("n")

            n_amps = len(amps)

            for qubit in qubits:
                p1_st[qubit.name].buffer(n_amps).average().save(f"p1_{qubit.name}")
                p2_st[qubit.name].buffer(n_amps).average().save(f"p2_{qubit.name}")
                pdiff_st[qubit.name].buffer(n_amps).average().save(f"pdiff_{qubit.name}")


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
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
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
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    ds_fit, fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {qname: ("successful" if r["success"] else "failed") for qname, r in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubits"],
        node.results.get("fit_results", {}),
    )
    node.results["figure"] = fig


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            opt_prefactor = node.results["fit_results"][q.name]["opt_amp"]
            current_amp = q.xy.operations[node.parameters.operation].amplitude
            new_amplitude = current_amp * opt_prefactor
            q.xy.operations[node.parameters.operation].amplitude = new_amplitude
            if node.parameters.operation == "x180" and node.parameters.update_x90:
                q.xy.operations["x90"].amplitude = new_amplitude / 2


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

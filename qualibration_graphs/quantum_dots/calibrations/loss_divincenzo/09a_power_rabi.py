# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam

from calibration_utils.power_rabi import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_all,
    generate_simulated_dataset,
)
from qualibration_libs.parameters.experiment import get_qubits
from calibration_utils.measurement_utils import (
    declare_streams,
    save_measurement,
    buffer_streams,
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
    name="09a_power_rabi",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)

# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1"]
    node.parameters.use_simulated_data = True
    pass


# Instantiate the QUAM class from the default state file
node.machine = Quam.load()

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    node.namespace["qubits"] = qubits = get_qubits(node)

    n_avg = node.parameters.num_shots  # The number of averages
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(
            amps, attrs={"long_name": "pulse amplitude prefactor"}
        ),
    }
    operation = node.parameters.operation


    with program() as node.namespace["qua_program"]:
        # Declare QUA variables
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        n = declare(int)

        p2, p1, parity_streams = declare_streams(node, qubits, stream_fn=declare_output_stream)

        n_st = declare_output_stream()

        # Main experiment loop
        for qubit in qubits:
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(a, amps)):
                    if node.parameters.parity_measurement:
                        qubit.empty()
                        a1 = qubit.measure()

                    qubit.initialize(
                        target_state=node.parameters.target_state,
                        max_loops=node.parameters.max_loops,
                        conditional_drive=True,
                    )

                    align()
                    qubit.apply(operation, amplitude_scale = a)
                    # qubit.x180(amplitude_scale=a)
                    align()

                    a2 = qubit.measure()
                    
                    qubit.voltage_sequence.ramp_to_zero()

                    # Important align() - otherwise next qubit pulse seems to play way too early
                    align()

                    assign(p2, Cast.to_int(a2))

                    if node.parameters.parity_measurement:
                        assign(p1, Cast.to_int(a1))

                    save_measurement(node, qubit.name, p1, p2, parity_streams)

        # Stream processing
        with stream_processing():
            n_st.save("n")

            n_amps = len(amps)

            for qubit in qubits:
                buffer_streams(node, qubit.name, parity_streams, n_amps)


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.simulate
    or node.parameters.use_simulated_data
)
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


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated power-Rabi data so the full analysis pipeline can run without hardware."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process joint-outcome streams, fit power-Rabi data, and store results."""
    ds_processed = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qname: ("successful" if r["success"] else "failed")
        for qname, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot processed data and fit overlays; store figures in ``node.results["figures"]``."""
    node.results["figures"] = plot_all(
        node.results["ds_fit"],
        node.namespace["qubits"],
        node.results.get("fit_results", {}),
        analysis_signal=node.parameters.analysis_signal,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            opt_prefactor = node.results["fit_results"][q.name]["opt_amp"]
            getattr(q, node.parameters.operation).update(amplitude_scale=opt_prefactor)
            if node.parameters.operation == "x180":
                getattr(q, "x90").update(amplitude_scale=opt_prefactor / 2)

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

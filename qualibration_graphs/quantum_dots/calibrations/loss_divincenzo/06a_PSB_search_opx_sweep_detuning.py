# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from quam_builder.architecture.quantum_dots.operations.names import VoltagePointName
from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.psb_search_sweep_detuning import Parameters
from calibration_utils.common_utils.experiment import (
    get_sensors,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.iq_sweep import (
    fit_raw_data,
    log_fitted_results,
    plot_fidelity_vs_sweep,
    plot_visibility_vs_sweep,
    plot_sweep_summary,
)

from qualibrate.models.outcome import Outcome


# %% {Node initialisation}
description = """
        PAULI SPIN BLOCKADE SEARCH - Sweep Detuning
The goal of this sequence is to find the Pauli Spin Blockade (PSB) region.
To do so, the following triangle in voltage space (empty - random initialization - measurement) is applied using OPX
channels on the fast lines of the bias-tees while sweeping the "measure" voltage point along the detuning axis.

The OPX measures the response via RF reflectometry or DC current sensing during the readout window
(last segment of the triangle). A single-point averaging is performed and the data is extracted while
the program is running to display the results.

Depending on the cut-off frequency of the bias-tee, it may be necessary to adjust the barycenter (voltage offset) of each
triangle so that the fast line of the bias-tees sees zero voltage on average. Otherwise, the high-pass filtering effect
of the bias-tee will distort the fast pulses over time, unless a compensation pulse is played.

Prerequisites:
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the "empty" and "initialization" voltage points, and having defined the detuning axis.

State update:
    - The optimal detuning value for PSB readout, as the voltage point associated with the .measure macro.
"""


node = QualibrationNode[Parameters, Quam](
    name="06a_PSB_search_opx_sweep_detuning", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubit_pairs = ["q1_q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


def _resolve_qubit_pairs(node: QualibrationNode[Parameters, Quam]):
    """Resolve logical qubit pairs from parameters or default to all machine pairs."""
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]

    return list(node.machine.qubit_pairs.values())


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    qubit_pairs = _resolve_qubit_pairs(node)
    dot_pair_objects = [qp.quantum_dot_pair for qp in qubit_pairs]

    # Detuning axes may be added after the voltage sequence was first cached.
    # Refresh the sequence so keep-level tracking includes the detuning virtual axis.
    for gate_set_id in {dot_pair.voltage_sequence.gate_set.id for dot_pair in dot_pair_objects}:
        node.machine.reset_voltage_sequence(gate_set_id)
    for dot_pair in dot_pair_objects:
        if len(dot_pair.sensor_dots) != 1:
            raise ValueError(
                f"06a_PSB_search_opx_sweep_detuning expects exactly one sensor dot per pair; "
                f"{dot_pair.id!r} has {len(dot_pair.sensor_dots)}"
            )

    node.namespace["dot_pairs"] = dot_pair_objects
    node.namespace["qubit_pairs"] = qubit_pairs

    detuning_min = node.parameters.detuning_min
    detuning_max = node.parameters.detuning_max
    detuning_points = node.parameters.detuning_points
    detuning_step = (detuning_max - detuning_min) / detuning_points

    detuning_array = np.arange(detuning_min, detuning_max, detuning_step)

    # The swept axes. n_runs is the shot index (inner→outer: detuning, n_runs).
    # The qubit_pair dim is inferred from the per-pair stream names.
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "n_runs": xr.DataArray(np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}),
        "detuning": xr.DataArray(detuning_array, attrs={"long_name": "voltage", "units": "V"}),
    }
    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()

        detuning = declare(fixed)

        # Each logical qubit pair is read out through the single sensor dot of
        # its underlying quantum-dot pair.
        I_st = {qp.name: declare_stream() for qp in qubit_pairs}
        Q_st = {qp.name: declare_stream() for qp in qubit_pairs}
        I = {qp.name: declare(fixed) for qp in qubit_pairs}
        Q = {qp.name: declare(fixed) for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            # Perform them all sequentially for now. Can add footprint batching later
            for qubit_pair in qubit_pairs:
                dot_pair = qubit_pair.quantum_dot_pair
                with for_(*from_array(detuning, detuning_array)):
                    # ---------------------------------------------------------
                    # Step 1a: Empty - step to empty point (fixed duration)
                    # ---------------------------------------------------------
                    dot_pair.empty()
                    # Requires the dot pair object to have the empty macro, in addition to the qubits
                    # Equivalet step to the lvl_init
                    align()
                    # ---------------------------------------------------------
                    # Step 2: Initialize - load electron into dots (fixed duration)
                    # ---------------------------------------------------------
                    dot_pair.initialize()
                    # Requires the dot pair object to have the initialize macro, in addition to the qubits
                    align()
                    # ---------------------------------------------------------
                    # Step 3: Measure
                    # ---------------------------------------------------------
                    # No macro used here, since this node will characterise the macro

                    # First ramp to the fixed detuning point
                    dot_pair.ramp_to_detuning(
                        detuning,
                        ramp_duration=node.parameters.ramp_duration,
                        duration=node.parameters.buffer_duration,
                    )

                    align()

                    # And then explicitly measure using the pair's single sensor dot.
                    sensor = dot_pair.sensor_dots[0]
                    rr = sensor.readout_resonator
                    readout_length = rr.operations["readout"].length

                    dot_pair.step_to_voltages({}, duration=readout_length)
                    rr.measure("readout", qua_vars=(I[qubit_pair.name], Q[qubit_pair.name]))

                    save(I[qubit_pair.name], I_st[qubit_pair.name])
                    save(Q[qubit_pair.name], Q_st[qubit_pair.name])

                    align()
                    # Apply the compensation pulse via the voltage sequence
                    dot_pair.voltage_sequence.apply_compensation_pulse()

        with stream_processing():
            n_st.save("n")
            for qubit_pair in qubit_pairs:
                # Shot-by-shot: inner buffer = detuning, outer buffer = n_runs.
                I_st[qubit_pair.name].buffer(len(detuning_array)).buffer(node.parameters.num_shots).save(
                    f"I_{qubit_pair.name}"
                )
                Q_st[qubit_pair.name].buffer(len(detuning_array)).buffer(node.parameters.num_shots).save(
                    f"Q_{qubit_pair.name}"
                )


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
    # Reshape the per-pair streams into a qubit_pair-indexed ds with I/Q variables.
    pair_names = [pair.name for pair in node.namespace["qubit_pairs"]]
    I_arr = xr.concat([dataset[f"I_{p}"] for p in pair_names], dim="qubit_pair")
    Q_arr = xr.concat([dataset[f"Q_{p}"] for p in pair_names], dim="qubit_pair")
    I_arr = I_arr.assign_coords(qubit_pair=pair_names)
    Q_arr = Q_arr.assign_coords(qubit_pair=pair_names)
    node.results["ds_raw"] = xr.Dataset({"I": I_arr, "Q": Q_arr})


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = _resolve_qubit_pairs(node)
    # Get the active sensors from the loaded node parameters
    node.namespace["sensors"] = get_sensors(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""

    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: (Outcome.SUCCESSFUL if fit_result["success"] else Outcome.FAILED)
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot fidelity and visibility vs detuning for each qubit pair."""
    qubit_pairs = node.namespace["qubit_pairs"]
    sweep_name = node.parameters.sweep_name

    fig_fidelity = plot_fidelity_vs_sweep(
        node.results["ds_raw"], qubit_pairs, node.results["ds_fit"], sweep_name=sweep_name
    )
    fig_visibility = plot_visibility_vs_sweep(
        node.results["ds_raw"], qubit_pairs, node.results["ds_fit"], sweep_name=sweep_name
    )
    fig_summary = plot_sweep_summary(node.results["ds_raw"], qubit_pairs, node.results["ds_fit"], sweep_name=sweep_name)
    plt.show()
    node.results["figures"] = {
        "fidelity_vs_detuning": fig_fidelity,
        "visibility_vs_detuning": fig_visibility,
        "sweep_summary": fig_summary,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor data analysis was successful."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            fit_result = node.results["fit_results"][qp.name]
            if not fit_result["success"]:
                continue

            dot_pair = qp.quantum_dot_pair
            dot_pair.add_point(
                VoltagePointName.MEASURE,
                voltages={dot_pair.detuning_axis_name: float(fit_result["optimal_sweep_value"])},
                duration=node.parameters.buffer_duration,
                replace_existing_point=True,
            )

            # Current PSB readout assumes the first sensor dot defines the pair readout.
            sensor_dot = dot_pair.sensor_dots[0]

            operation = sensor_dot.readout_resonator.operations[node.parameters.operation]
            operation.integration_weights_angle -= float(fit_result["iw_angle"])

            # SensorDot.measure("readout") already returns IQ demodulated using
            # the operation's integration_weights_angle, so PSB state
            # discrimination should threshold the rotated I quadrature directly.
            pair_ids = {getattr(dot_pair, "id", None), getattr(dot_pair, "name", None)} - {None, ""}
            for pair_id in pair_ids:
                sensor_dot._add_readout_params(pair_id, threshold=float(fit_result["I_threshold"]))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

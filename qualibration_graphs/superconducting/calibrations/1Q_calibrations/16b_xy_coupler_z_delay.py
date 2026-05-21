"""XY-Coupler Z delay calibration for aligning microwave and flux pulses."""

# %% {Imports}
import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.xyc_delay import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
    baked_cplr_flux_xy_segments,
)
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        XY-Coupler Z DELAY CALIBRATION
This calibration determines the relative delay between the microwave XY control line (e.g. x180) and the coupler Z line (flux)
for each qubit pair. The goal is to ensure that the coupler flux pulse reaches the qubit at the same time as the XY drive and correct
for any latency differences between the two control lines. By applying the XY and coupler Z pulses simultaneously, the qubit
frequency is shifted during the XY rotation, which can lead to incomplete rotations if the pulses are not properly aligned
in time. By inserting variable leading / trailing zeros around the fixed XY and coupler Z pulse shapes, the sequence scans the
relative timing at 1ns resolution and measures the qubit state for two initial preparations
(|e> created by an initial x180 and |g> with identity). The resulting population (or I/Q) versus relative timing is
fitted to extract the optimal flux delay that best aligns the coupler Z pulse with the qubit XY drive.

Prerequisites:
    - Having calibrated a pi-pulse (x180) for the given qubit.
    - Having found the anticrossing point of the qubit frequency versus coupler flux bias.
    - (Optional) State discrimination calibrated if use_state_discrimination = True.

State update:
    - Adds extracted flux delay (fit_results[qubit]["flux_delay"]) to qp.coupler.opx_output.delay per successful qubit pair.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="16b_xy_coupler_z_delay",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for XY-Coupler Z delay calibration."""
    # Class containing tools to help handling units and conversions.
    u = unit(coerce_to_integer=True)
    machine = node.machine

    # Generate the OPX and Octave configurations
    config = node.machine.generate_config()

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

    # Check if couplers are attached
    for qp in qubit_pairs:
        assert qp.coupler is not None, f"Coupler not found for qubit pair {qp.name}. Cannot run without a coupler."

    n_avg = node.parameters.num_shots  # Number of averages (used in the QUA averaging loop)

    flux_waveform_list = {}  # Will store per-qubit flux pulse sample lists prior to baking
    delay_segments = {}  # Baked flux pulse segments with 1ns resolution

    # Build baked waveforms for each qubit pair
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubit = qp.qubit_control
        else:
            measured_qubit = qp.qubit_target

        flux_waveform_list[measured_qubit.xy.name] = [
            node.parameters.coupler_pulse_amplitude
        ] * measured_qubit.xy.operations["x180"].length

        delay_segments[measured_qubit.xy.name] = baked_cplr_flux_xy_segments(
            config,
            flux_waveform_list[measured_qubit.xy.name],
            measured_qubit,
            qp.coupler,
            node.parameters.zeros_before_after_pulse,
        )
        print(f"Baked waveform for {measured_qubit.xy.name}")

    node.namespace["config"] = config
    node.namespace["delay_segments"] = delay_segments
    relative_time = np.arange(
        -node.parameters.zeros_before_after_pulse, node.parameters.zeros_before_after_pulse, 1
    )  # x-axis for plotting - Must be in ns.
    node.namespace["relative_time"] = relative_time
    number_of_segments = 2 * node.parameters.zeros_before_after_pulse

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "init_state": xr.DataArray(["e", "g"], attrs={"long_name": "initial qubit state", "units": "a.u."}),
        "relative_time": xr.DataArray(
            relative_time, attrs={"long_name": "relative delay between pulses", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        if node.parameters.use_state_discrimination:
            state = [declare(bool) for _ in range(num_qubit_pairs)]
            state_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
        segment = declare(int)  # QUA variable for the flux pulse segment index

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU for all qubits in the batch
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            # Pre-compute which qubit to drive/measure for each pair in this batch
            measured_qubits_map = {
                ii: qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                for ii, qp in multiplexed_qubit_pairs.items()
            }

            # --- Averaging loop (outside the qubit-pair loop for multiplexed execution)
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)  # Save current average index for live progress

                # --- Initial state preparation loop: prepare |e> (via x180) and |g> (idle) for contrast
                for init_state in ["e", "g"]:
                    # --- Relative delay scan loop (index over baked XY+Z aligned segments)
                    with for_(segment, 0, segment < number_of_segments, segment + 1):

                        for ii, _ in multiplexed_qubit_pairs.items():
                            measured_qubit = measured_qubits_map[ii]
                            measured_qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()

                        # Phase 2: State prep + baked XY+Z segment for all pairs
                        for ii, _ in multiplexed_qubit_pairs.items():
                            measured_qubit = measured_qubits_map[ii]

                            # State preparation: excited (x180) or ground (wait same duration)
                            if init_state == "e":
                                measured_qubit.xy.play("x180")
                            elif init_state == "g":
                                measured_qubit.xy.wait(measured_qubit.xy.operations["x180"].length)

                            measured_qubit.align()
                            # Optional coarse pre-wait (accounts for leading padding before fine scan)
                            measured_qubit.wait(node.parameters.zeros_before_after_pulse // 4)
                            # Apply baked XY+Z segment with specific relative shift
                            with switch_(segment):
                                for j in range(0, number_of_segments):
                                    with case_(j):
                                        delay_segments[measured_qubit.xy.name][j].run()

                            measured_qubit.align()
                        align()

                        # Phase 3: Readout for all qubit pairs
                        for ii, _ in multiplexed_qubit_pairs.items():
                            measured_qubit = measured_qubits_map[ii]
                            measured_qubit.resonator.measure("readout", qua_vars=(I[ii], Q[ii]))
                            if node.parameters.use_state_discrimination:
                                assign(state[ii], I[ii] > measured_qubit.resonator.operations["readout"].threshold)
                                save(state[ii], state_st[ii])
                            else:
                                save(I[ii], I_st[ii])
                                save(Q[ii], Q_st[ii])
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_st[i].boolean_to_int().buffer(number_of_segments).buffer(2).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(number_of_segments).buffer(2).average().save(f"I{i + 1}")
                    Q_st[i].buffer(number_of_segments).buffer(2).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.namespace["config"]
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}
    plt.show()


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch raw data."""
    qmm = node.machine.connect()
    config = node.namespace["config"]
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    # Rename qubit_pair dimension to qubit for compatibility with analysis functions
    # Use unique pair names as coordinate values to avoid duplicates when multiple
    # pairs share the same measured qubit (e.g. q3 in q0-3, q3-4, q3-6).
    if "qubit_pair" in dataset.dims:
        qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
        measured_qubit_names = [q.name for q in node.namespace["measured_qubits"]]
        dataset = dataset.rename({"qubit_pair": "qubit"})
        dataset = dataset.assign_coords(qubit=qubit_pair_names)
        dataset = dataset.assign_coords(measured_qubit_name=("qubit", measured_qubit_names))
    # Store each coupler's set point as a per-pair coordinate so it survives save/load
    # and is unambiguous when pairs have different decouple offsets.
    qubit_pairs = node.namespace["qubit_pairs"]
    total_coupler_flux_mv = []
    for qp in qubit_pairs:
        if node.parameters.reset_coupler_bias:
            v = node.parameters.coupler_pulse_amplitude
        else:
            v = qp.coupler.decouple_offset + node.parameters.coupler_pulse_amplitude
        total_coupler_flux_mv.append(v * 1e3)
    dataset = dataset.assign_coords(total_coupler_flux_mv=("qubit", total_coupler_flux_mv))
    dataset.total_coupler_flux_mv.attrs = {"long_name": "Total coupler flux", "units": "mV"}
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
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

    # Rename qubit_pair dimension to qubit, using pair names as the coordinate
    # Use unique pair names as coordinate values (see execute_qua_program for rationale)
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
    """Analyse the raw data and store the fitted data in "ds_fit" and fit results."""
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
    """Plot the raw and fitted data."""
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
        "delay_scan": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    qubit_pairs = node.namespace["qubit_pairs"]
    measured_qubits = node.namespace["measured_qubits"]

    with node.record_state_updates():
        for qp, measured_qubit in zip(qubit_pairs, measured_qubits):
            if node.outcomes[qp.name] == "failed":
                continue

            res = node.results["fit_results"].get(qp.name)
            if res is None:
                continue

            # Update the coupler flux delay
            flux_delay = int(res.get("flux_delay", 0))
            qp.coupler.opx_output.delay += flux_delay
            print(f"Updated {qp.coupler.name} delay by {flux_delay} ns")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results and state updates."""
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()


# %%

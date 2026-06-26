# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool
from calibration_utils.common_utils.experiment import (
    get_qubit_pairs,
    enable_dual_drive_mw_pairs,
    progress_counter_with_log,
)

from qualibrate.core import QualibrationNode
from quam_config import Quam

from calibration_utils.common_utils.parity_streams import (
    declare_parity_streams,
    save_parity_measurement,
    buffer_parity_streams,
    process_parity_streams,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.cross_talk_rabi import (
    Parameters,
    CrossTalkCombo,
    fit_raw_data,
    log_fitted_results,
    build_crosstalk_matrix,
    plot_raw_data_with_fit,
    plot_crosstalk_matrix,
)
from qualibration_libs.runtime import simulate_and_plot

# %% {Description}
description = """
        CROSS-TALK RABI
This sequence quantifies the microwave cross-talk of each qubit's drive onto a
*different* readout pair. There are only two degrees of freedom:
    - the measured QUBIT PAIR, which initialises and reads out via PSB, and
    - the DRIVE QUBIT, a qubit outside that pair driven off-resonance.

For every measured qubit pair, each qubit that is NOT part of the pair is driven
in turn while sweeping its XY drive amplitude. If the drive bleeds onto the pair,
the conditional expectation traces out a (typically low-contrast) Rabi-like curve.

Example: with `qubit_pairs=["q2_q3"]`, the node sequentially drives q1 (reading
out q2-q3) and then q4 (reading out q2-q3). Passing several pairs builds the full
measured-pair x drive-qubit cross-talk matrix. By construction the driven qubit
is never part of the measured pair, so this never collapses to a standard Rabi.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the qubit frequencies.
    - Having set the qubit gate durations.

State update:
    - None (cross-talk is reported as a figure of merit / matrix only).
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="09c_cross_talk_rabi",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)

# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubit_pairs = ["q2_q3"]
    node.parameters.parity_pre_measurement = True
    node.parameters.simulate = True
    pass


# Instantiate the QUAM class from the default state file
node.machine = Quam.load()


# %% {Helpers}
def _pair_dot_ids(qubit_pair) -> set:
    """Return the set of quantum-dot ids that make up a qubit pair."""
    dot_ids = set()
    for member in (qubit_pair.qubit_control, qubit_pair.qubit_target):
        qd = member.quantum_dot
        dot_ids.add(qd if isinstance(qd, str) else qd.id)
    return dot_ids


def _build_combos(node: QualibrationNode[Parameters, Quam], qubit_pairs):
    """Build the (measured pair, external drive qubit) combinations to sweep."""
    combos = []
    all_qubits = list(node.machine.qubits.values())

    for qubit_pair in qubit_pairs:
        pair_dots = _pair_dot_ids(qubit_pair)

        drive_qubits = [
            q
            for q in all_qubits
            if (q.quantum_dot if isinstance(q.quantum_dot, str) else q.quantum_dot.id)
            not in pair_dots
        ]
        if not drive_qubits:
            node.log(
                f"Skipping pair '{qubit_pair.name}': no qubits outside the pair to drive.",
                level="warning",
            )
            continue

        for drive_qubit in drive_qubits:
            combos.append(
                CrossTalkCombo(
                    pair_name=qubit_pair.name,
                    drive_name=drive_qubit.name,
                    pair=qubit_pair,
                    drive_qubit=drive_qubit,
                )
            )

    if not combos:
        raise ValueError(
            "No (pair, drive) combinations to run. Check that the chosen qubit "
            "pairs have at least one qubit outside the pair."
        )
    return combos


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    node.namespace["combos"] = combos = _build_combos(node, qubit_pairs)
    node.log(
        "Cross-talk combinations: "
        + ", ".join(f"{c.drive_name}->{c.pair_name}" for c in combos)
    )

    n_avg = node.parameters.num_shots  # The number of averages
    # Drive-amplitude sweep (as a pre-factor of the drive-qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    node.namespace["amps"] = amps
    operation = node.parameters.operation

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw_pairs(node)

        # Declare QUA variables
        a = declare(fixed)  # QUA variable for the drive-qubit amplitude pre-factor
        n = declare(int)

        p2, p1, parity_streams = declare_parity_streams(
            node, combos, stream_fn=declare_output_stream
        )

        n_st = declare_output_stream()

        # Main experiment loop: one full sweep per (measured pair, drive qubit) combo
        for combo in combos:
            qubit_pair = combo.pair
            drive_qubit = combo.drive_qubit

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(a, amps)):
                    if node.parameters.parity_pre_measurement:
                        qubit_pair.empty()
                        a1 = qubit_pair.measure()

                    qubit_pair.initialize()

                    align()
                    # Drive a qubit OUTSIDE the measured pair to probe cross-talk.
                    drive_qubit.apply(operation, amplitude_scale=a)
                    align()

                    a2 = qubit_pair.measure()

                    qubit_pair.voltage_sequence.ramp_to_zero()

                    # Important align() - otherwise next qubit pulse seems to play way too early
                    align()

                    assign(p2, Cast.to_int(a2))

                    if node.parameters.parity_pre_measurement:
                        assign(p1, Cast.to_int(a1))

                    save_parity_measurement(node, combo.name, p1, p2, parity_streams)

        # Stream processing
        with stream_processing():
            n_st.save("n")

            n_amps = len(amps)
            for combo in combos:
                buffer_parity_streams(node, combo.name, parity_streams, n_amps)


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
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
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    combos = node.namespace["combos"]
    amps = node.namespace["amps"]
    parity_keys = (
        ["p0_p0", "p0_p1", "p1_p0", "p1_p1"]
        if node.parameters.parity_pre_measurement
        else ["p"]
    )

    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])

        # Live progress bar driven by the averaging counter
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter_with_log(
                n,
                node.parameters.num_shots,
                start_time=results.get_start_time(),
                node=node,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())

        # Assemble the dataset: each combo contributes its own 1-D amplitude sweep.
        handles = job.result_handles
        handles.wait_for_all_values()
        ds = xr.Dataset(
            coords={
                "amp_prefactor": (
                    "amp_prefactor",
                    amps,
                    {"long_name": "drive amplitude prefactor"},
                )
            }
        )
        for combo in combos:
            for key in parity_keys:
                var = f"{key}_{combo.name}"
                data = np.asarray(handles.get(var).fetch_all(), dtype=float)
                ds[var] = xr.DataArray(
                    data, dims=["amp_prefactor"], coords={"amp_prefactor": amps}
                )

    # Register the raw dataset
    node.results["ds_raw"] = ds


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome streams."""
    combos = node.namespace.get("combos")
    if combos is not None:
        item_names = [c.name for c in combos]
    else:
        # Load path: derive combo names directly from the dataset variables.
        from calibration_utils.cross_talk_rabi.analysis import _combo_names_from_ds

        item_names = _combo_names_from_ds(
            node.results["ds_raw"], node.parameters.analysis_signal
        )
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        item_names,
        parity_pre_measurement=node.parameters.parity_pre_measurement,
        item_dim="combo",
        sweep_dims=("amp_prefactor",),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    ds_fit, fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    node.results["crosstalk_matrix"] = build_crosstalk_matrix(fit_results)
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        cname: ("successful" if r["success"] else "failed")
        for cname, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data plus the cross-talk matrix."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.results.get("fit_results", {}),
        analysis_signal=node.parameters.analysis_signal,
    )
    fig_matrix = plot_crosstalk_matrix(node.results["crosstalk_matrix"])

    node.results["figure"] = fig
    node.results["figures"] = {
        "cross_talk_rabi": fig,
        "cross_talk_matrix": fig_matrix,
    }
    annotate_node_figures(node)
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Cross-talk is reported as a figure of merit only; no state update is performed."""
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

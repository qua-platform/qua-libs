# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_cafe import (
    NUM_STATES,
    Parameters,
    build_circuit_angles,
    fit_raw_data,
    log_fitted_results,
    plot_leakage,
    plot_raw_data_with_fit,
    primary_variant,
    process_raw_dataset,
    reference_gates_per_pair,
)
from calibration_utils.cz_cafe.qua_utils import (
    LAYER_STRIDE,
    QUBIT_STRIDE,
    play_compiled_layer,
    preparation_offset,
    undo_offset,
)
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Initialisation}
description = """
        CZ CONTEXT AWARE FIDELITY ESTIMATION (CAFE)
Measures the average gate fidelity of the CZ gate and splits its error into coherent and
incoherent parts, following D. M. Debroy et al., arXiv:2303.17565.

Protocol
--------
For each of the 16 states of a two-qubit SIC (a 2-design) and each repetition count n:
1. Prepare the state from |00> with one CZ and single-qubit rotations (Appendix A).
2. Apply the cycle n times. A cycle is the CZ macro; with the 'decaf' variant it is followed
   by X on both qubits, which echoes out single-qubit phase errors and low-frequency Z noise.
3. Undo the state the reference cycle would have produced, again with one CZ, and measure
   P(|00>).
Single-qubit rotations use only y90 pulses and virtual Z, which makes P(|00>) independent of
the sign conventions of both.

Analysis
--------
P(|00>) averaged over the 16 states is the average gate fidelity of n cycles. Coherent errors
grow as n^2 and incoherent errors as n, so fitting Eq. (B9) of the paper versus n separates
them. The node reports, per pair and variant:
- F: cycle fidelity against the reference unitary, corrected for SPAM.
- incoherent error: the infidelity left when coherent errors are removed.
- coherent error: the infidelity left when incoherent errors are removed.
A quadratic fit over the shallow depths (Eq. 8) is logged as a cross-check.

With reference_unitary='characterized', the undo step uses an fSim unitary built from the
per-pair angles [Δθ, Δγ, Δφ]; the coherent error then measures how well that model predicts
the gate.

Prerequisites:
- Calibrated single-qubit gates (x180, y90) on both qubits.
- Calibrated CZ gate (nodes 33a/33b, 34a/34b).
- Calibrated readout with state discrimination (GEF readout if record_leakage=True).

State update:
- qp.macros[operation].fidelity["CAFE"] (and ["DECAF"] if measured) =
  {"fidelity", "incoherent_error", "coherent_error", "spam"}. With a characterized reference
  the keys are "CAFE_characterized_reference" / "DECAF_characterized_reference".
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="37c_cz_cafe",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/cz_cafe/parameters.py
    machine=Quam.load(),  # Instantiate the QUAM class from the state file
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes."""
    # node.parameters.qubit_pairs = ["q1-q2"]
    # node.parameters.variants = ["cafe", "decaf"]
    pass


def _depths(parameters: Parameters) -> list:
    return list(range(0, parameters.max_depth + 1, parameters.depth_step))


def _variants(parameters: Parameters) -> list:
    return list(dict.fromkeys(parameters.variants))


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    if not node.parameters.use_state_discrimination:
        raise ValueError("CAFE needs the joint probability of |00>; set use_state_discrimination=True.")
    if not node.parameters.variants:
        raise ValueError("At least one variant ('cafe' or 'decaf') must be selected.")

    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)
    operation = node.parameters.operation
    for qp in qubit_pairs:
        if operation not in qp.macros:
            raise ValueError(f"Qubit pair {qp.name!r} has no macro {operation!r}. Available: {sorted(qp.macros)}")

    n_shots = node.parameters.num_shots
    depths = _depths(node.parameters)
    variants = _variants(node.parameters)
    record_leakage = node.parameters.record_leakage

    # Precompute the single-qubit layer angles of every circuit, per pair (the undo step
    # depends on the pair's reference unitary)
    reference_gates = reference_gates_per_pair(
        qubit_pairs.get_names(), node.parameters.reference_unitary, node.parameters.characterized_angles
    )
    node.namespace["reference_gates"] = reference_gates
    circuit_angles = {name: build_circuit_angles(variants, depths, gate) for name, gate in reference_gates.items()}
    preparation_angles = next(iter(circuit_angles.values())).flat_preparation_2pi()

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "variant": xr.DataArray(variants, attrs={"long_name": "cycle variant"}),
        "depth": xr.DataArray(depths, attrs={"long_name": "cycle repetitions"}),
        "state": xr.DataArray(np.arange(NUM_STATES), attrs={"long_name": "SIC state index"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_output_stream()
        state_idx = declare(int)
        depth_idx = declare(int)
        depth = declare(int)
        depths_qua = declare(int, value=depths)
        prep_qua = declare(fixed, value=preparation_angles)
        undo_qua = [declare(fixed, value=circuit_angles[qp.name].flat_undo_2pi()) for qp in qubit_pairs]
        # Per-pair variables so that multiplexed pairs run in parallel
        count = [declare(int) for _ in range(num_qubit_pairs)]
        offset = [declare(int) for _ in range(num_qubit_pairs)]
        angle_c = [declare(fixed) for _ in range(num_qubit_pairs)]
        angle_t = [declare(fixed) for _ in range(num_qubit_pairs)]
        state_c = [declare(int) for _ in range(num_qubit_pairs)]
        state_t = [declare(int) for _ in range(num_qubit_pairs)]
        flag = [declare(bool) for _ in range(num_qubit_pairs)]
        result = [declare(int) for _ in range(num_qubit_pairs)]
        p_return_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
        if record_leakage:
            f_control_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
            f_target_st = [declare_output_stream() for _ in range(num_qubit_pairs)]

        def play_layer(ii, qp, angles, start):
            """Play one compiled two-qubit layer whose angles start at ``start``."""
            play_compiled_layer(qp.qubit_control, angles, start, angle_c[ii])
            play_compiled_layer(qp.qubit_target, angles, start + QUBIT_STRIDE, angle_t[ii])
            qp.align()

        def save_flag(ii, condition, stream):
            assign(flag[ii], condition)
            assign(result[ii], Cast.to_int(flag[ii]))
            save(result[ii], stream)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_shots, n + 1):
                save(n, n_st)
                for iv, variant in enumerate(variants):
                    with for_(depth_idx, 0, depth_idx < len(depths), depth_idx + 1):
                        assign(depth, depths_qua[depth_idx])
                        with for_(state_idx, 0, state_idx < NUM_STATES, state_idx + 1):
                            # Reset the qubits and their frames
                            for qp in multiplexed_qubit_pairs.values():
                                qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                                qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()

                            for ii, qp in multiplexed_qubit_pairs.items():
                                # reset_frame(qp.qubit_control.xy.name)
                                # reset_frame(qp.qubit_target.xy.name)
                                # 1. Prepare the SIC state with one CZ
                                assign(offset[ii], preparation_offset(state_idx))
                                play_layer(ii, qp, prep_qua, offset[ii])
                                qp.macros[operation].apply()
                                # qp.align()
                                play_layer(ii, qp, prep_qua, offset[ii] + LAYER_STRIDE)
                                # 2. Repeat the cycle
                                with for_(count[ii], 0, count[ii] < depth, count[ii] + 1):
                                    qp.macros[operation].apply()
                                    if variant == "decaf":
                                        # qp.align()
                                        qp.qubit_control.xy.play("x180")
                                        qp.qubit_target.xy.play("x180")
                                    # qp.align()
                                # 3. Undo the state expected from the reference cycle
                                assign(offset[ii], undo_offset(iv, depth_idx, state_idx, len(depths)))
                                play_layer(ii, qp, undo_qua[ii], offset[ii])
                                qp.macros[operation].apply()
                                # qp.align()
                                play_layer(ii, qp, undo_qua[ii], offset[ii] + LAYER_STRIDE)
                            qp.align()

                            # Measure both qubits and record whether they returned to |00>
                            for ii, qp in multiplexed_qubit_pairs.items():
                                if record_leakage:
                                    qp.qubit_control.readout_state_gef(state_c[ii])
                                    qp.qubit_target.readout_state_gef(state_t[ii])
                                else:
                                    qp.qubit_control.readout_state(state_c[ii])
                                    qp.qubit_target.readout_state(state_t[ii])
                                save_flag(ii, (state_c[ii] == 0) & (state_t[ii] == 0), p_return_st[ii])
                                if record_leakage:
                                    save_flag(ii, state_c[ii] == 2, f_control_st[ii])
                                    save_flag(ii, state_t[ii] == 2, f_target_st[ii])
            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                p_return_st[i].buffer(NUM_STATES).buffer(len(depths)).buffer(len(variants)).average().save(
                    f"p_return{i + 1}"
                )
                if record_leakage:
                    f_control_st[i].buffer(NUM_STATES).buffer(len(depths)).buffer(len(variants)).average().save(
                        f"f_control{i + 1}"
                    )
                    f_target_st[i].buffer(NUM_STATES).buffer(len(depths)).buffer(len(variants)).average().save(
                        f"f_target{i + 1}"
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
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset."""
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


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubit pairs from the loaded node parameters
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    node.namespace["reference_gates"] = reference_gates_per_pair(
        qubit_pairs.get_names(), node.parameters.reference_unitary, node.parameters.characterized_angles
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse raw data, fit, log results, set outcomes and store structured fit results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {
        qp_name: {variant: asdict(fr) for variant, fr in per_variant.items()}
        for qp_name, per_variant in fit_results.items()
    }
    log_fitted_results(fit_results, log_callable=node.log)
    # The node outcome follows the CAFE variant when it was measured
    main_variant = primary_variant(node.results["ds_fit"].variant.values.tolist())
    node.outcomes = {
        qp_name: ("successful" if per_variant[main_variant].success else "failed")
        for qp_name, per_variant in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit pair grid locations."""
    qubit_pairs = node.namespace["qubit_pairs"]
    ds_fit = node.results["ds_fit"]

    fig = plot_raw_data_with_fit(ds_fit, qubit_pairs)
    plt.show()
    node.results["figure"] = fig

    if "f_control" in ds_fit.data_vars:
        fig_leakage = plot_leakage(ds_fit, qubit_pairs)
        plt.show()
        node.results["leakage_figure"] = fig_leakage


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Store the CAFE error budget of every successful fit in the CZ macro's fidelity dictionary."""
    operation = node.parameters.operation
    suffix = "" if node.parameters.reference_unitary == "ideal" else "_characterized_reference"
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                node.log(f"Skipping state update for {qp.name}: fit flagged unsuccessful.")
                continue
            for variant, fr in node.results["fit_results"][qp.name].items():
                if not fr["success"]:
                    continue
                qp.macros[operation].fidelity[f"{variant.upper()}{suffix}"] = {
                    "fidelity": fr["fidelity"],
                    "incoherent_error": fr["incoherent_error"],
                    "coherent_error": fr["coherent_error"],
                    "spam": fr["spam"],
                }


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome
from quam_config import Quam
from quam_builder.architecture.quantum_dots.components.readout_resonator import (
    ReadoutResonatorIQ,
    ReadoutResonatorSingle,
)
from calibration_utils.psb_search_sweep_measure_duration import (
    Parameters,
    build_psb_readout_sweep,
    fit_measure_duration_raw_data,
    generate_simulated_dataset,
    log_fitted_results,
    plot_measure_duration_sweep_figures,
    plot_simulated_dataset_histograms,
)
from calibration_utils.psb_search_fixed_detuning import plot_rotated_iq_density_at_optimum
from calibration_utils.common_utils.experiment import get_sensors
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Node initialisation}
description = """
        PAULI SPIN BLOCKADE SEARCH - Fixed Detuning, Sweep Readout Length
The goal of this sequence is to probe PSB contrast while sweeping how long the resonator
is integrated (readout pulse length / accumulated demod segments) at a fixed measure-point
detuning (optionally overridden via node parameters).

The readout pulse length is set to an exact integer number of demod chunks
(``N * 4 * segment_length`` ns) so QM integration weights match the accumulated measure.

Prerequisites:
    - Initialized Quam, calibrated sensor resonators, empty/init/measure macros.
    - Prefer having run 06a/06b to set the measure detuning; optional ``detuning`` override.

State update:
    Reverts temporary detuning override and extended readout pulse, then (if the fit succeeded)
    persists the optimal readout ``length``, integration-weights angle, and discrimination threshold
    on the pair's sensor dot (same pattern as 05c length + 06a readout calibration).
"""


node = QualibrationNode[Parameters, Quam](
    name="06b_PSB_search_opx_fixed_detuning_measure_duration",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.num_shots = 10
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.simulate = False
    node.parameters.simulation_duration_ns = 60_000
    node.parameters.use_simulated_data = True


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


def _resolve_qubit_pairs(node: QualibrationNode[Parameters, Quam]):
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create sweep axes and QUA program: PSB triangle + accumulated readout vs integration time."""
    qubit_pairs = _resolve_qubit_pairs(node)
    dot_pair_objects = [qp.quantum_dot_pair for qp in qubit_pairs]

    for gate_set_id in {dot_pair.voltage_sequence.gate_set.id for dot_pair in dot_pair_objects}:
        node.machine.reset_voltage_sequence(gate_set_id)
    for dot_pair in dot_pair_objects:
        if len(dot_pair.sensor_dots) != 1:
            raise ValueError(
                f"06e expects exactly one sensor dot per pair; {dot_pair.id!r} has {len(dot_pair.sensor_dots)}"
            )

    node.namespace["dot_pairs"] = dot_pair_objects
    node.namespace["qubit_pairs"] = qubit_pairs

    node.namespace["tracked_original_detunings"] = {}
    for dot_pair in dot_pair_objects:
        if node.parameters.detuning is not None:
            dot_pair_gate_set = dot_pair.voltage_sequence.gate_set
            point_name = dot_pair._create_point_name("measure")
            point = dot_pair_gate_set.get_macros()[point_name]
            node.namespace["tracked_original_detunings"][dot_pair.name] = point.voltages.get(
                dot_pair.name
            )
            point.voltages[dot_pair.name] = node.parameters.detuning

    readout_max = node.parameters.readout_length_max
    if readout_max is None:
        readout_max = dot_pair_objects[0].sensor_dots[0].readout_resonator.operations["readout"].length

    sweep = build_psb_readout_sweep(
        node.parameters.readout_length_min,
        readout_max,
        node.parameters.readout_length_points,
    )
    array_size = sweep["array_size"]
    segment_length = sweep["segment_length"]
    pulse_length = sweep["pulse_length"]
    sweep_coord = sweep["sweep_coord"]
    sweep_name = node.parameters.sweep_name
    node.namespace["readout_sweep"] = sweep

    num_segments = sweep["num_segments"]
    if array_size > num_segments:
        raise ValueError(
            f"Sweep has {array_size} save points but pulse allows {num_segments} segments "
            f"(pulse_length={pulse_length})."
        )

    kinds = {type(qp.quantum_dot_pair.sensor_dots[0].readout_resonator) for qp in qubit_pairs}
    if len(kinds) != 1:
        raise TypeError(f"06e expects all qubit pairs to use the same readout resonator class; got {kinds}.")
    (readout_cls,) = tuple(kinds)
    if readout_cls not in (ReadoutResonatorSingle, ReadoutResonatorIQ):
        raise TypeError(f"06e supports ReadoutResonatorSingle and ReadoutResonatorIQ; got {readout_cls}.")

    node.namespace["tracked_resonators"] = []
    seen = set()
    for qubit_pair in qubit_pairs:
        rr = qubit_pair.quantum_dot_pair.sensor_dots[0].readout_resonator
        op_name = "readout" + f"_{qubit_pair.quantum_dot_pair.name}"
        rid = id(rr)
        if rid in seen:
            continue
        seen.add(rid)
        with tracked_updates(rr, auto_revert=False, dont_assign_to_none=True) as resonator:
            resonator.operations[op_name].length = pulse_length
            node.namespace["tracked_resonators"].append(resonator)

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([qp.name for qp in qubit_pairs]),
        "n_runs": xr.DataArray(np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}),
        sweep_name: xr.DataArray(sweep_coord, attrs={"long_name": "readout length", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()
        idx = declare(int)

        I_st = {qp.name: declare_stream() for qp in qubit_pairs}
        Q_st = {qp.name: declare_stream() for qp in qubit_pairs}
        tmp_i = declare(fixed)
        tmp_q = declare(fixed)

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                dot_pair = qubit_pair.quantum_dot_pair
                readout_pulse_name = "readout" + f"_{dot_pair.name}"
                dot_pair.macros[node.parameters.initialization_macro].apply(max_loops = 2)

                dot_pair.ramp_to_point(
                    "measure",
                    ramp_duration=node.parameters.ramp_duration,
                    duration=node.parameters.buffer_duration,
                )

                sensor = dot_pair.sensor_dots[0]
                rr = sensor.readout_resonator
                readout_len_qua = rr.operations[readout_pulse_name].length
                dot_pair.voltage_sequence.track_sticky_duration(readout_len_qua)
                align(rr.id, dot_pair.physical_channel.id)

                if readout_cls is ReadoutResonatorSingle:
                    I_acc, Q_acc = rr.measure_accumulated(readout_pulse_name, segment_length=segment_length)
                else:
                    II_a, IQ_a, QI_a, QQ_a = rr.measure_accumulated(readout_pulse_name, segment_length=segment_length)

                align(rr.id, dot_pair.physical_channel.id)
                dot_pair.voltage_sequence.apply_compensation_pulse(go_to_zero = True, return_to_zero = True)
                dot_pair.voltage_sequence.ramp_to_zero()

                if readout_cls is ReadoutResonatorSingle:
                    with for_(idx, 0, idx < array_size, idx + 1):
                        save(I_acc[idx], I_st[qubit_pair.name])
                        save(Q_acc[idx], Q_st[qubit_pair.name])
                else:
                    with for_(idx, 0, idx < array_size, idx + 1):
                        assign(tmp_i, II_a[idx] - QQ_a[idx])
                        assign(tmp_q, IQ_a[idx] + QI_a[idx])
                        save(tmp_i, I_st[qubit_pair.name])
                        save(tmp_q, Q_st[qubit_pair.name])

        with stream_processing():
            n_st.save("n")
            for qp in qubit_pairs:
                I_st[qp.name].buffer(array_size).buffer(node.parameters.num_shots).save(f"I_{qp.name}")
                Q_st[qp.name].buffer(array_size).buffer(node.parameters.num_shots).save(f"Q_{qp.name}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect(timeout = 600)
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated PSB readout-length dataset generated successfully.")


@node.run_action(skip_if=not node.parameters.use_simulated_data)
def plot_simulated_data(node: QualibrationNode[Parameters, Quam]):
    fig = plot_simulated_dataset_histograms(
        node.results["ds_raw"],
        sweep_name=node.parameters.sweep_name,
    )
    plt.show()


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.use_simulated_data
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute QUA and reshape per-pair streams into ``ds_raw``."""
    qmm = node.machine.connect(timeout=node.parameters.timeout)
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        job.wait_until("Done")
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
            )
        node.log(job.execution_report())

    pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
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
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = _resolve_qubit_pairs(node)
    node.namespace["dot_pairs"] = [qp.quantum_dot_pair for qp in node.namespace["qubit_pairs"]]
    node.namespace["sensors"] = get_sensors(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit PCA + two-Gaussian readout model at each readout length (same stack as 06a)."""
    node.results["ds_fit"], fit_results = fit_measure_duration_raw_data(node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: (Outcome.SUCCESSFUL if fit_result["success"] else Outcome.FAILED)
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot fidelity, visibility, sweep summary, and histograms vs readout length."""
    sweep_name = node.parameters.sweep_name
    figs = plot_measure_duration_sweep_figures(node, sweep_name=sweep_name)
    fig_iq = plot_rotated_iq_density_at_optimum(
        node.results["ds_raw"],
        node.results["fit_results"],
        node.namespace["qubit_pairs"],
    )
    plt.show()
    node.results["figures"] = {
        "fidelity_vs_readout_length": figs["fidelity_vs_sweep"],
        "visibility_vs_readout_length": figs["visibility_vs_sweep"],
        "sweep_summary": figs["sweep_summary"],
        "histograms_vs_readout_length": figs["histograms_vs_sweep"],
        "rotated_iq_density": fig_iq,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Revert temporary patches, then persist optimal readout length and readout calibration."""
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    for dot_pair in node.namespace["dot_pairs"]:
        if dot_pair.name in node.namespace.get("tracked_original_detunings", {}):
            dot_pair_gate_set = dot_pair.voltage_sequence.gate_set
            point_name = dot_pair._create_point_name("measure")
            point = dot_pair_gate_set.get_macros()[point_name]
            point.voltages[dot_pair.name] = node.namespace["tracked_original_detunings"][dot_pair.name]

    fit_results = node.results.get("fit_results")
    if not fit_results:
        return

    with node.record_state_updates():
        op_name = node.parameters.operation
        for qp in node.namespace["qubit_pairs"]:
            fit_result = fit_results[qp.name]
            if not fit_result["success"]:
                continue

            dot_pair = qp.quantum_dot_pair
            op_name = "readout" + f"_{dot_pair.name}"
            sensor_dot = dot_pair.sensor_dots[0]
            operation = sensor_dot.readout_resonator.operations[op_name]

            optimal_ns = int(round(float(fit_result["optimal_sweep_value"])))
            operation.length = optimal_ns 

            operation.integration_weights_angle -= float(fit_result["iw_angle"])
            print(f"For sensor {sensor_dot.name}, pair {dot_pair.name}, threshold calculated to be {fit_result["I_threshold"]} and angle {float(fit_result["iw_angle"])}")
            sensor_dot._add_readout_params(dot_pair.name, threshold=float(fit_result["I_threshold"]))
            sensor_dot.readout_thresholds[dot_pair.name] = float(fit_result["I_threshold"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

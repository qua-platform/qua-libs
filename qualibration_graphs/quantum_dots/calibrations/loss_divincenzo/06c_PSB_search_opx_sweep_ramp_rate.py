# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log
from qualang_tools.loops import from_array

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome
from quam_config import Quam
from calibration_utils.psb_search_sweep_ramp_rate import (
    Parameters,
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


# %% {Node initialisation}
description = """
        PAULI SPIN BLOCKADE SEARCH - Fixed Measure Point, Sweep Ramp to Measure
The goal of this sequence is to probe PSB contrast while sweeping how long the voltage ramp
to the PSB measurement point lasts (nanoseconds). For a fixed voltage trajectory, shorter ramps
correspond to higher effective ramp rates on the OPX fast lines.

The sequence matches 06e except the swept axis is ramp duration: preparation via
``initialization_macro`` (default ``empty``), then for each ramp duration
``ramp_to_point('measure', ...)``, then resonator readout at the fixed measure
voltages (optional ``detuning`` override like 06e).

Prerequisites:
    - Initialized Quam, calibrated sensor resonators, empty/init/measure macros.
    - Prefer having run 06a/06b to set the measure detuning; optional ``detuning`` override.

State update:
    Reverts temporary detuning override, then (if the fit succeeded) persists the optimal ramp
    duration on the pair ``measure`` macro when supported, and updates integration-weights angle
    and discrimination threshold on the sensor dot (readout pulse length is not changed).
"""


node = QualibrationNode[Parameters, Quam](
    name="06c_PSB_search_opx_sweep_ramp_rate",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.simulate = False
    node.parameters.simulation_duration_ns = 60_000
    node.parameters.use_simulated_data = False


node.machine = Quam.load()


def _resolve_qubit_pairs(node: QualibrationNode[Parameters, Quam]):
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create sweep axes and QUA program: PSB path with inner sweep over ramp duration to measure."""
    qubit_pairs = _resolve_qubit_pairs(node)
    dot_pair_objects = [qp.quantum_dot_pair for qp in qubit_pairs]

    for gate_set_id in {dot_pair.voltage_sequence.gate_set.id for dot_pair in dot_pair_objects}:
        node.machine.reset_voltage_sequence(gate_set_id)
    for dot_pair in dot_pair_objects:
        if len(dot_pair.sensor_dots) != 1:
            raise ValueError(
                f"06f expects exactly one sensor dot per pair; {dot_pair.id!r} has {len(dot_pair.sensor_dots)}"
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

    ramp_min = int(node.parameters.ramp_duration_min)
    ramp_max = int(node.parameters.ramp_duration_max)
    ramp_step = int(node.parameters.ramp_duration_step)
    if ramp_min % 4 != 0 or ramp_max % 4 != 0 or ramp_step % 4 != 0:
        raise ValueError(
            "Ramp settings must be divisible by 4. Received "
            f"ramp_duration_min={ramp_min}, ramp_duration_max={ramp_max}, ramp_duration_step={ramp_step}"
        )

    ramp_duration_array = np.arange(ramp_min, ramp_max, ramp_step, dtype=int)
    if len(ramp_duration_array) == 0:
        raise ValueError("Empty ramp duration sweep: require ramp_duration_min < ramp_duration_max with positive step.")

    sweep_name = node.parameters.sweep_name
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([qp.name for qp in qubit_pairs]),
        "n_runs": xr.DataArray(np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}),
        sweep_name: xr.DataArray(
            ramp_duration_array.astype(float),
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()
        ramp_d = declare(int)

        I_st = {qp.name: declare_stream() for qp in qubit_pairs}
        Q_st = {qp.name: declare_stream() for qp in qubit_pairs}
        I = {qp.name: declare(fixed) for qp in qubit_pairs}
        Q = {qp.name: declare(fixed) for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)
            with for_(*from_array(ramp_d, ramp_duration_array)):
                for qubit_pair in qubit_pairs:
                    dot_pair = qubit_pair.quantum_dot_pair
                    dot_pair.voltage_sequence.step_to_voltages(voltages = {}, duration = node.parameters.reset_wait_time)
                    align()

                    dot_pair.macros[node.parameters.initialization_macro].apply()
                    dot_pair.ramp_to_point(
                        "measure",
                        ramp_duration=ramp_d,
                        duration=node.parameters.buffer_duration,
                    )

                    sensor = dot_pair.sensor_dots[0]
                    rr = sensor.readout_resonator
                    readout_length = rr.operations[f"readout_{dot_pair.name}"].length
                    dot_pair.voltage_sequence.track_sticky_duration(readout_length)
                    align(rr.id, dot_pair.physical_channel.id)

                    rr.measure(f"readout_{dot_pair.name}", qua_vars=(I[qubit_pair.name], Q[qubit_pair.name]))

                    save(I[qubit_pair.name], I_st[qubit_pair.name])
                    save(Q[qubit_pair.name], Q_st[qubit_pair.name])
                    align(rr.id, dot_pair.physical_channel.id)

                    dot_pair.voltage_sequence.apply_compensation_pulse(go_to_zero = True, return_to_zero = True)

                    dot_pair.voltage_sequence.ramp_to_zero()

                    align()                   

        with stream_processing():
            n_st.save("n")
            for qp in qubit_pairs:
                n_r = len(ramp_duration_array)
                I_st[qp.name].buffer(n_r).buffer(node.parameters.num_shots).save(f"I_{qp.name}")
                Q_st[qp.name].buffer(n_r).buffer(node.parameters.num_shots).save(f"Q_{qp.name}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
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
    node.log("[sim] Simulated PSB ramp-duration dataset generated successfully.")


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
    qmm = node.machine.connect()
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
    """Fit PCA + two-Gaussian readout model at each ramp duration (same stack as 06a/06e)."""
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
    """Plot fidelity, visibility, sweep summary, and histograms vs ramp duration."""
    sweep_name = node.parameters.sweep_name
    figs = plot_measure_duration_sweep_figures(node, sweep_name=sweep_name)
    fig_iq = plot_rotated_iq_density_at_optimum(
        node.results["ds_raw"],
        node.results["fit_results"],
        node.namespace["qubit_pairs"],
    )
    plt.show()
    node.results["figures"] = {
        "fidelity_vs_ramp_duration": figs["fidelity_vs_sweep"],
        "visibility_vs_ramp_duration": figs["visibility_vs_sweep"],
        "sweep_summary": figs["sweep_summary"],
        "histograms_vs_ramp_duration": figs["histograms_vs_sweep"],
        "rotated_iq_density": fig_iq,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Revert temporary detuning override, then persist optimal ramp and readout calibration."""
    # for dot_pair in node.namespace["dot_pairs"]:
    #     if dot_pair.name in node.namespace.get("tracked_original_detunings", {}):
    #         dot_pair_gate_set = dot_pair.voltage_sequence.gate_set
    #         point_name = dot_pair._create_point_name("measure")
    #         point = dot_pair_gate_set.get_macros()[point_name]
    #         point.voltages[dot_pair.name] = node.namespace["tracked_original_detunings"][dot_pair.name]

    fit_results = node.results.get("fit_results")
    if not fit_results:
        return

    with node.record_state_updates():
        op_name = node.parameters.operation
        for qp in node.namespace["qubit_pairs"]:
            fit_result = fit_results[qp.name]
            if not fit_result["success"]:
                continue

            # dot_pair = qp.quantum_dot_pair
            # sensor_dot = dot_pair.sensor_dots[0]
            # operation = sensor_dot.readout_resonator.operations[op_name]

            # optimal_ns = int(round(float(fit_result["optimal_sweep_value"])))

            # measure_macro = dot_pair.macros.get("measure")
            # if measure_macro is not None:
            #     try:
            #         measure_macro.update(ramp_duration=optimal_ns)
            #     except TypeError:
            #         node.log(
            #             f"Skipping measure macro ramp_duration update for {dot_pair.id!r}: "
            #             "macro.update does not accept ramp_duration."
            #         )

            # operation.integration_weights_angle -= float(fit_result["iw_angle"])

            # pair_ids = {
            #     getattr(dot_pair, "id", None),
            #     getattr(dot_pair, "name", None),
            # } - {None, ""}
            # for pair_id in pair_ids:
            #     sensor_dot._add_readout_params(pair_id, threshold=float(fit_result["I_threshold"]))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

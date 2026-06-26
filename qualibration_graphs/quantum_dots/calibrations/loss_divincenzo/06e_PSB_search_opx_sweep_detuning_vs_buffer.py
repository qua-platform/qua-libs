# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.loops import from_array

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome

from quam_config import Quam
from calibration_utils.common_utils.experiment import progress_counter_with_log, enable_dual_drive_mw_pairs
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot

from calibration_utils.psb_search_sweep_detuning_vs_buffer import (
    Parameters,
    analyse_detuning_vs_buffer,
    plot_detuning_vs_buffer_pca_map,
)


# %% {Node initialisation}
description = """
        PAULI SPIN BLOCKADE SEARCH - Sweep Detuning vs Buffer Duration
This node sweeps the measure-point detuning and pre-readout buffer duration to map
where PSB contrast is strongest.

Sequence:
    prepare (empty/initialize) -> ramp to (detuning, barrier) with variable buffer
    -> align -> readout

The analysis is intentionally slim for now: it computes PCA-derived spread metrics
from shot-by-shot I/Q data and plots a 2D heatmap vs (detuning, buffer duration).
"""


node = QualibrationNode[Parameters, Quam](
    name="06e_PSB_search_opx_sweep_detuning_vs_buffer",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.simulate = False
    node.parameters.detuning_points = 11
    node.parameters.num_shots = 100


node.machine = Quam.load()


def _resolve_qubit_pairs(node: QualibrationNode[Parameters, Quam]):
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


def _reshape_to_shot_detuning_buffer(
    arr: np.ndarray,
    *,
    n_shots: int,
    n_detuning: int,
    n_buffer: int,
) -> np.ndarray:
    """Normalize stream shape to (n_runs, detuning, buffer_duration)."""
    arr = np.asarray(arr)
    if arr.shape == (n_shots, n_detuning, n_buffer):
        return arr

    if arr.ndim == 3:
        expected = (n_shots, n_detuning, n_buffer)
        for axes in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ):
            candidate = np.transpose(arr, axes)
            if candidate.shape == expected:
                return candidate

    if arr.size == n_shots * n_detuning * n_buffer:
        return arr.reshape(n_shots, n_detuning, n_buffer)

    raise ValueError(
        f"Unexpected stream shape {arr.shape}; cannot coerce to "
        f"({n_shots}, {n_detuning}, {n_buffer})."
    )


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the 2D QUA program."""
    qubit_pairs = _resolve_qubit_pairs(node)
    dot_pairs = [qp.quantum_dot_pair for qp in qubit_pairs]

    for gate_set_id in {dot_pair.voltage_sequence.gate_set.id for dot_pair in dot_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)

    for dot_pair in dot_pairs:
        if len(dot_pair.sensor_dots) != 1:
            raise ValueError(
                "06e expects exactly one sensor dot per pair; "
                f"{dot_pair.id!r} has {len(dot_pair.sensor_dots)}."
            )

    buffer_min = int(node.parameters.buffer_duration_min)
    buffer_max = int(node.parameters.buffer_duration_max)
    buffer_step = int(node.parameters.buffer_duration_step)
    if buffer_min % 4 != 0 or buffer_max % 4 != 0 or buffer_step % 4 != 0:
        raise ValueError(
            "Buffer settings must be divisible by 4. Received "
            f"buffer_duration_min={buffer_min}, buffer_duration_max={buffer_max}, "
            f"buffer_duration_step={buffer_step}"
        )
    buffer_ns_array = np.arange(buffer_min, buffer_max, buffer_step, dtype=int)
    buffer_cc_array = (buffer_ns_array // 4).astype(int)
    if len(buffer_ns_array) == 0:
        raise ValueError("Empty buffer sweep: require min < max with a positive step.")

    detuning_array = np.linspace(
        node.parameters.detuning_min,
        node.parameters.detuning_max,
        int(node.parameters.detuning_points),
    )

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["dot_pairs"] = dot_pairs
    node.namespace["detuning_array"] = detuning_array
    node.namespace["buffer_ns_array"] = buffer_ns_array

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "n_runs": xr.DataArray(np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}),
        "detuning": xr.DataArray(detuning_array, attrs={"long_name": "detuning", "units": "V"}),
        "buffer_duration": xr.DataArray(
            buffer_ns_array, attrs={"long_name": "buffer duration", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw_pairs(node)

        n = declare(int)
        n_st = declare_output_stream()
        detuning = declare(fixed)
        buffer_cc = declare(int)

        i_stream = {qp.name: declare_output_stream() for qp in qubit_pairs}
        q_stream = {qp.name: declare_output_stream() for qp in qubit_pairs}
        i_var = {qp.name: declare(fixed) for qp in qubit_pairs}
        q_var = {qp.name: declare(fixed) for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)
            for qubit_pair in qubit_pairs:
                dot_pair = qubit_pair.quantum_dot_pair
                with for_(*from_array(detuning, detuning_array)):
                    with for_(*from_array(buffer_cc, buffer_cc_array)):
                        # if node.parameters.qubit_pair_to_initialize is not None:
                        #     init_pair = node.machine.qubit_pairs[node.parameters.qubit_pair_to_initialize]
                        #     init_pair.quantum_dot_pair.macros[node.parameters.initialization_macro].apply()
                        # else:
                        #     dot_pair.macros[node.parameters.initialization_macro].apply()

                        if node.parameters.qubit_to_pulse is not None:
                            node.machine.qubits[node.parameters.qubit_to_pulse].x180()

                        dot_pair.ramp_to_voltages(
                            {
                                dot_pair.name: detuning,
                                dot_pair.barrier_gate.name: node.parameters.barrier_gate_voltage,
                            },
                            ramp_duration=node.parameters.ramp_duration,
                            duration=buffer_cc * 4,
                        )

                        # Explicit align before sensor measurement at each buffer point.
                        sensor = dot_pair.sensor_dots[0]
                        rr = sensor.readout_resonator
                        op_name = f"readout_{dot_pair.name}"
                        readout_length = rr.operations[op_name].length
                        dot_pair.voltage_sequence.track_sticky_duration(readout_length)
                        align(rr.id, dot_pair.physical_channel.id)

                        rr.measure(op_name, qua_vars=(i_var[qubit_pair.name], q_var[qubit_pair.name]))
                        save(i_var[qubit_pair.name], i_stream[qubit_pair.name])
                        save(q_var[qubit_pair.name], q_stream[qubit_pair.name])

                        align(rr.id, dot_pair.physical_channel.id)
                        dot_pair.voltage_sequence.apply_compensation_pulse(
                            go_to_zero=True, return_to_zero=True
                        )
                        dot_pair.voltage_sequence.ramp_to_zero()
                        wait(1000000//4)
                        align()

        with stream_processing():
            n_st.save("n")
            for qp in qubit_pairs:
                i_stream[qp.name].buffer(len(buffer_cc_array)).buffer(len(detuning_array)).buffer(
                    node.parameters.num_shots
                ).save(f"I_{qp.name}")
                q_stream[qp.name].buffer(len(buffer_cc_array)).buffer(len(detuning_array)).buffer(
                    node.parameters.num_shots
                ).save(f"Q_{qp.name}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute QUA and reshape per-pair streams into ds_raw."""
    qmm = node.machine.connect(timeout=node.parameters.timeout)
    config = node.machine.generate_config()

    n_shots = int(node.parameters.num_shots)
    n_detuning = len(node.namespace["detuning_array"])
    n_buffer = len(node.namespace["buffer_ns_array"])

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = job.result_handles
        data_fetcher.wait_for_all_values()
        progress_counter_with_log(
            n_shots - 1,
            n_shots,
            node=node,
        )

        pair_names = [pair.name for pair in node.namespace["qubit_pairs"]]
        i_arrays = []
        q_arrays = []
        for pair_name in pair_names:
            i_raw = data_fetcher.get(f"I_{pair_name}").fetch_all()
            q_raw = data_fetcher.get(f"Q_{pair_name}").fetch_all()
            i_arrays.append(
                _reshape_to_shot_detuning_buffer(
                    i_raw,
                    n_shots=n_shots,
                    n_detuning=n_detuning,
                    n_buffer=n_buffer,
                )
            )
            q_arrays.append(
                _reshape_to_shot_detuning_buffer(
                    q_raw,
                    n_shots=n_shots,
                    n_detuning=n_detuning,
                    n_buffer=n_buffer,
                )
            )

        node.log(job.execution_report())

    i_data = np.stack(i_arrays, axis=0)
    q_data = np.stack(q_arrays, axis=0)
    pair_names = [pair.name for pair in node.namespace["qubit_pairs"]]

    node.results["ds_raw"] = xr.Dataset(
        data_vars={
            "I": xr.DataArray(
                i_data,
                dims=["qubit_pair", "n_runs", "detuning", "buffer_duration"],
                coords={
                    "qubit_pair": pair_names,
                    "n_runs": np.arange(n_shots),
                    "detuning": node.namespace["detuning_array"],
                    "buffer_duration": node.namespace["buffer_ns_array"],
                },
                attrs={"long_name": "in-phase quadrature", "units": "V"},
            ),
            "Q": xr.DataArray(
                q_data,
                dims=["qubit_pair", "n_runs", "detuning", "buffer_duration"],
                coords={
                    "qubit_pair": pair_names,
                    "n_runs": np.arange(n_shots),
                    "detuning": node.namespace["detuning_array"],
                    "buffer_duration": node.namespace["buffer_ns_array"],
                },
                attrs={"long_name": "quadrature-phase", "units": "V"},
            ),
        }
    )


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = _resolve_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Run slim 2D PCA-spread analysis for plotting."""
    ds_fit, fit_results = analyse_detuning_vs_buffer(node.results["ds_raw"])
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results

    node.outcomes = {
        pair_name: (Outcome.SUCCESSFUL if result["success"] else Outcome.FAILED)
        for pair_name, result in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot 2D PCA-based metric map over detuning and buffer duration."""
    fig = plot_detuning_vs_buffer_pca_map(
        node.results["ds_fit"],
        metric_name=node.parameters.pca_metric,
    )
    node.results["figure"] = fig
    node.results["figures"] = {"detuning_vs_buffer_pca_map": fig}
    annotate_node_figures(node)
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No persistent state update yet for this exploratory 2D scan."""
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

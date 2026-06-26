# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome

from quam_config import Quam

from calibration_utils.common_utils.experiment import (
    get_qubits,
    progress_counter_with_log,
    suppress_fetcher_axis_log_spam,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.reset_pulse_detuning_barrier import (
    Parameters,
    plot_2d_summary,
    analyze_reset_pulse_maps,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        RESET PULSE CALIBRATION (DETUNING x BARRIER VOLTAGE)

This node calibrates an active-reset pulse applied after a first measurement.
The reset pulse drive settings are fixed (amplitude and frequency detuning),
while the dot-pair detuning and barrier-gate voltage are swept in 2D.
The sequence is: empty -> conditional reset -> [optional op] -> measure.
"""

node = QualibrationNode[Parameters, Quam](
    name="07d_reset_pulse_detuning_barrier",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local parameter edits for debugging."""
    node.parameters.qubits = ["q1"]
    node.parameters.simulate = False
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create sweep axes and generate the reset pulse 2D program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)

    det_half_span = 0.5 * float(node.parameters.detuning_span)
    det_center = float(node.parameters.detuning_center)
    det_step = float(node.parameters.detuning_step)
    detuning_values = np.arange(
        det_center - det_half_span,
        det_center + det_half_span + 0.5 * det_step,
        det_step,
    )

    bar_half_span = 0.5 * float(node.parameters.barrier_gate_voltage_span)
    bar_center = float(node.parameters.barrier_gate_voltage_center)
    bar_step = float(node.parameters.barrier_gate_voltage_step)
    barrier_values = np.arange(
        bar_center - bar_half_span,
        bar_center + bar_half_span + 0.5 * bar_step,
        bar_step,
    )

    node.namespace["sweep_axes"] = {
        "detuning": xr.DataArray(
            detuning_values,
            attrs={"long_name": "dot-pair detuning", "units": "V"},
        ),
        "barrier_gate_voltage": xr.DataArray(
            barrier_values,
            attrs={"long_name": "barrier gate voltage", "units": "V"},
        ),
    }

    reset_operation = node.parameters.reset_operation
    drive_amplitude = float(node.parameters.drive_amplitude_scale)
    drive_detuning_hz = int(node.parameters.drive_frequency_detuning_MHz * u.MHz)

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_output_stream()
        detuning = declare(fixed)
        barrier = declare(fixed)

        init_state_int = {q.name: declare(int) for q in qubits}
        state_int = {q.name: declare(int) for q in qubits}
        init_i_placeholder = declare(fixed, value=0.0)
        init_q_placeholder = declare(fixed, value=0.0)

        CONDITIONS = [0, 1]
        init_state_st = {q.name: {c: declare_output_stream() for c in CONDITIONS} for q in qubits}
        init_i_st    = {q.name: {c: declare_output_stream() for c in CONDITIONS} for q in qubits}
        init_q_st    = {q.name: {c: declare_output_stream() for c in CONDITIONS} for q in qubits}
        state_st     = {q.name: {c: declare_output_stream() for c in CONDITIONS} for q in qubits}
        i_st         = {q.name: {c: declare_output_stream() for c in CONDITIONS} for q in qubits}
        q_st         = {q.name: {c: declare_output_stream() for c in CONDITIONS} for q in qubits}

        for val, cond in zip([False, True], CONDITIONS):
            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                for qubit in qubits:
                    qd_pair = node.machine.quantum_dot_pairs[
                        node.machine.find_quantum_dot_pair(
                            qubit.quantum_dot.name,
                            qubit.preferred_readout_quantum_dot,
                        )
                    ]
                    with for_(*from_array(detuning, detuning_values)):
                        with for_(*from_array(barrier, barrier_values)):
                            voltages = {
                                qd_pair.name: detuning,
                                qd_pair.barrier_gate.name: barrier,
                            }

                            qubit.empty()

                            init_state = qubit.empty(
                                measure_and_conditional_drive=True,
                                state="less_than_one",
                                drive_at_readout_point=True,
                                pulse_name=reset_operation,
                                amplitude_scale=drive_amplitude,
                                frequency_detuning_Hz=drive_detuning_hz,
                                ramp_duration=node.parameters.ramp_duration,
                                buffer_duration=node.parameters.buffer_duration,
                                hold_duration=node.parameters.hold_duration,
                                drive_point=voltages,
                            )
                            init_i = init_i_placeholder
                            init_q = init_q_placeholder

                            if val:
                                align()
                                qubit.apply(node.parameters.operation)
                                align()

                            i, q, state = qubit.measure(return_iq=True)

                            qubit.voltage_sequence.ramp_to_zero()

                            align()

                            assign(init_state_int[qubit.name], Cast.to_int(init_state))
                            save(init_state_int[qubit.name], init_state_st[qubit.name][cond])
                            save(init_i, init_i_st[qubit.name][cond])
                            save(init_q, init_q_st[qubit.name][cond])

                            assign(state_int[qubit.name], Cast.to_int(state))
                            save(state_int[qubit.name], state_st[qubit.name][cond])
                            save(i, i_st[qubit.name][cond])
                            save(q, q_st[qubit.name][cond])

        with stream_processing():
            n_st.save("n")
            for q in qubits:
                for c in CONDITIONS:
                    init_state_st[q.name][c].buffer(len(barrier_values)).buffer(len(detuning_values)).average().save(f"init_state_{q.name}_{c}")
                    init_i_st[q.name][c].buffer(len(barrier_values)).buffer(len(detuning_values)).average().save(f"init_I_{q.name}_{c}")
                    init_q_st[q.name][c].buffer(len(barrier_values)).buffer(len(detuning_values)).average().save(f"init_Q_{q.name}_{c}")
                    state_st[q.name][c].buffer(len(barrier_values)).buffer(len(detuning_values)).average().save(f"state_{q.name}_{c}")
                    i_st[q.name][c].buffer(len(barrier_values)).buffer(len(detuning_values)).average().save(f"I_{q.name}_{c}")
                    q_st[q.name][c].buffer(len(barrier_values)).buffer(len(detuning_values)).average().save(f"Q_{q.name}_{c}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to QOP and simulate the QUA program."""
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
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch raw data."""
    suppress_fetcher_axis_log_spam()
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        n_conditions = 2  # no-op and with-op conditions
        last_n = -1
        epoch = 0
        for dataset in data_fetcher:
            current_n = int(data_fetcher.get("n", 0))
            if current_n < last_n:
                epoch += 1
            last_n = current_n
            effective_n = epoch * node.parameters.num_shots + current_n
            progress_counter_with_log(
                effective_n,
                node.parameters.num_shots * n_conditions,
                start_time=data_fetcher.t_start,
                node=node,
                total_averages=node.parameters.num_shots * n_conditions,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse state/I difference between no-op and with-op conditions."""
    qubits = node.namespace["qubits"]
    ds_raw = node.results["ds_raw"]
    qubit_names = [q.name for q in qubits]
    fit_results, optimal_points, map_results = analyze_reset_pulse_maps(ds_raw, qubit_names)
    node.results["fit_results"] = fit_results
    node.results["optimal_points"] = optimal_points
    node.results["map_results"] = map_results

    for q_name, result in fit_results.items():
        node.log(
            f"  {q_name}: max |state_diff| at detuning="
            f"{result['optimal_detuning']:.4f}, "
            f"barrier={result['optimal_barrier_gate_voltage']:.4f}, "
            f"|state_diff|={result['max_abs_state_diff']:.4f}, "
            f"|I_diff|={result['max_abs_i_diff']:.4f}"
        )

    node.outcomes = {
        q_name: (Outcome.SUCCESSFUL if result["success"] else Outcome.FAILED)
        for q_name, result in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot init map, raw pulse/no-pulse maps, and diff maps."""
    qubits = node.namespace["qubits"]
    q_names = [q.name for q in qubits]

    figures = plot_2d_summary(
        node.results["ds_raw"],
        q_names,
        fit_results=node.results.get("fit_results"),
        plot_init_i=False,
    )

    node.results["figures"] = figures
    annotate_node_figures(node)


# %%
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Placeholder for state updates after calibration."""
    pass


# %%
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

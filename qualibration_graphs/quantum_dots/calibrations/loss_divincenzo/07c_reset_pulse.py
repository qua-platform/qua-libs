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
    enable_dual_drive_mw,
    progress_counter_with_log,
    suppress_fetcher_axis_log_spam,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.reset_pulse import (
    Parameters,
    plot_2d_summary,
    analyze_reset_pulse_maps,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        RESET PULSE CALIBRATION (FREQUENCY DETUNING x AMPLITUDE)

In this node, we calibrate an active-reset pulse applied after a first measurement.
The sequence is: init -> reset pulse -> measure -> operating pulse -> measure.
The reset pulse detuning and amplitude scale are swept in 2D while the operating pulse
remains fixed. The goal is to minimize the post-reset excited-state probability.
"""

node = QualibrationNode[Parameters, Quam](
    name="07c_reset_pulse",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local parameter edits for debugging."""
    node.parameters.qubits = ["q1"]
    node.parameters.simulate = False
    node.parameters.detuning = 0.8
    node.parameters.drive_frequency_detuning_span_MHz = 1
    node.parameters.drive_frequency_detuning_step_MHz = 0.1
    node.parameters.drive_amplitude_scale_span = 1
    node.parameters.drive_amplitude_scale_step = 0.1
    node.parameters.simulation_duration_ns = 40_000
    node.parameters.num_shots = 100
    pass

node.machine = Quam.load()

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the reset pulse calibration program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)

    f_span = int(node.parameters.drive_frequency_detuning_span_MHz * u.MHz)
    f_step = int(node.parameters.drive_frequency_detuning_step_MHz * u.MHz)
    dfs = np.arange(-f_span // 2, f_span // 2, f_step, dtype=int)

    a_span = float(node.parameters.drive_amplitude_scale_span)
    a_step = float(node.parameters.drive_amplitude_scale_step)
    das = np.arange(1 - a_span / 2, 1 + a_span / 2, a_step)

    node.namespace["sweep_axes"] = {
        "frequency_detuning": xr.DataArray(
            dfs,
            attrs={"long_name": "frequency detuning", "units": "Hz"},
        ),
        "amplitude_scale": xr.DataArray(
            das,
            attrs={"long_name": "amplitude scale", "units": ""},
        ),
    }

    reset_operation = node.parameters.reset_operation

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw(node)

        n = declare(int)
        n_st = declare_output_stream()
        df = declare(int)
        a = declare(fixed)

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
                    qubit_pair = next(
                        (qp for qp in node.machine.qubit_pairs.values() if qp.quantum_dot_pair is qd_pair),
                        None,
                    )

                    voltages = None
                    if (
                        node.parameters.detuning is not None
                        or node.parameters.barrier_gate_voltage is not None
                    ):
                        voltages = {}
                        if node.parameters.detuning is not None:
                            voltages[qd_pair.name] = node.parameters.detuning
                        if node.parameters.barrier_gate_voltage is not None:
                            voltages[qd_pair.barrier_gate.name] = (
                                node.parameters.barrier_gate_voltage
                            )
                    with for_(*from_array(df, dfs)):
                        with for_(*from_array(a, das)):

                            # Conditional-reset path via the new balanced empty macro.
                            # Keep init_I/init_Q scaffolding as placeholders so the
                            # existing analysis/plot code can remain unchanged for now.
                            init_state = qubit.empty(
                                measure_and_conditional_drive=True,
                                state="less_than_one",
                                drive_at_readout_point=True,
                                pulse_name=reset_operation,
                                amplitude_scale=a,
                                frequency_detuning_Hz=df,
                                ramp_duration=node.parameters.ramp_duration,
                                buffer_duration=node.parameters.buffer_duration,
                                hold_duration=node.parameters.hold_duration,
                                drive_point=voltages,
                            )
                            init_i = init_i_placeholder
                            init_q = init_q_placeholder

                            # Should be at zero at this point. 
                            if val: 
                                align()
                                qubit.apply(node.parameters.operation)
                                align()
                            else: 
                                align()
                                gates = [ch_name for ch_name in qd_pair.voltage_sequence.gate_set.channels.keys()]
                                wait(qubit.xy.operations[f"{node.machine.pulse_family}_{node.parameters.operation}"].length//4, qubit.xy.id, qd_pair.sensor_dots[0].readout_resonator.id, *gates)
                                align()

                            # Aligned, now measure
                            (i, q, state) = qubit.measure(return_iq=True)

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
                    init_state_st[q.name][c].buffer(len(das)).buffer(len(dfs)).average().save(f"init_state_{q.name}_{c}")
                    init_i_st[q.name][c].buffer(len(das)).buffer(len(dfs)).average().save(f"init_I_{q.name}_{c}")
                    init_q_st[q.name][c].buffer(len(das)).buffer(len(dfs)).average().save(f"init_Q_{q.name}_{c}")
                    state_st[q.name][c].buffer(len(das)).buffer(len(dfs)).average().save(f"state_{q.name}_{c}")
                    i_st[q.name][c].buffer(len(das)).buffer(len(dfs)).average().save(f"I_{q.name}_{c}")
                    q_st[q.name][c].buffer(len(das)).buffer(len(dfs)).average().save(f"Q_{q.name}_{c}")

# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
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
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch the raw data."""
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
    """Load a previously acquired dataset."""
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
    fit_results, optimal_points, map_results = analyze_reset_pulse_maps(
        ds_raw,
        qubit_names,
    )
    node.results["fit_results"] = fit_results
    node.results["optimal_points"] = optimal_points
    node.results["map_results"] = map_results

    for q_name, result in fit_results.items():
        node.log(
            f"  {q_name}: max |state_diff| at detuning="
            f"{result['optimal_frequency_detuning_hz'] / 1e6:.3f} MHz, "
            f"amp={result['optimal_amplitude_scale']:.3f}, "
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
    """Update the measure macro to include the pulse."""
    # TODO: Update measure macro
    pass    

# %%
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

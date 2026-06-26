# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_qubit_pairs, enable_dual_drive_mw_pairs
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.parity_streams import process_parity_streams
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.geometric_cz_duration import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        GEOMETRIC CZ DURATION CALIBRATION - fixed exchange amplitude
This node calibrates the exchange pulse duration for a geometric CZ gate at a fixed exchange amplitude.
It runs the cphase Ramsey sequence from node 16 (control in |0> or |1>), with X90 and Y90
closing pulses for quadrature readout (I and Q).


The analysis uses X90/Y90 quadrature readout, extracts branch phases with arctan2, and selects the
exchange duration at which the conditional phase (phi1 - phi0 - pi) crosses pi on the unwrapped
trace.  The common-mode Stark shift cancels in the subtraction.

Prerequisites:
    - Having calibrated single-qubit gates (X90, X180) for both qubits.
    - Having calibrated the readout for the qubit pair (parity readout).
    - Having chosen a fixed exchange amplitude for the barrier gate.

State update:
    - CZ voltage point on qubit pair (barrier gate voltage)
    - CZ macro duration
"""


node = QualibrationNode[Parameters, Quam](
    name="16a_geometric_cz_duration_calibration",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 2
    pass


node.machine = Quam.load()


def _get_cz_exchange_amplitude(qubit_pair) -> float:
    """Read the saved CZ barrier-gate voltage for a qubit pair."""
    point_name = qubit_pair.quantum_dot_pair._create_point_name("CZ")
    tuning_point = qubit_pair.voltage_sequence.gate_set.macros.get(point_name)
    if tuning_point is None:
        return float("nan")
    barrier_gate = qubit_pair.quantum_dot_pair.barrier_gate
    try:
        return float(tuning_point.voltages[barrier_gate.id])
    except KeyError:
        return float("nan")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the duration sweep axis and generate the QUA program."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    exchange_amplitude = _get_cz_exchange_amplitude(qubit_pairs[0])
    if not np.isfinite(exchange_amplitude):
        raise ValueError(
            "No CZ exchange amplitude found in state machine for "
            f"{qubit_pairs[0].name}. Run node 16 first."
        )
    node.namespace["exchange_amplitude"] = exchange_amplitude

    for gate_set_id in {qp.voltage_sequence.gate_set.id for qp in qubit_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)

    duration_array = np.arange(
        node.parameters.min_exchange_duration_in_ns,
        node.parameters.max_exchange_duration_in_ns,
        node.parameters.duration_step_in_ns,
    ).astype(int)

    control_state_values = [0, 1]
    analysis_axis_values = [0, 1]

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "control_state": xr.DataArray(
            control_state_values,
            attrs={
                "long_name": "control state",
                "units": "",
                "description": "0=control in |0>, 1=control in |1>",
            },
        ),
        "analysis_axis": xr.DataArray(
            analysis_axis_values,
            attrs={
                "long_name": "analysis quadrature",
                "units": "",
                "description": "0=final X90, 1=final Y90",
            },
        ),
        "exchange_duration": xr.DataArray(
            duration_array,
            attrs={"long_name": "exchange duration", "units": "ns"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw_pairs(node)

        n = declare(int)
        n_st = declare_output_stream()

        duration = declare(int)

        p2 = declare(int)
        p_st = {qp.name: declare_output_stream() for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                for control_state_val in control_state_values:
                    for analysis_axis_val in analysis_axis_values:
                        with for_(*from_array(duration, duration_array)):
                            qubit_pair.initialize()

                            align()

                            if control_state_val == 0:
                                qubit_pair.qubit_target.x90()
                            elif control_state_val == 1:
                                qubit_pair.qubit_control.x180()
                                align(qubit_pair.qubit_control.xy.name, qubit_pair.qubit_target.xy.name)
                                qubit_pair.qubit_target.x90()

                            align()

                            # Exchange pulse
                            qubit_pair.cz(
                                wait_duration=duration
                            )

                            align()
                            if analysis_axis_val == 1:
                                frame_rotation_2pi(
                                    0.25, qubit_pair.qubit_target.xy.name
                                )
                                qubit_pair.qubit_target.x90()
                                frame_rotation_2pi(
                                    -0.25, qubit_pair.qubit_target.xy.name
                                )
                            else:
                                qubit_pair.qubit_target.x90()

                            if control_state_val == 1:
                                qubit_pair.qubit_control.x180()

                            align()

                            a2 = qubit_pair.measure()

                            align()

                            qubit_pair.cz.balance()

                            align()

                            qubit_pair.voltage_sequence.ramp_to_zero()

                            align()

                            assign(p2, Cast.to_int(a2))
                            save(p2, p_st[qubit_pair.name])

        with stream_processing():
            n_st.save("n")

            n_control_states = len(control_state_values)
            n_analysis_axes = len(analysis_axis_values)
            n_durations = len(duration_array)
            for qubit_pair in qubit_pairs:
                p_st[qubit_pair.name].buffer(
                    n_control_states, n_analysis_axes, n_durations
                ).average().save(f"p_{qubit_pair.name}")


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
    """Connect to the QOP, execute the QUA program and fetch raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
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
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome streams."""
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [qp.name for qp in node.namespace["qubit_pairs"]],
        parity_pre_measurement=False,
        item_dim="qubit_pair",
        sweep_dims=("control_state", "analysis_axis", "exchange_duration"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit phase oscillations and extract the pi cphase duration."""
    ds_fit, fit_results = fit_raw_data(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        exchange_amplitude=node.namespace["exchange_amplitude"],
        analysis_signal=node.parameters.analysis_signal,
        quadrature_signal_center=node.parameters.quadrature_signal_center,
    )
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        name: ("successful" if r["success"] else "failed")
        for name, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot duration traces, fitted branches, and summed phase."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubit_pairs"],
        node.results.get("fit_results", {}),
        analysis_signal=node.parameters.analysis_signal,
        quadrature_signal_center=node.parameters.quadrature_signal_center,
    )
    node.results["figure"] = fig
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the CZ voltage point and macro duration for successful pairs."""

    with node.record_state_updates():
        for qubit_pair in node.namespace["qubit_pairs"]:
            if not node.results["fit_results"][qubit_pair.name]["success"]:
                continue
            fit_result = node.results["fit_results"][qubit_pair.name]
            qubit_pair.macros["cz"].wait_duration = int(
                round(fit_result["optimal_duration"] / 4)
            ) * 4


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

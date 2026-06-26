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
from calibration_utils.common_utils.parity_streams import (
    declare_parity_streams,
    save_parity_measurement,
    buffer_parity_streams,
    process_parity_streams,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.geometric_cz import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)


# %% {Node initialisation}
description = """
        GEOMETRIC CZ GATE CALIBRATION - quadrature readout (I/Q)
This node calibrates a geometric controlled-Z (CZ) gate by sweeping exchange pulse amplitude and duration.
It runs the cphase Ramsey sequence with X90 and Y90 closing pulses for quadrature readout (I and Q),
for both control qubit states (|0> and |1>).

The analysis extracts the branch phases phi0, phi1 via arctan2(Q-c, I-c), then computes the
conditional phase phi1 - phi0 - pi.  The common-mode Stark shift from the barrier gate
cancels in the subtraction; the -pi removes the known parity-readout offset.  The CZ
operating point is where this quantity reaches pi at the user-specified target duration.

This measurement performs a 2D sweep of exchange pulse amplitude vs duration with four experiment types:
    control_state=0, analysis_axis=0 : target X90, exchange, closing X90 (I quadrature, ctrl ground)
    control_state=0, analysis_axis=1 : target X90, exchange, closing Y90 (Q quadrature, ctrl ground)
    control_state=1, analysis_axis=0 : ctrl X180 + target X90, exchange, closing X90 (I, ctrl excited)
    control_state=1, analysis_axis=1 : ctrl X180 + target X90, exchange, closing Y90 (Q, ctrl excited)

Prerequisites:
    - Having calibrated single-qubit gates (X90, X180) for both qubits.
    - Having calibrated the readout for the qubit pair (parity readout).
    - Having characterized the exchange coupling vs barrier voltage (from CROT spectroscopy).
    - Having set appropriate voltage points for initialization, operation, and exchange.

State update:
    - CZ voltage point on qubit pair (barrier gate voltage)
    - CZ macro duration
"""


node = QualibrationNode[Parameters, Quam](
    name="16_geometric_cz_calibration",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 10


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    for gate_set_id in {qp.voltage_sequence.gate_set.id for qp in qubit_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)

    amplitude_array = np.arange(
        node.parameters.min_exchange_amplitude,
        node.parameters.max_exchange_amplitude,
        node.parameters.amplitude_step,
    )
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
        "exchange_amplitude": xr.DataArray(
            amplitude_array,
            attrs={"long_name": "barrier gate voltage", "units": "V"},
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

        amplitude = declare(fixed)
        duration = declare(int)

        p2, p1, parity_streams = declare_parity_streams(node, qubit_pairs)

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                for control_state_val in control_state_values:
                    for analysis_axis_val in analysis_axis_values:
                        with for_(*from_array(amplitude, amplitude_array)):
                            with for_(*from_array(duration, duration_array)):

                                reset_frame(
                                    qubit_pair.qubit_target.xy.name,
                                    qubit_pair.qubit_control.xy.name
                                )

                                if node.parameters.parity_pre_measurement:
                                    qubit_pair.empty()
                                    a1 = qubit_pair.measure()

                                qubit_pair.initialize()

                                align()

                                if control_state_val == 0:
                                    qubit_pair.qubit_target.x90()
                                elif control_state_val == 1:
                                    qubit_pair.qubit_control.x180()
                                    align(qubit_pair.qubit_control.xy.name, qubit_pair.qubit_target.xy.name)
                                    qubit_pair.qubit_target.x90()

                                align()

                                qubit_pair.cz(
                                    point={
                                        qubit_pair.quantum_dot_pair.barrier_gate.id: amplitude
                                    },
                                    wait_duration=duration,
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

                                align()

                                a2 = qubit_pair.measure()

                                align()

                                qubit_pair.cz.balance()

                                align()

                                qubit_pair.voltage_sequence.ramp_to_zero()

                                align()

                                assign(p2, Cast.to_int(a2))

                                if node.parameters.parity_pre_measurement:
                                    assign(p1, Cast.to_int(a1))

                                save_parity_measurement(node, qubit_pair.name, p1, p2, parity_streams)

        with stream_processing():
            n_st.save("n")

            n_control_states = len(control_state_values)
            n_analysis_axes = len(analysis_axis_values)
            n_amplitudes = len(amplitude_array)
            n_durations = len(duration_array)
            for qubit_pair in qubit_pairs:
                buffer_parity_streams(
                    node,
                    qubit_pair.name,
                    parity_streams,
                    n_control_states,
                    n_analysis_axes,
                    n_amplitudes,
                    n_durations,
                )


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
        parity_pre_measurement=node.parameters.parity_pre_measurement,
        item_dim="qubit_pair",
        sweep_dims=("control_state", "analysis_axis", "exchange_amplitude", "exchange_duration"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse: extract conditional phase and find CZ point."""
    ds_fit, fit_results = fit_raw_data(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        target_duration_ns=node.parameters.target_duration_ns,
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
    """Plot the 2D colormaps and CZ operating point."""
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
            barrier_gate = qubit_pair.quantum_dot_pair.barrier_gate
            qubit_pair.quantum_dot_pair.add_point(
                "CZ",
                voltages={barrier_gate.id: fit_result["optimal_amplitude"]},
            )
            qubit_pair.macros["cz"].wait_duration = int(
                round(fit_result["optimal_duration"])
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

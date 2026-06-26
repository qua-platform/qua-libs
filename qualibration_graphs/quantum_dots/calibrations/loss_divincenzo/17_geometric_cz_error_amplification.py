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

from calibration_utils.geometric_cz_error_amplification import (
    Parameters,
    fit_raw_data,
    plot_raw_data_with_fit,
    plot_raw_state_panels,
)

# %% {Node initialisation}
description = """
        GEOMETRIC CZ ERROR AMPLIFICATION - fixed duration, amplitude refinement
This node refines the exchange amplitude of a geometric CZ/CPhase gate using coherent error amplification.
It keeps the CZ duration fixed from the currently calibrated CZ macro and sweeps a small set of exchange
amplitudes around the saved CZ voltage point.

For each amplitude, the target qubit is prepared in a Ramsey sequence and the raw exchange CPhase pulse is
repeated N times. The control qubit is prepared in |0> or |1>, and two final analysis quadratures are measured
for the target qubit. The repetition axis defaults to a dense integer range, but can be replaced by an explicit
sparse list of N values. The central summary panel shows **Δφ(V, N)** (wrapped φ1−φ0). The optimal exchange
voltage and green 1D curve use **cos(Δφ)** — mean over N and the same 2D chevron fit as power-Rabi error
amplification. The **left panel** is a horizontal slice of the Δφ map (φ0, φ1, Δφ and cos(Δφ) vs N at the
reference amplitude; cyan dash-dot line on the center panel marks the same row).

Prerequisites:
    - Having calibrated the geometric CZ duration and saved the CZ voltage point (node 16 or 16a).
    - Having calibrated single-qubit gates (X90, X180) for both qubits.
    - Having calibrated the readout for the qubit pair (parity readout).

State update:
    - CZ voltage point on qubit pair (barrier gate voltage)
"""


node = QualibrationNode[Parameters, Quam](
    name="17_geometric_cz_error_amplification",
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
        raise ValueError(f"No saved CZ voltage point found for {qubit_pair.name}.")
    barrier_gate = qubit_pair.quantum_dot_pair.barrier_gate
    try:
        return float(tuning_point.voltages[barrier_gate.id])
    except KeyError as exc:
        raise ValueError(
            f"CZ voltage point for {qubit_pair.name} has no barrier gate "
            f"'{barrier_gate.id}' voltage."
        ) from exc


def _get_num_cphase_gate_array(parameters: Parameters) -> np.ndarray:
    """Build the raw CPhase repetition axis: 2, 4, …, max_num_cphase_gates."""
    max_gates = parameters.max_num_cphase_gates
    if max_gates < 2 or max_gates % 2 != 0:
        raise ValueError(
            f"max_num_cphase_gates must be a positive multiple of 2, got {max_gates}."
        )
    return np.arange(2, max_gates + 1, 2, dtype=int)


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the error-amplification sweep axes and generate the QUA program."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    for gate_set_id in {qp.voltage_sequence.gate_set.id for qp in qubit_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)

    exchange_amplitude_center = node.parameters.exchange_amplitude_center
    if exchange_amplitude_center is None:
        exchange_amplitude_center = _get_cz_exchange_amplitude(qubit_pairs[0])
        node.parameters.exchange_amplitude_center = exchange_amplitude_center

    amplitude_offsets = np.arange(
        -node.parameters.exchange_amplitude_span,
        node.parameters.exchange_amplitude_span
        + 0.5 * node.parameters.exchange_amplitude_step,
        node.parameters.exchange_amplitude_step,
    )
    amplitude_array = exchange_amplitude_center + amplitude_offsets
    num_cphase_gates = _get_num_cphase_gate_array(node.parameters)

    control_state_values = [0, 1]
    analysis_axis_values = [0, 1]

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "control_state": xr.DataArray(
            control_state_values,
            attrs={"long_name": "control state", "units": ""},
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
        "num_cphase_gates": xr.DataArray(
            num_cphase_gates,
            attrs={"long_name": "number of raw CPhase gates", "units": ""},
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw_pairs(node)

        n = declare(int)
        n_st = declare_output_stream()

        amplitude = declare(fixed)
        gate_count = declare(int)
        gate_idx = declare(int)

        p2, p1, parity_streams = declare_parity_streams(node, qubit_pairs)

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                for control_state_val in control_state_values:
                    for analysis_axis_val in analysis_axis_values:
                        with for_(*from_array(amplitude, amplitude_array)):
                            with for_(*from_array(gate_count, num_cphase_gates)):

                                reset_frame(
                                    qubit_pair.qubit_target.xy.name,
                                    qubit_pair.qubit_control.xy.name
                                )

                                if node.parameters.parity_pre_measurement:
                                    qubit_pair.empty()
                                    a1 = qubit_pair.measure()

                                qubit_pair.initialize()

                                align()

                                if control_state_val == 1:
                                    qubit_pair.qubit_control.x180()
                                qubit_pair.qubit_target.x90()

                                align()

                                with for_(
                                    gate_idx, 0, gate_idx < gate_count, gate_idx + 1
                                ):
                                    # Exchange pulse (zero out phase shifts to prevent
                                    # frame drift when re-running after node 18)
                                    qubit_pair.cz(
                                        point={
                                            qubit_pair.quantum_dot_pair.barrier_gate.id: amplitude
                                        },
                                        phase_shift_target=0,
                                        phase_shift_control=0,
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
            n_gate_counts = len(num_cphase_gates)
            for qubit_pair in qubit_pairs:
                buffer_parity_streams(node, qubit_pair.name, parity_streams, n_control_states, n_analysis_axes, n_amplitudes, n_gate_counts)


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
        sweep_dims=(
            "control_state",
            "analysis_axis",
            "exchange_amplitude",
            "num_cphase_gates",
        ),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse amplified CPhase errors and select the refined CZ amplitude."""
    ds_fit, fit_results = fit_raw_data(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        analysis_signal=node.parameters.analysis_signal,
        quadrature_signal_center=node.parameters.quadrature_signal_center,
    )
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    node.outcomes = {
        name: ("successful" if r["success"] else "failed")
        for name, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot amplified Ramsey traces and residual phase versus amplitude."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubit_pairs"],
        node.results.get("fit_results", {}),
        analysis_signal=node.parameters.analysis_signal,
        quadrature_signal_center=node.parameters.quadrature_signal_center,
        exchange_amplitude_center=node.parameters.exchange_amplitude_center,
    )
    node.results["figure"] = fig

    fig_panels = plot_raw_state_panels(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        analysis_signal=node.parameters.analysis_signal,
    )
    node.results["fig_raw_panels"] = fig_panels
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the CZ voltage point amplitude for successful pairs."""

    with node.record_state_updates():
        for qubit_pair in node.namespace["qubit_pairs"]:
            if not node.results["fit_results"][qubit_pair.name]["success"]:
                continue
            fit_result = node.results["fit_results"][qubit_pair.name]
            barrier_gate = qubit_pair.quantum_dot_pair.barrier_gate
            qubit_pair.add_point(
                "CZ",
                voltages={barrier_gate.id: fit_result["optimal_amplitude"]},
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

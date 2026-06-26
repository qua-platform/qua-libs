# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_qubit_pairs, enable_dual_drive_mw_pairs
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.parity_streams import process_parity_streams
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.geometric_cz_amplitude import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        GEOMETRIC CZ AMPLITUDE CALIBRATION - fixed exchange duration
This node calibrates the exchange pulse amplitude for a geometric CZ gate at a fixed exchange duration.
It runs cphase Ramsey experiments for control in |0> and |1>, each with X90 and Y90 analysis quadratures.

The analysis extracts phases via arctan2(I, Q) and selects the amplitude where the conditional phase
φ₁ − φ₀ equals π (unwrapped along the amplitude sweep).

Prerequisites:
    - Having calibrated single-qubit gates (X90, X180) for both qubits.
    - Having calibrated the readout for the qubit pair (parity readout).
    - Having chosen a fixed exchange duration for the CZ gate (e.g. from node 16a).

State update:
    - CZ voltage point on qubit pair (barrier gate voltage)
"""


node = QualibrationNode[Parameters, Quam](
    name="16b_geometric_cz_amplitude_calibration",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 2
    pass


node.machine = Quam.load()


def _get_cz_exchange_duration(qubit_pair) -> int:
    """Read the saved CZ exchange duration for a qubit pair (ns)."""
    cz_macro = qubit_pair.macros.get("cz")
    if cz_macro is None:
        return 0
    duration = getattr(cz_macro, "wait_duration", None)
    if duration is None:
        return 0
    return int(duration)


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the amplitude sweep axis and generate the QUA program."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    exchange_duration = _get_cz_exchange_duration(qubit_pairs[0])
    if exchange_duration <= 0:
        raise ValueError(
            "No CZ exchange duration found in state machine for "
            f"{qubit_pairs[0].name}. Run node 16 or 16a first."
        )
    node.namespace["exchange_duration"] = exchange_duration

    for gate_set_id in {qp.voltage_sequence.gate_set.id for qp in qubit_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)

    amplitude_array = np.arange(
        node.parameters.min_exchange_amplitude,
        node.parameters.max_exchange_amplitude,
        node.parameters.amplitude_step,
    )

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
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw_pairs(node)

        n = declare(int)
        n_st = declare_output_stream()

        amplitude = declare(fixed)

        p2 = declare(int)
        p_st = {qp.name: declare_output_stream() for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                barrier_gate_id = qubit_pair.quantum_dot_pair.barrier_gate.id
                for control_state_val in control_state_values:
                    for analysis_axis_val in analysis_axis_values:
                        with for_(*from_array(amplitude, amplitude_array)):
                            qubit_pair.initialize()

                            align()

                            if control_state_val == 0:
                                qubit_pair.qubit_target.x90()
                            else:
                                qubit_pair.qubit_control.x180()
                                align(qubit_pair.qubit_control.xy.name, qubit_pair.qubit_target.xy.name)
                                qubit_pair.qubit_target.x90()

                            align()

                            # Exchange pulse
                            qubit_pair.cz(
                                point={barrier_gate_id: amplitude},
                            )

                            align()
                            if analysis_axis_val == 1:
                                frame_rotation_2pi(0.25, qubit_pair.qubit_target.xy.name)
                                qubit_pair.qubit_target.x90()
                                frame_rotation_2pi(-0.25, qubit_pair.qubit_target.xy.name)
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
            n_amplitudes = len(amplitude_array)
            for qubit_pair in qubit_pairs:
                p_st[qubit_pair.name].buffer(
                    n_control_states, n_analysis_axes, n_amplitudes
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
    qmm = node.machine.connect(timeout=node.parameters.timeout)
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
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
        sweep_dims=("control_state", "analysis_axis", "exchange_amplitude"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit phase oscillations and extract the pi cphase amplitude."""
    ds_fit, fit_results = fit_raw_data(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        exchange_duration=node.namespace["exchange_duration"],
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
    """Plot quadrature signals, conditional phase, and V_π (or legacy exp-chirp fits)."""
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
    """Update the CZ voltage point for successful pairs."""

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
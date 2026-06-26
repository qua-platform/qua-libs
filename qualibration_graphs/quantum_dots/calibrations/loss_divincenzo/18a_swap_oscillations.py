# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_qubit_pairs, enable_dual_drive_mw_pairs, suppress_fetcher_axis_log_spam
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.swap_oscillations import (
    Parameters,
    analyse_swap_oscillations,
    log_fitted_results,
    plot_swap_oscillations,
)


# %% {Node initialisation}
description = """
        SWAP OSCILLATIONS - 2D amplitude/duration sweep
This node measures swap oscillations as a function of exchange pulse amplitude and
duration.  The experiment is run twice per qubit pair: once reading out the control
qubit and once reading out the target qubit.

Sequence (per measured qubit):
    initialize → CZ(amplitude, duration) → measure

The analysis extracts the 2π oscillation period at each amplitude via FFT,
discarding amplitudes where the oscillation has low SNR (white-noise regime).
The extracted (amplitude, T_2π) curve is overlaid on the 2D heatmaps and saved
as structured fit results.

A polynomial model T_2π(V) (up to degree 3) is fitted to the valid
(amplitude, T_2π) curve.  The model parameters are stored on the CZ macro
(``exchange_decay_model``), enabling downstream nodes to evaluate T_2π at
any amplitude via ``qubit_pair.macros["cz"].t_2pi(V)`` or get the CZ
half-period via ``qubit_pair.macros["cz"].t_cz(V)``.

Prerequisites:
    - Having calibrated single-qubit gates for both qubits.
    - Having calibrated readout for both individual qubits.
    - Having set appropriate voltage points for initialization and operation.

State update:
    - CZ voltage point (barrier gate) for the best operating amplitude.
    - CZ macro wait_duration = T_2π / 2 (half-period for CZ gate).
    - CZ macro exchange_decay_model = polynomial fit (coeffs, degree).
"""


node = QualibrationNode[Parameters, Quam](
    name="18a_swap_oscillations",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubit_pairs = ["q1_q2"]
    # node.parameters.simulate = True
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the 2D sweep axes and generate the QUA program."""

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

    node.namespace["amplitude_array"] = amplitude_array
    node.namespace["duration_array"] = duration_array

    node.namespace["sweep_axes"] = {
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
        state = declare(int)

        control_streams = {qp.name: declare_output_stream() for qp in qubit_pairs}
        target_streams = {qp.name: declare_output_stream() for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                # --- Experiment 1: measure control qubit ---
                with for_(*from_array(amplitude, amplitude_array)):
                    with for_(*from_array(duration, duration_array)):

                        reset_frame(
                            qubit_pair.qubit_target.xy.name,
                            qubit_pair.qubit_control.xy.name,
                        )

                        qubit_pair.initialize()
                        align()

                        qubit_pair.cz(
                            point={
                                qubit_pair.quantum_dot_pair.barrier_gate.id: amplitude
                            },
                            wait_duration=duration,
                        )
                        align()

                        a = qubit_pair.qubit_control.measure()
                        assign(state, Cast.to_int(a))
                        save(state, control_streams[qubit_pair.name])

                        align()

                        qubit_pair.cz.balance()

                        align()

                        qubit_pair.voltage_sequence.ramp_to_zero(reset_tracker=True)
                        align()

                # --- Experiment 2: measure target qubit ---
                # with for_(*from_array(amplitude, amplitude_array)):
                #     with for_(*from_array(duration, duration_array)):

                        reset_frame(
                            qubit_pair.qubit_target.xy.name,
                            qubit_pair.qubit_control.xy.name,
                        )

                        qubit_pair.initialize()
                        align()

                        qubit_pair.cz(
                            point={
                                qubit_pair.quantum_dot_pair.barrier_gate.id: amplitude
                            },
                            wait_duration=duration,
                        )
                        align()

                        a = qubit_pair.qubit_target.measure()
                        assign(state, Cast.to_int(a))
                        save(state, target_streams[qubit_pair.name])

                        align()

                        qubit_pair.cz.balance()

                        align()

                        qubit_pair.voltage_sequence.ramp_to_zero()
                        
                        align()

        with stream_processing():
            n_st.save("n")

            n_amplitudes = len(amplitude_array)
            n_durations = len(duration_array)
            for qp in qubit_pairs:
                (
                    control_streams[qp.name]
                    .buffer(n_durations)
                    .buffer(n_amplitudes)
                    .average()
                    .save(f"state_control_{qp.name}")
                )
                (
                    target_streams[qp.name]
                    .buffer(n_durations)
                    .buffer(n_amplitudes)
                    .average()
                    .save(f"state_target_{qp.name}")
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
    suppress_fetcher_axis_log_spam()
    qmm = node.machine.connect(timeout=node.parameters.timeout)
    config = node.machine.generate_config()
    qubit_pairs = node.namespace["qubit_pairs"]
    amplitude_array = node.namespace["amplitude_array"]
    duration_array = node.namespace["duration_array"]

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # job.result_handles.wait_for_all_values()

        # dataset = xr.Dataset()
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        for qp in qubit_pairs:
            for role in ["control", "target"]:
                key = f"state_{role}_{qp.name}"
                data = job.result_handles.get(key).fetch_all()
                dataset[key] = xr.DataArray(
                    data,
                    dims=["exchange_amplitude", "exchange_duration"],
                    coords={
                        "exchange_amplitude": amplitude_array,
                        "exchange_duration": duration_array,
                    },
                    attrs={"long_name": f"{role} qubit state", "units": ""},
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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Extract 2π oscillation period at each amplitude via FFT."""
    ds_fit, fit_results = analyse_swap_oscillations(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        snr_threshold=node.parameters.snr_threshold,
        analysis_role=node.parameters.analysis_role,
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
    """Plot 2D swap-oscillation heatmaps with 2π overlay."""
    fig = plot_swap_oscillations(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        ds_fit=node.results.get("ds_fit"),
        fit_results=node.results.get("fit_results"),
    )
    node.results["figure"] = fig
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Store the best operating point and the T_2π(V) model in QuAM.

    - CZ voltage point  → best amplitude (highest SNR).
    - CZ macro wait_duration → T_2π / 2 at that amplitude.
    - CZ macro exchange_decay_model → polynomial fit (coeffs, degree).
    """
    with node.record_state_updates():
        for qubit_pair in node.namespace["qubit_pairs"]:
            fit = node.results["fit_results"].get(qubit_pair.name, {})
            if not fit.get("success"):
                continue

            barrier_gate = qubit_pair.quantum_dot_pair.barrier_gate
            qubit_pair.add_point(
                "CZ",
                voltages={barrier_gate.id: fit["best_amplitude"]},
            )

            half_period = fit["best_t_2pi"] / 2.0
            qubit_pair.macros["cz"].wait_duration = int(
                round(half_period / 4) * 4
            )

            if fit.get("model_fit_success") and fit.get("exchange_decay_model"):
                qubit_pair.macros["cz"].exchange_decay_model = dict(
                    fit["exchange_decay_model"]
                )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

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
from calibration_utils.common_utils.parity_streams import (
    process_parity_streams,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.geometric_cz_amplitude_phase import (
    Parameters,
    analyse_amplitude_phase,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        GEOMETRIC CZ AMPLITUDE–PHASE CALIBRATION
This node calibrates the CZ exchange-pulse amplitude using a 2-D sweep:
  axis 1 (amplitude): exchange-pulse barrier-gate voltage, as in node 16b.
  axis 2 (phase):     phase θ of the second (closing) π/2 pulse on the target qubit,
                      swept uniformly over [0, 2π).

The experiment runs a cphase Ramsey sequence for both control-qubit states (|0⟩ and |1⟩):
  1. X90 on target  (+ X180 on control for the ctrl=|1⟩ branch)
  2. Exchange pulse at the swept amplitude
  3. Frame-rotation by θ followed by X90 on target (analysis phase sweep)
  4. Parity measurement

The parity difference D(V, θ) = S(ctrl=|1⟩, V, θ) − S(ctrl=|0⟩, V, θ) is computed
for every (amplitude, phase) point.  The mean absolute difference ⟨|D|⟩_θ is
minimised at the optimal CZ amplitude V*.

Two duration modes (controlled by ``use_t2pi_model``):
  - **Fixed** (default): uses the single ``wait_duration`` from the CZ macro.
  - **Model-based**: evaluates the T_2π(V) polynomial from ``exchange_decay_model``
    (fitted in 18a_swap_oscillations) to set a per-amplitude CZ duration
    (T_2π / 2, rounded to the nearest 4 ns).

Prerequisites:
    - Calibrated single-qubit gates (X90, X180) for both qubits.
    - Calibrated parity readout for the qubit pair.
    - Fixed mode: a known CZ exchange duration (e.g. from node 16 or 18a).
    - Model mode: ``exchange_decay_model`` on the CZ macro (from 18a).

State update:
    - CZ voltage point (barrier gate) for the optimal amplitude.
    - CZ macro ``wait_duration`` = exchange duration at V*.
"""


node = QualibrationNode[Parameters, Quam](
    name="18b_geometric_cz_amplitude_phase_calibration",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 2
    # node.parameters.simulate = True
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


def _build_duration_array(qubit_pair, amplitude_array: np.ndarray) -> np.ndarray:
    """Evaluate T_2π model → per-amplitude CZ duration (ns, multiples of 4)."""
    cz_macro = qubit_pair.macros.get("cz")
    if cz_macro is None:
        raise ValueError(f"No CZ macro on {qubit_pair.name}.")
    model = getattr(cz_macro, "exchange_decay_model", None)
    if model is None:
        raise ValueError(
            f"No exchange_decay_model on {qubit_pair.name}.macros['cz']. "
            "Run 18a_swap_oscillations first."
        )
    t_2pi = np.polyval(model["coeffs"], amplitude_array)
    t_cz = t_2pi / 2.0
    duration_array = (np.round(t_cz / 4) * 4).astype(int)
    duration_array = np.clip(duration_array, 16, None)
    return duration_array


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build sweep axes and generate the QUA program."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    for gate_set_id in {qp.voltage_sequence.gate_set.id for qp in qubit_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)

    amplitude_array = np.arange(
        node.parameters.min_exchange_amplitude,
        node.parameters.max_exchange_amplitude,
        node.parameters.amplitude_step,
    )

    use_model = node.parameters.use_t2pi_model

    if use_model:
        duration_array = _build_duration_array(qubit_pairs[0], amplitude_array)
        node.namespace["duration_array"] = duration_array
        node.namespace["exchange_duration"] = duration_array
    else:
        exchange_duration = _get_cz_exchange_duration(qubit_pairs[0])
        if exchange_duration <= 0:
            raise ValueError(
                "No CZ exchange duration found for "
                f"{qubit_pairs[0].name}. Run node 16, 16a, or 18a first."
            )
        node.namespace["exchange_duration"] = exchange_duration
        node.namespace["duration_array"] = None

    # Phase in [0, 1): used directly as the argument to frame_rotation_2pi
    # (which rotates by phase × 2π radians).
    phase_array = np.linspace(0, 1, node.parameters.num_phases, endpoint=False)

    control_state_values = [0, 1]

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "control_state": xr.DataArray(
            control_state_values,
            attrs={"long_name": "control state", "units": ""},
        ),
        "exchange_amplitude": xr.DataArray(
            amplitude_array,
            attrs={"long_name": "barrier gate voltage", "units": "V"},
        ),
        "analysis_phase": xr.DataArray(
            phase_array * 2 * np.pi,  # store in radians for plotting
            attrs={"long_name": "analysis phase", "units": "rad"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw_pairs(node)

        n = declare(int)
        n_st = declare_output_stream()

        amplitude = declare(fixed)
        phase = declare(fixed)
        state = declare(int)

        if use_model:
            amp_qua = declare(fixed, value=amplitude_array.tolist())
            dur_qua = declare(int, value=duration_array.tolist())
            amp_idx = declare(int)
            duration = declare(int)

        target_streams = {qp.name: declare_output_stream() for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                barrier_gate_id = qubit_pair.quantum_dot_pair.barrier_gate.id

                for control_state_val in control_state_values:

                    if use_model:
                        amp_loop = for_(
                            amp_idx, 0,
                            amp_idx < len(amplitude_array),
                            amp_idx + 1,
                        )
                    else:
                        amp_loop = for_(*from_array(amplitude, amplitude_array))

                    with amp_loop:
                        if use_model:
                            assign(amplitude, amp_qua[amp_idx])
                            assign(duration, dur_qua[amp_idx])

                        with for_(*from_array(phase, phase_array)):

                            # Start each iteration from a clean frame so neither
                            # the swept analysis phase nor the CZ macro's constant
                            # phase_shift_control/target frame rotations accumulate
                            # across the loop.
                            reset_frame(
                                qubit_pair.qubit_control.xy.name,
                                qubit_pair.qubit_target.xy.name,
                            )

                            qubit_pair.initialize()

                            align()

                            if control_state_val == 0:
                                qubit_pair.qubit_target.x90()
                            else:
                                qubit_pair.qubit_control.x180()
                                align(
                                    qubit_pair.qubit_control.xy.name,
                                    qubit_pair.qubit_target.xy.name,
                                )
                                qubit_pair.qubit_target.x90()

                            align()

                            # Analysis π/2 with swept phase: the swept frame
                            # rotation θ ∈ [0, 2π) on the target is folded into
                            # the CZ macro's phase_shift_target, exactly as node
                            # 18 does — applied after the exchange leg, before the
                            # closing X90. The calibrated constant offset is
                            # irrelevant to the fringe, so we sweep phase directly.
                            if use_model:
                                qubit_pair.cz(
                                    point={barrier_gate_id: amplitude},
                                    wait_duration=duration,
                                    phase_shift_target=phase,
                                )
                            else:
                                qubit_pair.cz(
                                    point={barrier_gate_id: amplitude},
                                    phase_shift_target=phase,
                                )

                            align()

                            if control_state_val == 0:
                                qubit_pair.qubit_target.x90()
                            else:
                                qubit_pair.qubit_target.x90()
                                align(
                                    qubit_pair.qubit_control.xy.name,
                                    qubit_pair.qubit_target.xy.name,
                                )
                                qubit_pair.qubit_control.x180()
                                
                            align()

                            a = qubit_pair.measure()

                            align()

                            qubit_pair.cz.balance()

                            align()

                            qubit_pair.voltage_sequence.ramp_to_zero(reset_tracker=True)

                            align()

                            assign(state, Cast.to_int(a))
                            save(state, target_streams[qubit_pair.name])

        with stream_processing():
            n_st.save("n")

            n_control_states = len(control_state_values)
            n_amplitudes = len(amplitude_array)
            n_phases = len(phase_array)

            for qubit_pair in qubit_pairs:
                # Buffer order mirrors the loop nesting:
                # outermost → control_state, middle → amplitude, innermost → phase
                (
                    target_streams[qubit_pair.name]
                    .buffer(n_phases)
                    .buffer(n_amplitudes)
                    .buffer(n_control_states)
                    .average()
                    .save(f"p_{qubit_pair.name}")
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
    """Map the raw target-qubit stream into the canonical analysis variable."""
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [qp.name for qp in node.namespace["qubit_pairs"]],
        parity_pre_measurement=False,
        item_dim="qubit_pair",
        sweep_dims=("control_state", "exchange_amplitude", "analysis_phase"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Find V* = argmin ⟨|S(ctrl |1⟩) − S(ctrl |0⟩)|⟩_θ."""
    duration_array = node.namespace.get("duration_array")
    exchange_dur = node.namespace["exchange_duration"]
    ds_fit, fit_results = analyse_amplitude_phase(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        exchange_duration=exchange_dur if duration_array is None else 0,
        duration_array=duration_array,
        analysis_signal=node.parameters.analysis_signal,
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
    """Plot heatmaps, phase-averaged difference, and phase cuts at V*."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubit_pairs"],
        node.results.get("fit_results", {}),
        analysis_signal=node.parameters.analysis_signal,
    )
    node.results["figure"] = fig
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the CZ voltage point and duration for successful qubit pairs."""
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

            opt_dur = fit_result.get("optimal_duration", 0)
            if opt_dur > 0:
                cz_macro = qubit_pair.macros.get("cz")
                if cz_macro is not None:
                    cz_macro.update(wait_duration=opt_dur)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

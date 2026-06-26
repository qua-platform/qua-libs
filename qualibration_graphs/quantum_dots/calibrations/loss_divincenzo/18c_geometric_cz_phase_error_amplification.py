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

from calibration_utils.geometric_cz_phase_error_amplification import (
    Parameters,
    analyse_phase_error_amplification,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        GEOMETRIC CZ PHASE ERROR AMPLIFICATION
This node diagnoses and corrects single-qubit phase errors accumulated
during the CZ exchange pulse.

Two independent Ramsey error-amplification sweeps are run at the saved
CZ amplitude:

  Sweep A — target qubit phase sweep
    axis 1: number of repeated CZ pulses (multiples of 2, up to max_num_cphase_gates)
    axis 2: analysis phase θ on the TARGET qubit's closing π/2 pulse
    conditional: control qubit prepared in |0⟩ or |1⟩

  Sweep B — control qubit phase sweep
    axis 1: same gate-count axis
    axis 2: analysis phase θ on the CONTROL qubit's closing π/2 pulse
    conditional: target qubit prepared in |0⟩ or |1⟩

Both differences D(N,θ) = S(cond=|1⟩) − S(cond=|0⟩) are fitted to
extract the per-gate phase offset φ accumulated by each qubit.

At the correct CZ operating point each even-N repetition produces the same
⟨|D(N,θ)|⟩_θ (flat curve). The phase of the sinusoidal cut at peak N gives
the single-axis phase error:

    phase_compensation = −φ_acc / (N* · 2π)   [turns, for frame_rotation_2pi]

State update:
    - Writes phase_shift_target  to qubit_pair.macros["cz"]
    - Writes phase_shift_control to qubit_pair.macros["cz"]
"""

node = QualibrationNode[Parameters, Quam](
    name="18c_geometric_cz_phase_error_amplification",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 2


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
    """Build the CPhase repetition axis: 2, 4, …, max_num_cphase_gates."""
    max_gates = parameters.max_num_cphase_gates
    if max_gates < 2 or max_gates % 2 != 0:
        raise ValueError(
            f"max_num_cphase_gates must be a positive multiple of 2, got {max_gates}."
        )
    return np.arange(2, max_gates + 1, 2, dtype=int)


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build sweep axes and generate the QUA program."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    for gate_set_id in {qp.voltage_sequence.gate_set.id for qp in qubit_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)

    exchange_amplitude = node.parameters.exchange_amplitude_center
    if exchange_amplitude is None:
        exchange_amplitude = _get_cz_exchange_amplitude(qubit_pairs[0])
        node.parameters.exchange_amplitude_center = exchange_amplitude
    node.namespace["exchange_amplitude"] = exchange_amplitude

    num_cphase_gates = _get_num_cphase_gate_array(node.parameters)

    # Phase in [0, 1): used as argument to frame_rotation_2pi (rotates by phase × 2π).
    phase_array = np.linspace(0, 1, node.parameters.num_phases, endpoint=False)

    conditional_state_values = [0, 1]

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "control_state": xr.DataArray(
            conditional_state_values,
            attrs={"long_name": "conditional state", "units": ""},
        ),
        "num_cphase_gates": xr.DataArray(
            num_cphase_gates,
            attrs={"long_name": "number of CZ repetitions", "units": ""},
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

        gate_count = declare(int)
        gate_idx = declare(int)
        phase = declare(fixed)
        state = declare(int)

        # One stream per pair per sweep (target qubit sweep and control qubit sweep).
        target_streams = {qp.name: declare_output_stream() for qp in qubit_pairs}
        ctrl_streams = {qp.name: declare_output_stream() for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                barrier_gate_id = qubit_pair.quantum_dot_pair.barrier_gate.id

                # ── Sweep A: target qubit phase sweep ─────────────────────────
                # Conditional: control qubit in |cond_val⟩.
                # Ramsey: X90(target) – N×CZ – θ·X90(target) – measure(target).
                for cond_val in conditional_state_values:
                    with for_(*from_array(gate_count, num_cphase_gates)):
                        with for_(*from_array(phase, phase_array)):

                            reset_frame(
                                qubit_pair.qubit_target.xy.name,
                                qubit_pair.qubit_control.xy.name,
                            )
                            qubit_pair.initialize()
                            align()

                            if cond_val == 1:
                                qubit_pair.qubit_control.x180()
                                align(
                                    qubit_pair.qubit_control.xy.name,
                                    qubit_pair.qubit_target.xy.name,
                                )
                            qubit_pair.qubit_target.x90()
                            align()

                            with for_(gate_idx, 0, gate_idx < gate_count, gate_idx + 1):
                                qubit_pair.cz(
                                    point={barrier_gate_id: exchange_amplitude},
                                    phase_shift_target=0,
                                    phase_shift_control=0,
                                )

                            align()

                            frame_rotation_2pi(phase, qubit_pair.qubit_target.xy.name)
                            qubit_pair.qubit_target.x90()
                            align()

                            a = qubit_pair.qubit_target.measure()

                            align()

                            qubit_pair.cz.balance()

                            align()

                            qubit_pair.voltage_sequence.ramp_to_zero(reset_tracker=True)
                            align()

                            assign(state, Cast.to_int(a))
                            save(state, target_streams[qubit_pair.name])

                # ── Sweep B: control qubit phase sweep ────────────────────────
                # Conditional: target qubit in |cond_val⟩.
                # Ramsey: X90(control) – N×CZ – θ·X90(control) – measure(control).
                for cond_val in conditional_state_values:
                    with for_(*from_array(gate_count, num_cphase_gates)):
                        with for_(*from_array(phase, phase_array)):

                            reset_frame(
                                qubit_pair.qubit_target.xy.name,
                                qubit_pair.qubit_control.xy.name,
                            )
                            qubit_pair.initialize()
                            align()

                            if cond_val == 1:
                                qubit_pair.qubit_target.x180()
                                align(
                                    qubit_pair.qubit_control.xy.name,
                                    qubit_pair.qubit_target.xy.name,
                                )
                            qubit_pair.qubit_control.x90()
                            align()

                            with for_(gate_idx, 0, gate_idx < gate_count, gate_idx + 1):
                                qubit_pair.cz(
                                    point={barrier_gate_id: exchange_amplitude},
                                    phase_shift_target=0,
                                    phase_shift_control=0,
                                )

                            align()

                            frame_rotation_2pi(phase, qubit_pair.qubit_control.xy.name)
                            qubit_pair.qubit_control.x90()
                            align()

                            a = qubit_pair.qubit_control.measure()

                            align()

                            qubit_pair.cz.balance()

                            align()

                            qubit_pair.voltage_sequence.ramp_to_zero()
                            align()

                            assign(state, Cast.to_int(a))
                            save(state, ctrl_streams[qubit_pair.name])

        with stream_processing():
            n_st.save("n")

            n_cond_states = len(conditional_state_values)
            n_gate_counts = len(num_cphase_gates)
            n_phases = len(phase_array)

            for qubit_pair in qubit_pairs:
                # Buffer order mirrors loop nesting:
                # outermost → conditional_state, middle → gate_count, innermost → phase
                (
                    target_streams[qubit_pair.name]
                    .buffer(n_phases)
                    .buffer(n_gate_counts)
                    .buffer(n_cond_states)
                    .average()
                    .save(f"p_{qubit_pair.name}")
                )
                (
                    ctrl_streams[qubit_pair.name]
                    .buffer(n_phases)
                    .buffer(n_gate_counts)
                    .buffer(n_cond_states)
                    .average()
                    .save(f"p_ctrl_{qubit_pair.name}")
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
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node,
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
    """Map raw streams into the canonical analysis variables.

    Target stream:  p_{name}      → E_p2_given_p1_0_{name}
    Control stream: p_ctrl_{name} → E_p2_given_p1_0_ctrl_{name}
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    analysis_signal = node.parameters.analysis_signal

    # Target sweep — processed via the standard parity-stream utility.
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [qp.name for qp in qubit_pairs],
        parity_pre_measurement=False,
        item_dim="qubit_pair",
        sweep_dims=("control_state", "num_cphase_gates", "analysis_phase"),
    )

    # Control sweep — rename p_ctrl_{name} → {analysis_signal}_ctrl_{name}.
    ds = node.results["ds_raw"]
    renames = {}
    for qp in qubit_pairs:
        old = f"p_ctrl_{qp.name}"
        new = f"{analysis_signal}_ctrl_{qp.name}"
        if old in ds.data_vars:
            renames[old] = new
    if renames:
        ds = ds.rename(renames)
    node.results["ds_raw"] = ds


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit D(N,θ) for both sweeps; extract per-gate phase compensations."""
    ds_fit, fit_results = analyse_phase_error_amplification(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
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
    """Plot D(N,θ) heatmaps, ⟨|D|⟩_θ vs N, and phase cuts for both sweeps."""
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
    """Write fitted phase compensations to the CZ macro in QuAM."""
    with node.record_state_updates():
        for qubit_pair in node.namespace["qubit_pairs"]:
            fit = node.results["fit_results"].get(qubit_pair.name, {})
            macro = qubit_pair.macros["cz"]

            if fit.get("success_target"):
                comp_target = fit.get("phase_compensation_target", float("nan"))
                if np.isfinite(comp_target):
                    macro.update(phase_shift_target=comp_target)
                    node.log(
                        f"{qubit_pair.name}: phase_shift_target set to "
                        f"{comp_target:.5f} turns"
                    )

            if fit.get("success_control"):
                comp_control = fit.get("phase_compensation_control", float("nan"))
                if np.isfinite(comp_control):
                    macro.update(phase_shift_control=comp_control)
                    node.log(
                        f"{qubit_pair.name}: phase_shift_control set to "
                        f"{comp_control:.5f} turns"
                    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
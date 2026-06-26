# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import (
    Cast,
    align,
    assign,
    declare,
    declare_output_stream,
    for_,
    program,
    save,
    stream_processing,
    reset_frame,
)

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

from calibration_utils.iswap_error_amplification import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        RESIDUAL iSWAP ERROR AMPLIFICATION - fixed CPhase amplitude and duration
This node diagnoses the residual iSWAP/SWAP component of the calibrated geometric CZ/CPhase block.
It does not sweep exchange amplitude or duration. Instead, it reads the saved CZ voltage point and CZ
macro duration, repeats the raw exchange block at that fixed operating point, and fits the coherent
population-transfer buildup in the odd-parity subspace.

The protocol prepares |10> and |01>, applies repeated *two-beat* blocks (raw CPhase, π pulses, raw
CPhase, π pulses), and measures odd-subspace transfer. The π pulses are not an accident of gate
decomposition: they are part of a **component-selective** readout, not a naive stack of n identical
CZ operations. In an average-Hamiltonian (toggling-frame) picture, instantaneously applied local Pauli
pulses change which parts of a small two-qubit error Hamiltonian are coherently accumulated between
sub-blocks. Recent work on superconducting two-qubit hardware implements similar interleaved-π
constructions in the *two-qubit* context (tunable exchange / coupler), both to mitigate coherent
entangling error and to probe noise during two-qubit gates [1,2]. The original toggling-frame /
average-Hamiltonian formalism is older NMR theory [3]. Here, two `π` settings—`X⊗X` and `Y⊗X`—act as
quadrature projections of a small residual iSWAP-like term in the odd subspace, so the analysis can
fit a magnitude |theta| from the pair (not just a single unphysical projection of an unknown error
axis). Repeating only raw exchange `n` times with no `π` sandwich would coherently amplify *some*
unwanted rotation, but would not, by itself, supply the two independent traces required for the
`theta_x` / `theta_y` / `|theta_iswap|` decomposition implemented in
`calibration_utils.iswap_error_amplification` without changing the fit model.
Tradeoff: the sequence is more sensitive to π-pulse infidelity and calibration drift. Prerequisites
treat `X180` and `Y180` as pre-characterized; if those dominate, interpret results cautiously.
References: [1] J. Qiu et al., "Suppressing Coherent Two-Qubit Errors via Dynamical Decoupling",
   Phys. Rev. Applied 16, 054047 (2021), doi:10.1103/PhysRevApplied.16.054047 (tunable-coupler
   superconducting device: DD-style sequences in the entangling subspace, mixed X/Y control).
   [2] T. McCourt et al., "Learning Noise via Dynamical Decoupling of Entangled Qubits", Phys. Rev. A
   107, 052610 (2023), doi:10.1103/PhysRevA.107.052610 (DD sequences to characterize non-single-
   qubit-dephasing noise that is on during two-qubit gates, interleaving local pulses with
   entangling evolution).  [3] U. Haeberlen, J. S. Waugh, "Coherent Averaging Effects in Magnetic
   Resonance", Phys. Rev. 175, 453 (1968) (classical NMR reference for the toggling / average
   Hamiltonian picture underlying pulse interleaving).

The protocol measures odd-subspace transfer with two `dd_axis` settings:
    - X⊗X selects the residual swap component proportional to theta*cos(chi).
    - Y⊗X selects the complementary component proportional to theta*sin(chi).

The analysis reports the absolute component magnitudes and their quadrature sum |theta_iswap|. It is a
diagnostic node in v1; it does not update the CZ state.

Prerequisites:
    - Having calibrated the geometric CZ duration and saved the CZ voltage point (node 16/16a and node 17).
    - Having calibrated single-qubit X180 and Y180 gates for both qubits.
    - Having calibrated the readout for the qubit pair (parity readout).

State update:
    - None (diagnostic measurement).
"""


node = QualibrationNode[Parameters, Quam](
    name="17a_iswap_error_amplification",
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
    point_name = qubit_pair._create_point_name("CZ")
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


def _get_cz_duration_ns(qubit_pair) -> int:
    """Read the saved CZ macro wait_duration in nanoseconds."""
    cz_macro = qubit_pair.macros.get("cz")
    if cz_macro is None or getattr(cz_macro, "wait_duration", None) is None:
        raise ValueError(f"No saved CZ duration found for {qubit_pair.name}.")
    duration = int(round(float(cz_macro.wait_duration)))
    if duration < 16:
        raise ValueError(f"CZ duration must be >= 16 ns, got {duration} ns.")
    return duration


def _get_num_cycle_array(parameters: Parameters) -> np.ndarray:
    """Build the two-cycle repetition axis from dense or explicit settings."""
    if parameters.num_cycle_repetitions is not None:
        num_cycles = np.asarray(parameters.num_cycle_repetitions, dtype=int)
    else:
        if parameters.num_cycle_step <= 0:
            raise ValueError("num_cycle_step must be positive.")
        num_cycles = np.arange(
            parameters.min_num_cycles,
            parameters.max_num_cycles + 1,
            parameters.num_cycle_step,
            dtype=int,
        )

    if len(num_cycles) == 0:
        raise ValueError("The iSWAP repetition axis must not be empty.")
    if np.any(num_cycles < 0):
        raise ValueError("iSWAP repetition counts must be non-negative.")
    return np.unique(num_cycles)


def _apply_dd_pair(qubit_pair, dd_axis_val: int) -> None:
    """Apply the single-qubit DD pair for one half of the amplification cycle."""
    align()
    if dd_axis_val == 0:
        qubit_pair.qubit_target.x180()
        qubit_pair.qubit_control.x180()
    else:
        qubit_pair.qubit_target.x180()
        qubit_pair.qubit_control.y180()
    align()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the fixed-operating-point iSWAP amplification QUA program."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    for gate_set_id in {qp.voltage_sequence.gate_set.id for qp in qubit_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)

    exchange_amplitude = node.parameters.exchange_amplitude
    if exchange_amplitude is None:
        exchange_amplitude = _get_cz_exchange_amplitude(qubit_pairs[0])
        node.parameters.exchange_amplitude = exchange_amplitude

    exchange_duration = node.parameters.exchange_duration_in_ns
    if exchange_duration is None:
        exchange_duration = _get_cz_duration_ns(qubit_pairs[0])
        node.parameters.exchange_duration_in_ns = exchange_duration
    exchange_duration = int(exchange_duration)

    num_cycles = _get_num_cycle_array(node.parameters)
    initial_state_values = [0, 1]  # 0=|10> (control excited), 1=|01> (target excited)
    dd_axis_values = [0, 1]  # 0=X⊗X, 1=Y⊗X

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "initial_state": xr.DataArray(
            initial_state_values,
            attrs={
                "long_name": "initial odd-parity state",
                "units": "",
                "description": "0=|10> control excited, 1=|01> target excited",
            },
        ),
        "dd_axis": xr.DataArray(
            dd_axis_values,
            attrs={
                "long_name": "DD axis selector",
                "units": "",
                "description": "0=X⊗X, 1=Y⊗X",
            },
        ),
        "num_cphase_cycles": xr.DataArray(
            num_cycles,
            attrs={"long_name": "number of raw CPhase two-cycles", "units": ""},
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw_pairs(node)

        n = declare(int)
        n_st = declare_output_stream()

        cycle_count = declare(int)
        cycle_idx = declare(int)

        p2 = declare(int)
        p_st = {qp.name: declare_output_stream() for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                for initial_state_val in initial_state_values:
                    for dd_axis_val in dd_axis_values:
                        with for_(*from_array(cycle_count, num_cycles)):

                            reset_frame(
                                qubit_pair.qubit_target.xy.name,
                                qubit_pair.qubit_control.xy.name
                            )

                            qubit_pair.initialize()

                            align()

                            if initial_state_val == 0:
                                qubit_pair.qubit_control.x180()
                            else:
                                qubit_pair.qubit_target.x180()

                            align()

                            with for_(
                                cycle_idx, 0, cycle_idx < cycle_count, cycle_idx + 1
                            ):
                                # Exchange pulses (zero out phase shifts to prevent
                                # frame drift when re-running after node 18)
                                qubit_pair.cz(
                                    phase_shift_target=0,
                                    phase_shift_control=0,
                                )
                                _apply_dd_pair(qubit_pair, dd_axis_val)
                                qubit_pair.cz(
                                    phase_shift_target=0,
                                    phase_shift_control=0,
                                )
                                _apply_dd_pair(qubit_pair, dd_axis_val)

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

            n_initial_states = len(initial_state_values)
            n_dd_axes = len(dd_axis_values)
            n_cycle_counts = len(num_cycles)
            for qubit_pair in qubit_pairs:
                p_st[qubit_pair.name].buffer(
                    n_initial_states,
                    n_dd_axes,
                    n_cycle_counts,
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
        sweep_dims=("initial_state", "dd_axis", "num_cphase_cycles"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse residual iSWAP error amplification."""
    ds_fit, fit_results = fit_raw_data(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        analysis_signal=node.parameters.analysis_signal,
        max_theta_rad=node.parameters.max_theta_rad,
        min_fit_contrast=node.parameters.min_fit_contrast,
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
    """Plot residual iSWAP transfer traces and fits."""
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
def update_state(node: QualibrationNode[Parameters, Quam]):  # noqa: ARG001
    """Diagnostic node: no state update in v1."""
    return None


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

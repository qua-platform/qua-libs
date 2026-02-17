# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.single_qubit_randomized_benchmarking import (
    Parameters,
    build_single_qubit_clifford_tables,
    play_rb_gate,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.common_utils.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Node initialisation}
description = """
        SINGLE QUBIT RANDOMIZED BENCHMARKING (PPU-optimized)
The program plays random sequences of single-qubit Clifford gates and measures the
survival probability (return to ground state) afterward.  The 24 single-qubit
Cliffords are decomposed via Qiskit transpilation (basis: rx, ry, rz) into native
gates {x90, x180, -x90, y90, y180, -y90} plus virtual Z rotations
(frame_rotation_2pi, zero duration).

The PPU generates random Clifford circuits on-chip:
  1. PPU PHASE: For each circuit, random Cliffords are generated incrementally
     across depth checkpoints using preloaded composition and inverse tables.
  2. EXPERIMENT PHASE: For each (depth, shot), the pre-computed gate sequences
     are played back from arrays — no RNG or composition in the hot path.

Depth convention: depth d = d-1 random Cliffords + 1 recovery (inverse) = d total.
A single random circuit of max length is generated per circuit_idx; shorter depths
are truncations of the same circuit (standard RB truncation approach).

The survival probability vs circuit depth is fit to F(m) = A·α^m + B.  The average
error per Clifford is epc = (1 − α)·(d − 1)/d with d = 2, giving the average
Clifford gate fidelity F_avg = 1 − epc.

Prerequisites:
    - Having calibrated the sensor dots and resonators (nodes 2a, b, 3).
    - Having calibrated initialization, operation and PSB measurement points (nodes 4, 5).
    - Having calibrated π and π/2 pulse parameters (nodes 08a, 08b, 10a).
    - Native gate operations (x90, x180, -x90, y90, y180, -y90) defined on the qubit XY channel.

State update:
    - The averaged single qubit gate fidelity: qubit.gate_fidelity["averaged"].
"""


node = QualibrationNode[Parameters, Quam](
    name="14_single_qubit_randomized_benchmarking",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build Clifford lookup tables and generate the PPU-optimized QUA program.

    The program has a 3-phase structure per random circuit:

    Phases 1+2 (PPU computation — no real-time constraints):
        Generate random Cliffords incrementally across depth checkpoints.
        Store them in ``circuit_array[]`` and pre-compute the inverse at
        each checkpoint in ``inverses[]``.

    Phase 3 (Experiment — gate playback only):
        For each (depth, shot) pair, play pre-computed gates from arrays.
        Only array reads occur in the hot path.
    """
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    # Build Clifford tables (Python-side, once)
    node.log("Building single-qubit Clifford tables...")
    clifford_tables = build_single_qubit_clifford_tables()
    num_cliffords = clifford_tables["num_cliffords"]

    depths = node.parameters.get_depths()
    num_depths = len(depths)
    num_circuits = node.parameters.num_circuits_per_length
    num_shots = node.parameters.num_shots
    max_depth = int(max(depths))
    max_random = max_depth - 1  # d-1 random Cliffords for depth d

    node.log(
        f"RB config: {num_depths} depths (max {max_depth}), " f"{num_circuits} circuits, {num_shots} shots/circuit"
    )

    # Register sweep axes for xarray dataset
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "circuit": xr.DataArray(
            np.arange(num_circuits),
            attrs={"long_name": "circuit index"},
        ),
        "depth": xr.DataArray(
            depths,
            attrs={"long_name": "number of Cliffords"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        # ── QUA variables ─────────────────────────────────────────────────
        n = declare(int)
        n_st = declare_stream()
        depth_idx = declare(int)
        circuit_idx = declare(int)
        gate_idx = declare(int)
        cliff_idx = declare(int)

        # Lookup tables (read-only, precomputed in Python)
        depths_qua = declare(int, value=depths.tolist())
        clifford_compose_qua = declare(int, value=clifford_tables["compose"])
        clifford_inverse_qua = declare(int, value=clifford_tables["inverse"])
        clifford_decomp_qua = declare(int, value=clifford_tables["decomp_flat"])
        clifford_decomp_offsets_qua = declare(int, value=clifford_tables["decomp_offsets"])
        clifford_decomp_lengths_qua = declare(int, value=clifford_tables["decomp_lengths"])

        # Mutable arrays for pre-computed circuit data
        circuit_array = declare(int, size=max(max_random, 1))
        inverses = declare(int, size=num_depths)

        # PPU phase variables
        prev_random_count = declare(int)
        num_random = declare(int)
        i = declare(int)
        total_clifford = declare(int)
        rand_clifford = declare(int)
        inverse_clifford = declare(int)

        # Experiment phase variables
        current_gate = declare(int)
        decomp_offset = declare(int)
        decomp_length = declare(int)

        # Measurement variables (per qubit)
        state = declare(int, size=num_qubits)
        state_st = {qubit.name: declare_stream() for qubit in qubits}

        # RNG for on-PPU Clifford generation
        rng = Random(seed=node.parameters.seed)

        for batched_qubits in qubits.batch():
            # ═════════════════════════════════════════════════════════════
            # Outermost loop: circuits
            # ═════════════════════════════════════════════════════════════
            with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                save(circuit_idx, n_st)

                # ─────────────────────────────────────────────────────────
                # PHASES 1+2: PPU Computation (no real-time constraints)
                #
                # Build one random circuit of max length incrementally.
                # Depths are sorted, so each checkpoint extends the
                # previous one.  For depth d we need d-1 random Cliffords.
                #
                # total_clifford tracks the running composition C_0∘C_1∘…
                # At each checkpoint we snapshot its inverse.
                # ─────────────────────────────────────────────────────────
                assign(total_clifford, 0)  # identity
                assign(prev_random_count, 0)

                with for_(depth_idx, 0, depth_idx < num_depths, depth_idx + 1):
                    assign(num_random, depths_qua[depth_idx] - 1)

                    # Generate only the NEW Cliffords since last checkpoint
                    with for_(i, prev_random_count, i < num_random, i + 1):
                        assign(rand_clifford, rng.rand_int(num_cliffords))
                        assign(circuit_array[i], rand_clifford)
                        assign(
                            total_clifford,
                            clifford_compose_qua[rand_clifford * num_cliffords + total_clifford],
                        )

                    # Snapshot inverse at this depth checkpoint
                    assign(
                        inverses[depth_idx],
                        clifford_inverse_qua[total_clifford],
                    )
                    assign(prev_random_count, num_random)

                # ─────────────────────────────────────────────────────────
                # PHASE 3: Experiment (gate playback only)
                # ─────────────────────────────────────────────────────────
                with for_(depth_idx, 0, depth_idx < num_depths, depth_idx + 1):
                    assign(num_random, depths_qua[depth_idx] - 1)

                    with for_(n, 0, n < num_shots, n + 1):
                        # --- Reset frame ---
                        for i_q, qubit in batched_qubits.items():
                            reset_frame(qubit.xy.name)
                        align()

                        # --- Empty ---
                        for i_q, qubit in batched_qubits.items():
                            qubit.empty()
                        align()

                        # --- Initialize ---
                        for i_q, qubit in batched_qubits.items():
                            qubit.initialize(duration=node.parameters.gap_wait_time_in_ns)
                        align()

                        # --- Gate sequence: d-1 random Cliffords ---
                        with for_(
                            cliff_idx,
                            0,
                            cliff_idx < num_random,
                            cliff_idx + 1,
                        ):
                            assign(rand_clifford, circuit_array[cliff_idx])
                            assign(
                                decomp_offset,
                                clifford_decomp_offsets_qua[rand_clifford],
                            )
                            assign(
                                decomp_length,
                                clifford_decomp_lengths_qua[rand_clifford],
                            )
                            with for_(
                                gate_idx,
                                0,
                                gate_idx < decomp_length,
                                gate_idx + 1,
                            ):
                                assign(
                                    current_gate,
                                    clifford_decomp_qua[decomp_offset + gate_idx],
                                )
                                for i_q, qubit in batched_qubits.items():
                                    play_rb_gate(qubit, current_gate)

                        # --- Inverse: play recovery Clifford ---
                        assign(inverse_clifford, inverses[depth_idx])
                        assign(
                            decomp_offset,
                            clifford_decomp_offsets_qua[inverse_clifford],
                        )
                        assign(
                            decomp_length,
                            clifford_decomp_lengths_qua[inverse_clifford],
                        )
                        with for_(
                            gate_idx,
                            0,
                            gate_idx < decomp_length,
                            gate_idx + 1,
                        ):
                            assign(
                                current_gate,
                                clifford_decomp_qua[decomp_offset + gate_idx],
                            )
                            for i_q, qubit in batched_qubits.items():
                                play_rb_gate(qubit, current_gate)

                        # --- Measure ---
                        align()
                        for i_q, qubit in batched_qubits.items():
                            assign(
                                state[i_q],
                                Cast.to_int(qubit.measure()),
                            )
                            save(state[i_q], state_st[qubit.name])

                        # --- Compensation ---
                        # TODO: Compensation pulse won't work until we set up
                        # the mixin to track durations of macro operations for
                        # the voltage pulse compensation.
                        align()
                        for i_q, qubit in batched_qubits.items():
                            qubit.voltage_sequence.apply_compensation_pulse()

        # ── Stream processing ─────────────────────────────────────────
        # Buffer order matches loop nesting: circuit → depth → shot
        with stream_processing():
            n_st.save("n")
            for qubit in qubits:
                (
                    state_st[qubit.name]
                    .buffer(num_shots)
                    .map(FUNCTIONS.average())
                    .buffer(num_depths)
                    .buffer(num_circuits)
                    .save(f"state_{qubit.name}")
                )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_circuits_per_length,
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
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit the RB exponential decay for each qubit."""
    ds_raw = node.results["ds_raw"]
    qubits = node.namespace["qubits"]

    fit_results = fit_raw_data(ds_raw, qubits)
    node.results["fit_results"] = fit_results

    log_fitted_results(fit_results, node.log)

    node.outcomes = {qname: ("successful" if r["success"] else "failed") for qname, r in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted RB data."""
    ds_raw = node.results["ds_raw"]
    fit_results = node.results["fit_results"]
    qubits = node.namespace["qubits"]

    fig = plot_raw_data_with_fit(ds_raw, fit_results, qubits)
    node.results["figure"] = fig


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            q.gate_fidelity["averaged"] = float(node.results["fit_results"][q.name]["gate_fidelity"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.single_qubit_randomized_benchmarking import (
    build_single_qubit_clifford_tables,
    avg_physical_gates_per_clifford,
    decomposition_type,
    play_rb_gate,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.cmaes.pair_rb_parameters import PairRBParameters
from calibration_utils.common_utils.experiment import (
    get_qubit_pairs,
    progress_counter_with_log,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Node initialisation}
description = """
        PAIR-TARGETED SINGLE QUBIT RANDOMIZED BENCHMARKING (PPU-optimized)
Qubit-pair analogue of node 14_single_qubit_randomized_benchmarking.  Instead
of targeting individual qubits, this node targets ``qubit_pairs`` and runs
single-qubit RB on both members of each pair (qubit_target then qubit_control)
in turn, measuring each one's average Clifford gate fidelity.

It exists so the CMA-ES ORBIT gate-optimisation graph can verify, on the same
target convention as the ORBIT nodes (``qubit_pairs``), the single-qubit gate
fidelity reached for each qubit in a pair.  The QUA program, Clifford tables,
fit, and plotting are reused unchanged from the single-qubit RB node — the
only difference is how the qubit list is resolved (flattened from the pairs).

The survival probability vs circuit depth is fit to F(m) = A·α^m + B.  The
average error per Clifford is epc = (1 − α)·(d − 1)/d with d = 2, giving the
average Clifford gate fidelity F_avg = 1 − epc.

Prerequisites:
    - Having calibrated the sensor dots and resonators (nodes 2a, b, 3).
    - Having calibrated initialization, operation and PSB measurement points (nodes 4, 5).
    - Having calibrated π and π/2 pulse parameters (nodes 09a, 09b, 11a).
    - Native gate operations (x90, x180, -x90, y90, y180, -y90) defined on the qubit XY channel.

State update:
    - The averaged single qubit gate fidelity: qubit.gate_fidelity["averaged"].
"""


node = QualibrationNode[PairRBParameters, Quam](
    name="03c_pair_single_qubit_rb",
    description=description,
    parameters=PairRBParameters(),
)


# Any parameters that should change for debugging purposes only should go in here.
# These parameters are ignored when run through the GUI or as part of a graph.
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[PairRBParameters, Quam]):
    # node.parameters.qubit_pairs = ["q1_q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


def _qubits_from_pairs(node: QualibrationNode[PairRBParameters, Quam]):
    """Resolve the unique member qubits of the targeted pairs (target then control)."""
    qubit_pairs = get_qubit_pairs(node)
    node.namespace["qubit_pairs"] = qubit_pairs
    qubits = []
    seen = set()
    for qp in qubit_pairs:
        for qubit in (qp.qubit_target, qp.qubit_control):
            if qubit.name not in seen:
                seen.add(qubit.name)
                qubits.append(qubit)
    return qubits


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[PairRBParameters, Quam]):
    """Build Clifford lookup tables and generate the PPU-optimized QUA program.

    Identical 3-phase (PPU generate → snapshot inverses → playback) structure
    as the single-qubit RB node; the only change is that the qubit list is
    flattened from the targeted qubit pairs.
    """
    node.namespace["qubits"] = qubits = _qubits_from_pairs(node)
    qubit_names = [qubit.name for qubit in qubits]

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
        f"Pair RB on qubits {qubit_names}: {num_depths} depths (max {max_depth}), "
        f"{num_circuits} circuits, {num_shots} shots/circuit"
    )

    # Register sweep axes for xarray dataset
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubit_names),
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
        n_st = declare_output_stream()
        depth_idx = declare(int)
        circuit_idx = declare(int)
        gate_idx = declare(int)
        cliff_idx = declare(int)

        # Lookup tables (read-only, precomputed in Python)
        depths_qua = declare(int, value=depths.tolist())
        clifford_compose_qua = declare(int, value=clifford_tables["compose"])
        clifford_inverse_qua = declare(int, value=clifford_tables["inverse"])
        clifford_decomp_qua = declare(int, value=clifford_tables["decomp_flat"])
        clifford_decomp_offsets_qua = declare(
            int, value=clifford_tables["decomp_offsets"]
        )
        clifford_decomp_lengths_qua = declare(
            int, value=clifford_tables["decomp_lengths"]
        )

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
        state = declare(int)
        state_st = {qubit.name: declare_output_stream() for qubit in qubits}

        # RNG for on-PPU Clifford generation
        rng = Random(seed=node.parameters.seed)

        for qubit in qubits:
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
                            clifford_compose_qua[
                                rand_clifford * num_cliffords + total_clifford
                            ],
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
                        reset_frame(qubit.xy.name)
                        align()

                        # --- Initialize ---
                        qubit.initialize(
                            qubit_name=qubit.name,
                            conditional_drive=True,
                        )
                        align()

                        # --- Gate sequence: d-1 random Cliffords ---
                        with for_(cliff_idx, 0, cliff_idx < num_random, cliff_idx + 1):
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
                                gate_idx, 0, gate_idx < decomp_length, gate_idx + 1
                            ):
                                assign(
                                    current_gate,
                                    clifford_decomp_qua[decomp_offset + gate_idx],
                                )
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
                        with for_(gate_idx, 0, gate_idx < decomp_length, gate_idx + 1):
                            assign(
                                current_gate,
                                clifford_decomp_qua[decomp_offset + gate_idx],
                            )
                            play_rb_gate(qubit, current_gate)

                        # --- Measure ---
                        align()
                        p = qubit.measure(return_iq=False)
                        align()

                        # --- Compensation ---
                        qubit.voltage_sequence.ramp_to_zero()
                        align()

                        assign(state, Cast.to_int(p))
                        save(state, state_st[qubit.name])

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
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[PairRBParameters, Quam]):
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
def execute_qua_program(node: QualibrationNode[PairRBParameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch raw data."""
    qmm = node.machine.connect(timeout=node.parameters.timeout)
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_circuits_per_length,
                start_time=data_fetcher.t_start,
                node=node,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[PairRBParameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = _qubits_from_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[PairRBParameters, Quam]):
    """Fit the RB exponential decay for each qubit in the pairs."""
    ds_raw = node.results["ds_raw"]
    qubits = node.namespace["qubits"]

    fit_results = fit_raw_data(
        ds_raw, qubits, avg_physical_gates_per_clifford(decomposition_type)
    )
    node.results["fit_results"] = fit_results

    log_fitted_results(fit_results, node.log)

    node.outcomes = {
        qname: ("successful" if r["success"] else "failed")
        for qname, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[PairRBParameters, Quam]):
    """Plot the raw and fitted RB data."""
    ds_raw = node.results["ds_raw"]
    fit_results = node.results["fit_results"]
    qubits = node.namespace["qubits"]

    fig = plot_raw_data_with_fit(ds_raw, fit_results, qubits)
    node.results["figure"] = fig
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[PairRBParameters, Quam]):
    """Update the averaged gate fidelity for each successfully fitted qubit."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            q.gate_fidelity["averaged"] = float(
                node.results["fit_results"][q.name]["native_gate_fidelity"]
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[PairRBParameters, Quam]):
    node.save()
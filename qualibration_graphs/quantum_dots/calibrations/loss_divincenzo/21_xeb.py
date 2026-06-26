# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.xeb import (
    Parameters,
    NUM_XEB_GATES,
    play_xeb_gate,
    calc_ideal_probs_1q,
    calc_ideal_probs_2q,
    calc_linear_xeb_fidelity,
    calc_log_xeb_fidelity,
    calc_purity,
    log_xeb_results,
    plot_xeb_fidelity,
    plot_state_heatmap,
    plot_purity,
)
from calibration_utils.common_utils.experiment import get_qubits, get_qubit_pairs, enable_dual_drive_mw
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Node initialisation}
description = """
        CROSS-ENTROPY BENCHMARKING (XEB)
Measures gate fidelity by running random circuits and comparing the measured
output distribution to classically computed ideal probabilities.

Random single-qubit gates are sampled from a 3-gate set per cycle:
  - "sw": {SX (x90), SY (y90), SW (z_neg90 + x90)} — matches Google XEB convention
  - "t":  {SX (x90), SY (y90), T (virtual Z(π/4))}

No consecutive same gate is allowed on a given qubit.

Modes:
  - 1Q XEB (apply_two_qubit_gate=False): benchmarks single-qubit gate layer fidelity
  - 2Q XEB (apply_two_qubit_gate=True): interleaves a CZ gate between random 1Q layers
    to benchmark two-qubit gate fidelity

Analysis computes both linear and log XEB fidelities vs circuit depth, fits
F(d) = a · r^d to extract the layer fidelity r, and optionally estimates the
actual 2Q gate unitary via Nelder-Mead optimization.

Prerequisites:
    - Calibrated sensor dots and resonators (nodes 2a, b, 3).
    - Calibrated initialization, operation and PSB measurement points (nodes 4, 5).
    - Calibrated π and π/2 pulse parameters (nodes 09a, 09b, 11a).
    - Native gate operations (x90, y90, z_neg90) defined on the qubit XY channel.
    - (2Q mode) Calibrated CZ gate (node 16 or 16a).

State update:
    - 1Q mode: qubit.gate_fidelity["XEB"] = layer_fidelity
    - 2Q mode: qubit_pair.macros[cz_macro_name].fidelity["XEB"] = layer_fidelity
"""


node = QualibrationNode[Parameters, Quam](
    name="21_xeb",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]
    # node.parameters.apply_two_qubit_gate = False
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build sweep axes and generate the QUA program for XEB.

    The program has a 2-phase structure per random circuit:

    Phase 1 (PPU — no real-time constraints):
        Generate random gate indices for all qubits across all depths.
        No consecutive same gate on a given qubit.
        Gate indices saved to streams for classical analysis.

    Phase 2 (Experiment — gate playback + measurement):
        For each (depth, shot), play pre-generated gates and measure.
    """
    p = node.parameters
    depths = p.get_depths()
    n_depths = len(depths)
    max_depth = int(depths[-1])
    gate_set = p.gate_set

    if p.apply_two_qubit_gate:
        node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
        target_pair = qubit_pairs[0]
        qubits_list = [target_pair.qubit_control, target_pair.qubit_target]
        node.namespace["qubits"] = qubits_list
        num_qubits = 2
    else:
        node.namespace["qubits"] = qubits_list = list(get_qubits(node))
        num_qubits = len(qubits_list)

    node.namespace["depths"] = depths
    node.namespace["num_qubits_per_run"] = num_qubits

    sweep_axes = {}
    if p.apply_two_qubit_gate:
        sweep_axes["qubit_pair"] = xr.DataArray([target_pair.name])
    else:
        sweep_axes["qubit"] = xr.DataArray([q.name for q in qubits_list])
    sweep_axes["sequence"] = xr.DataArray(
        np.arange(p.n_sequences), attrs={"long_name": "sequence index"}
    )
    sweep_axes["depth"] = xr.DataArray(
        depths, attrs={"long_name": "circuit depth"}
    )
    node.namespace["sweep_axes"] = sweep_axes

    node.log(
        f"XEB config: {n_depths} depths (max {max_depth}), "
        f"{p.n_sequences} sequences, {p.n_shots} shots/circuit, "
        f"gate_set={gate_set!r}, 2Q={p.apply_two_qubit_gate}"
    )

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw(node)

        s = declare(int)
        depth_idx = declare(int)
        depth_val = declare(int)
        n = declare(int)
        d = declare(int)

        depths_qua = declare(int, value=depths.tolist())
        s_st = declare_output_stream()

        gate = [declare(int, size=max_depth) for _ in range(num_qubits)]
        gate_st = [declare_output_stream() for _ in range(num_qubits)]
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_output_stream() for _ in range(num_qubits)]

        rng = Random(seed=p.seed)

        with for_(s, 0, s < p.n_sequences, s + 1):
            save(s, s_st)

            # ── Phase 1: PPU random gate generation ───────────────
            for q in range(num_qubits):
                assign(gate[q][0], rng.rand_int(NUM_XEB_GATES))
                save(gate[q][0], gate_st[q])

            with for_(d, 1, d < max_depth, d + 1):
                for q in range(num_qubits):
                    assign(gate[q][d], rng.rand_int(NUM_XEB_GATES))
                    with while_(gate[q][d] == gate[q][d - 1]):
                        assign(gate[q][d], rng.rand_int(NUM_XEB_GATES))
                    save(gate[q][d], gate_st[q])

            # ── Phase 2: experiment (playback + measure) ──────────
            with for_(depth_idx, 0, depth_idx < n_depths, depth_idx + 1):
                assign(depth_val, depths_qua[depth_idx])

                with for_(n, 0, n < p.n_shots, n + 1):
                    for qubit in qubits_list:
                        reset_frame(qubit.xy.name)
                    align()

                    for qubit in qubits_list:
                        qubit.empty()

                    for qubit in qubits_list:
                        qubit.initialize()
                    align()

                    with for_(d, 0, d < depth_val, d + 1):
                        for q_idx, qubit in enumerate(qubits_list):
                            play_xeb_gate(qubit, gate[q_idx][d], gate_set=gate_set)

                        if p.apply_two_qubit_gate:
                            align()
                            target_pair.macros[p.cz_macro_name].apply()
                            align()

                    align()

                    for q_idx, qubit in enumerate(qubits_list):
                        p_meas = qubit.measure()
                        assign(state[q_idx], Cast.to_int(p_meas))
                        save(state[q_idx], state_st[q_idx])

                    align()

                    for qubit in qubits_list:
                        qubit.voltage_sequence.ramp_to_zero()
                    align()

        with stream_processing():
            s_st.save("n")
            for q_idx in range(num_qubits):
                gate_st[q_idx].buffer(max_depth).save_all(
                    f"gate_indices_{q_idx}"
                )
                if p.apply_two_qubit_gate:
                    (
                        state_st[q_idx]
                        .buffer(p.n_shots)
                        .buffer(n_depths)
                        .buffer(p.n_sequences)
                        .save(f"state_{q_idx}")
                    )
                else:
                    qubit_name = qubits_list[q_idx].name
                    (
                        state_st[q_idx]
                        .buffer(p.n_shots)
                        .map(FUNCTIONS.average())
                        .buffer(n_depths)
                        .buffer(p.n_sequences)
                        .save(f"state_{qubit_name}")
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
    p = node.parameters
    num_qubits = node.namespace["num_qubits_per_run"]
    with qm_session(qmm, config, timeout=p.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])

        if p.apply_two_qubit_gate:
            # 2Q: per-shot state data has an extra n_shots dimension and
            # stream names (state_0/1) don't match the qubit_pair entity
            # axis, so XarrayDataFetcher cannot handle this mode.
            job.result_handles.wait_for_all_values()
            dataset = xr.Dataset()
            for q_idx in range(num_qubits):
                dataset[f"state_{q_idx}"] = xr.DataArray(
                    job.result_handles.get(f"state_{q_idx}").fetch_all()
                )
                dataset[f"gate_indices_{q_idx}"] = xr.DataArray(
                    job.result_handles.get(f"gate_indices_{q_idx}").fetch_all()
                )
        else:
            # 1Q: averaged state data is compatible with XarrayDataFetcher
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    p.n_sequences,
                    start_time=data_fetcher.t_start,
                )
            for q_idx in range(num_qubits):
                key = f"gate_indices_{q_idx}"
                if key not in dataset:
                    handle = job.result_handles.get(key)
                    handle.wait_for_all_values()
                    dataset[key] = xr.DataArray(handle.fetch_all())

        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    if node.parameters.apply_two_qubit_gate:
        node.namespace["qubit_pairs"] = get_qubit_pairs(node)
        target_pair = node.namespace["qubit_pairs"][0]
        node.namespace["qubits"] = [
            target_pair.qubit_control, target_pair.qubit_target
        ]
    else:
        node.namespace["qubits"] = list(get_qubits(node))


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Compute ideal probabilities, XEB fidelities, and fit decay curves."""
    p = node.parameters
    ds_raw = node.results["ds_raw"]
    depths = p.get_depths()
    qubits_list = node.namespace["qubits"]
    num_qubits = node.namespace.get("num_qubits_per_run", len(qubits_list))

    gate_indices_list = []
    for q_idx in range(num_qubits):
        arr = np.array(ds_raw[f"gate_indices_{q_idx}"].values, dtype=int)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        gate_indices_list.append(arr)
    gate_indices = np.stack(gate_indices_list, axis=1)

    node.results["gate_indices"] = gate_indices
    node.results["fit_results"] = {}
    node.outcomes = {}

    if p.apply_two_qubit_gate:
        # ── 2Q XEB ────────────────────────────────────────────────
        state_0 = ds_raw["state_0"].values  # (n_seq, n_depths, n_shots)
        state_1 = ds_raw["state_1"].values
        n_seq, n_dep, n_shots = state_0.shape

        measured_probs = np.zeros((n_seq, n_dep, 4))
        for s_idx in range(n_seq):
            for d_idx in range(n_dep):
                s0 = state_0[s_idx, d_idx]
                s1 = state_1[s_idx, d_idx]
                for shot in range(n_shots):
                    joint = int(s0[shot]) * 2 + int(s1[shot])
                    measured_probs[s_idx, d_idx, joint] += 1
                measured_probs[s_idx, d_idx] /= n_shots

        ideal_probs = calc_ideal_probs_2q(gate_indices, depths, p.gate_set)
        dim = 4

        linear_fid = calc_linear_xeb_fidelity(measured_probs, ideal_probs, dim=dim)
        log_fid = calc_log_xeb_fidelity(measured_probs, ideal_probs, dim=dim)
        purity = calc_purity(measured_probs, dim=dim)

        target_pair = node.namespace["qubit_pairs"][0]
        results = log_xeb_results(depths, linear_fid, log_fid, purity, label=target_pair.name)
        node.results["fit_results"][target_pair.name] = results
        node.results["measured_probs"] = measured_probs

        is_success = not np.isnan(results["linear_fit"]["r"])
        node.outcomes[target_pair.name] = "successful" if is_success else "failed"

        node.log(
            f"  {target_pair.name}: linear r={results['linear_fit']['r']:.5f}, "
            f"log r={results['log_fit']['r']:.5f}"
        )

        if p.estimate_2q_unitary and is_success:
            from calibration_utils.xeb import estimate_2q_unitary

            node.log("Running 2Q unitary estimation...")
            est = estimate_2q_unitary(gate_indices, depths, measured_probs, p.gate_set)
            node.results["unitary_estimation"] = est
            node.log(
                f"  Estimated: θ_iSWAP={est['theta_iswap']:.4f}, "
                f"φ_CPhase={est['phi_cphase']:.4f}, "
                f"fidelity={est['fidelity']:.4f}"
            )
    else:
        # ── 1Q XEB (per qubit) ────────────────────────────────────
        for q_idx, qubit in enumerate(qubits_list):
            gi_1q = gate_indices[:, q_idx, :]  # (n_seq, max_depth)
            state_avg = ds_raw[f"state_{qubit.name}"].values  # (n_seq, n_depths)

            measured_probs_1q = np.stack(
                [1 - state_avg, state_avg], axis=-1
            )  # (n_seq, n_depths, 2)

            ideal_probs_1q = calc_ideal_probs_1q(gi_1q, depths, p.gate_set)

            linear_fid = calc_linear_xeb_fidelity(measured_probs_1q, ideal_probs_1q, dim=2)
            log_fid = calc_log_xeb_fidelity(measured_probs_1q, ideal_probs_1q, dim=2)
            purity = calc_purity(measured_probs_1q, dim=2)

            results = log_xeb_results(depths, linear_fid, log_fid, purity, label=qubit.name)
            node.results["fit_results"][qubit.name] = results

            is_success = not np.isnan(results["linear_fit"]["r"])
            node.outcomes[qubit.name] = "successful" if is_success else "failed"

            node.log(
                f"  {qubit.name}: linear r={results['linear_fit']['r']:.5f}, "
                f"log r={results['log_fit']['r']:.5f}"
            )


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot XEB fidelity, state heatmaps, and purity."""
    fit_results = node.results["fit_results"]

    for target_name, results in fit_results.items():
        fig_fid = plot_xeb_fidelity(
            results["depths"],
            results["linear_fidelity_mean"],
            results["log_fidelity_mean"],
            results["linear_fit"],
            results["log_fit"],
            label=target_name,
        )
        node.results[f"figure_fidelity_{target_name}"] = fig_fid

        fig_pur = plot_purity(
            results["depths"],
            results["purity_mean"],
            results["purity_fit"],
            label=target_name,
        )
        node.results[f"figure_purity_{target_name}"] = fig_pur

    if "measured_probs" in node.results:
        depths = node.parameters.get_depths()
        fig_hm = plot_state_heatmap(
            depths,
            node.results["measured_probs"],
            label="2Q XEB",
        )
        node.results["figure_heatmap"] = fig_hm

    if "unitary_estimation" in node.results:
        from calibration_utils.xeb import plot_unitary_estimation

        fig_est = plot_unitary_estimation(
            node.results["unitary_estimation"],
            label=node.namespace["qubit_pairs"][0].name,
        )
        node.results["figure_unitary_estimation"] = fig_est

    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update qubit/pair state with XEB fidelity results."""
    p = node.parameters
    with node.record_state_updates():
        if p.apply_two_qubit_gate:
            for qp in node.namespace["qubit_pairs"]:
                if node.outcomes.get(qp.name) == "failed":
                    continue
                results = node.results["fit_results"][qp.name]
                qp.macros[p.cz_macro_name].fidelity["XEB"] = float(
                    results["linear_fit"]["r"]
                )
                if "unitary_estimation" in node.results:
                    est = node.results["unitary_estimation"]
                    macro_fid = qp.macros[p.cz_macro_name].fidelity
                    macro_fid["XEB_theta_iswap"] = float(est["theta_iswap"])
                    macro_fid["XEB_phi_cphase"] = float(est["phi_cphase"])
                    macro_fid["XEB_phi_rz1"] = float(est["phi_rz1"])
                    macro_fid["XEB_phi_rz2"] = float(est["phi_rz2"])
        else:
            for qubit in node.namespace["qubits"]:
                if node.outcomes.get(qubit.name) == "failed":
                    continue
                results = node.results["fit_results"][qubit.name]
                qubit.gate_fidelity["XEB"] = float(results["linear_fit"]["r"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

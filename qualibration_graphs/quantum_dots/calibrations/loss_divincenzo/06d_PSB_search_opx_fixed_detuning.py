# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log, enable_dual_drive_mw

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome

from quam_config import Quam

from calibration_utils.iq_blobs import fit_raw_data, log_fitted_results
from calibration_utils.psb_search_fixed_detuning import (
    Parameters,
    build_labeled_dataset,
    fit_gmm_labeled,
    plot_labeled_histogram_barthel,
    plot_labeled_histogram_gmm,
    resolve_qubits_and_dot_pairs,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot


# %% {Node initialisation}
description = """
        PAULI SPIN BLOCKADE SEARCH - Fixed Configuration, Labeled Two-State Readout
Acquires labeled shot-by-shot IQ data at the configured measure point for each qubit by measuring
twice per shot: once without a pi pulse (loading ``init_state_label``) and once with a pi pulse
(loading the complementary state). The two streams are used to fit either the Barthel 1D readout
model or a 2-component Gaussian mixture model.

Qubit pairs are resolved automatically from ``qubit.preferred_readout_quantum_dot``; no explicit
``qubit_pairs`` parameter is required.

Prerequisites:
    - Initialized Quam, calibrated sensor resonators, x180 pulse calibrated on each qubit.
    - Empty / initialize / measure voltages defined; optional ``detuning`` override for this run.

Node parameters:
    init_state_label : 'decay' | 'no_decay'
        Which physical state is prepared WITHOUT the pi pulse.
        'decay'    → no pi  = T (triplet, decays during measurement); pi = S (singlet)
        'no_decay' → no pi  = S (singlet);                            pi = T (triplet)
    analysis_model : 'barthel' | 'gmm'
        'barthel' – physics-based Barthel 1D model (MCMC, recommended).
        'gmm'     – 2-component Gaussian mixture model via PCA + sklearn.

State update:
    Reverts any temporary detuning override, then (if the fit succeeded) updates the
    integration-weights angle and discrimination threshold on the sensor dot.
"""


node = QualibrationNode[Parameters, Quam](
    name="06d_PSB_search_opx_fixed_detuning",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubits = ["q1", "q2"]
    node.parameters.simulate = False
    node.parameters.simulation_duration_ns = 60_000
    node.parameters.num_shots = 1000
    node.parameters.init_state_label = "decay"
    node.parameters.analysis_model = "barthel"


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Fixed-point PSB IQ acquisition with two labeled arms (no-pi and pi-pulse)."""
    qubits, qubit_dot_pairs = resolve_qubits_and_dot_pairs(node)

    dot_pairs = [dp for _, dp in qubit_dot_pairs]
    for gate_set_id in {dp.voltage_sequence.gate_set.id for dp in dot_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)
    for dp in dot_pairs:
        if len(dp.sensor_dots) != 1:
            raise ValueError(
                f"06d expects exactly one sensor dot per pair; {dp.id!r} has {len(dp.sensor_dots)}."
            )

    node.namespace["qubits"] = qubits
    node.namespace["qubit_dot_pairs"] = qubit_dot_pairs

    node.namespace["tracked_original_detunings"] = {}
    for dp in dot_pairs:
        if node.parameters.detuning is not None:
            gate_set = dp.voltage_sequence.gate_set
            point_name = dp._create_point_name("measure")
            point = gate_set.get_macros()[point_name]
            node.namespace["tracked_original_detunings"][dp.name] = point.voltages.get(dp.name)
            point.voltages[dp.name] = node.parameters.detuning

    node.namespace["sweep_axes"] = {
        "n_runs": xr.DataArray(
            np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw(node)

        n = declare(int)
        n_st = declare_output_stream()

        I_no_pi_st = {q.name: declare_output_stream() for q in qubits}
        Q_no_pi_st = {q.name: declare_output_stream() for q in qubits}
        I_pi_st    = {q.name: declare_output_stream() for q in qubits}
        Q_pi_st    = {q.name: declare_output_stream() for q in qubits}
        heralded_and_return_n_loops = getattr(node.parameters, "return_n_loops", False)
        n_loops_st = (
            {q.name: declare_output_stream() for q in qubits}
            if heralded_and_return_n_loops
            else {}
        )

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit, dot_pair in qubit_dot_pairs:
                # --- Arm 1: no pi pulse ---
                n_init = dot_pair.initialize(
                    qubit_name=qubit.name,
                    conditional_drive=True,
                )
                if heralded_and_return_n_loops:
                    save(n_init, n_loops_st[qubit.name])
                align()
                (i_no_pi, q_no_pi, _) = dot_pair.measure(return_iq=True)
                save(i_no_pi, I_no_pi_st[qubit.name])
                save(q_no_pi, Q_no_pi_st[qubit.name])
                align()
                dot_pair.voltage_sequence.ramp_to_zero()

                # --- Arm 2: pi pulse ---
                dot_pair.initialize(
                    qubit_name=qubit.name,
                    conditional_drive=True,
                )
                align()
                qubit.x180()
                align()
                (i_pi, q_pi, _) = dot_pair.measure(return_iq=True)
                save(i_pi, I_pi_st[qubit.name])
                save(q_pi, Q_pi_st[qubit.name])
                align()
                dot_pair.voltage_sequence.ramp_to_zero()

        with stream_processing():
            n_st.save("n")
            for q in qubits:
                I_no_pi_st[q.name].buffer(node.parameters.num_shots).save(f"I_no_pi_{q.name}")
                Q_no_pi_st[q.name].buffer(node.parameters.num_shots).save(f"Q_no_pi_{q.name}")
                I_pi_st[q.name].buffer(node.parameters.num_shots).save(f"I_pi_{q.name}")
                Q_pi_st[q.name].buffer(node.parameters.num_shots).save(f"Q_pi_{q.name}")
                if heralded_and_return_n_loops:
                    n_loops_st[q.name].buffer(node.parameters.num_shots).average().save(
                        f"n_loops_{q.name}"
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
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute QUA and assemble ``ds_raw`` with ``I_no_pi``, ``Q_no_pi``, ``I_pi``, ``Q_pi``."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    qubits = node.namespace["qubits"]
    qnames = [q.name for q in qubits]

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

    def _concat(prefix):
        arrays = [dataset[f"{prefix}_{n}"] for n in qnames]
        return xr.concat(arrays, dim="qubit").assign_coords(qubit=qnames)

    node.results["ds_raw"] = xr.Dataset({
        "I_no_pi": _concat("I_no_pi"),
        "Q_no_pi": _concat("Q_no_pi"),
        "I_pi":    _concat("I_pi"),
        "Q_pi":    _concat("Q_pi"),
    })


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    qubits, qubit_dot_pairs = resolve_qubits_and_dot_pairs(node)
    node.namespace["qubits"] = qubits
    node.namespace["qubit_dot_pairs"] = qubit_dot_pairs


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit labeled IQ shots with the selected model (Barthel or GMM)."""
    qubits = node.namespace["qubits"]
    ds_labeled = build_labeled_dataset(node.results["ds_raw"], node.parameters.init_state_label)

    if node.parameters.analysis_model == "barthel":
        ds_fit, fit_results = fit_raw_data(ds_labeled, node)
        node.results["ds_fit"] = ds_fit
        # fit_raw_data reports confusion-matrix fidelity; replace with the analytic
        # model optimum stored in ds_fit.fidelity_opt (same as fit_barthel_mixed_iq).
        for q in qubits:
            fit_results[q.name].readout_fidelity = 100.0 * float(
                ds_fit["fidelity_opt"].sel(qubit=q.name)
            )
    else:  # "gmm"
        fit_results, ds_gmm_fit = fit_gmm_labeled(ds_labeled, qubits)
        node.results["ds_gmm_fit"] = ds_gmm_fit

    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qname: (Outcome.SUCCESSFUL if fr["success"] else Outcome.FAILED)
        for qname, fr in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Raw + rotated IQ scatter (matching the state-update angle & threshold) + model histogram."""
    qubits = node.namespace["qubits"]
    ds_labeled = build_labeled_dataset(node.results["ds_raw"], node.parameters.init_state_label)
    fit_results = node.results.get("fit_results", {})
    figures = {}

    # --- IQ scatter: raw (left) + rotated with state-update threshold (right) ---
    n_qubits = len(list(qubits))
    fig_iq, axes = plt.subplots(n_qubits, 2, figsize=(11, 4.5 * n_qubits), squeeze=False)
    for idx, q in enumerate(qubits):
        ax_raw, ax_rot = axes[idx, 0], axes[idx, 1]
        Ig = ds_labeled["Ig"].sel(qubit=q.name).values
        Qg = ds_labeled["Qg"].sel(qubit=q.name).values
        Ie = ds_labeled["Ie"].sel(qubit=q.name).values
        Qe = ds_labeled["Qe"].sel(qubit=q.name).values
        fr = fit_results.get(q.name, {})
        iw_angle = fr.get("iw_angle", 0.0)
        I_thr = fr.get("I_threshold")

        ax_raw.plot(Ig * 1e3, Qg * 1e3, ".", alpha=0.4, markersize=2, label="S", color="C0")
        ax_raw.plot(Ie * 1e3, Qe * 1e3, ".", alpha=0.4, markersize=2, label="T", color="C1")
        ax_raw.set_xlabel("I [mV]")
        ax_raw.set_ylabel("Q [mV]")
        ax_raw.set_title(f"{q.name} — raw IQ")
        ax_raw.legend(fontsize=7)

        ca, sa = np.cos(iw_angle), np.sin(iw_angle)
        Ig_r, Qg_r = Ig * ca + Qg * sa, -Ig * sa + Qg * ca
        Ie_r, Qe_r = Ie * ca + Qe * sa, -Ie * sa + Qe * ca
        ax_rot.plot(Ig_r * 1e3, Qg_r * 1e3, ".", alpha=0.4, markersize=2, label="S", color="C0")
        ax_rot.plot(Ie_r * 1e3, Qe_r * 1e3, ".", alpha=0.4, markersize=2, label="T", color="C1")
        if I_thr is not None and np.isfinite(I_thr):
            ax_rot.axvline(
                I_thr * 1e3, color="C3", ls="--", lw=1.5,
                label=f"I_threshold = {I_thr * 1e3:.2f} mV",
            )
        ax_rot.set_xlabel("I_rot [mV]")
        ax_rot.set_ylabel("Q_rot [mV]")
        ax_rot.set_title(f"{q.name} — rotated (\u0394angle = {np.degrees(iw_angle):.1f}\u00b0)")
        ax_rot.legend(fontsize=7)

    fig_iq.suptitle("PSB IQ blobs — raw + rotated (state-update angle & threshold)")
    fig_iq.tight_layout()
    figures["iq_blobs"] = fig_iq

    # --- Model-specific histogram ---
    if node.parameters.analysis_model == "barthel" and "ds_fit" in node.results:
        fig_hist = plot_labeled_histogram_barthel(ds_labeled, node.results["ds_fit"], list(qubits))
        figures["histogram"] = fig_hist
    elif "ds_gmm_fit" in node.results:
        fig_hist = plot_labeled_histogram_gmm(node.results["ds_gmm_fit"], list(qubits))
        figures["histogram"] = fig_hist

    plt.show()
    node.results["figures"] = figures
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Revert detuning override; persist readout angle and threshold on the sensor dot."""
    for _, dot_pair in node.namespace["qubit_dot_pairs"]:
        if dot_pair.name in node.namespace.get("tracked_original_detunings", {}):
            gate_set = dot_pair.voltage_sequence.gate_set
            point_name = dot_pair._create_point_name("measure")
            point = gate_set.get_macros()[point_name]
            point.voltages[dot_pair.name] = node.namespace["tracked_original_detunings"][dot_pair.name]

    with node.record_state_updates():
        for qubit, dot_pair in node.namespace["qubit_dot_pairs"]:
            fit_result = node.results["fit_results"].get(qubit.name)
            if fit_result is None or not fit_result["success"]:
                continue

            sensor_dot = dot_pair.sensor_dots[0]
            op_name = "readout" + f"_{dot_pair.name}"
            operation = sensor_dot.readout_resonator.operations[op_name]
            operation.integration_weights_angle -= float(fit_result["iw_angle"])

            pair_ids = {getattr(dot_pair, "id", None), getattr(dot_pair, "name", None)} - {None, ""}
            for pair_id in pair_ids:
                sensor_dot._add_readout_params(pair_id, threshold=float(fit_result["I_threshold"]))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

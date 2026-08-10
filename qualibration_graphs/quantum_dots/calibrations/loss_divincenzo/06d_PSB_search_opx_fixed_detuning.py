# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam

from calibration_utils.psb_search_fixed_detuning import (
    Parameters,
    assemble_labeled_ds_raw,
    fit_fixed_detuning_raw_data,
    generate_simulated_dataset,
    log_fitted_results,
    modify_and_track_point,
    plot_all,
    process_raw_dataset,
    resolve_qubits_and_dot_pairs,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot


# %% {Node initialisation}
description = """
PAULI SPIN BLOCKADE SEARCH - Fixed detuning, labeled two-state readout (OPX)

This node acquires labeled shot-by-shot IQ data at a fixed PSB measurement point for each
selected qubit. Each shot is measured twice: first without a pi pulse and then with an
``x180`` pulse, so the two arms prepare complementary spin states according to
``init_state_label``.

The resulting labeled IQ data can be analyzed with either the physics-based Barthel 1D
readout model or a two-component Gaussian mixture model. Qubit/readout dot pairs are
resolved automatically from ``qubit.preferred_readout_quantum_dot``.

Prerequisites
-------------
- QUAM configured and loaded (``quam_config/populate_quam_state_*.py``).
- Sensor-dot readout calibrated (resonator calibration nodes completed).
- ``x180`` pulse calibrated on each selected qubit.
- Fixed readout point defined on the corresponding dot pair; this node can optionally
  override the measure-point detuning temporarily via ``parameters.detuning``.

Datasets
--------
- ``ds_raw``: shot-level ``I_no_pi``, ``Q_no_pi``, ``I_pi``, ``Q_pi`` (dims: ``qubit``, ``n_runs``).
- ``ds_processed``: labeled ``Ig``, ``Qg``, ``Ie``, ``Qe`` used by the analysis/plotting pipeline.
- ``ds_fit``: model-specific fitted dataset returned by the selected Barthel or GMM analysis.
- ``fit_results``: per-qubit scalar results (serialized dataclass) for logging and state updates.

Results
-------
For each qubit, the node extracts the readout-axis rotation and discrimination threshold
used for PSB state discrimination at the fixed measurement point.

Figures
-------
- Raw IQ and rotated IQ with the state-update threshold
- Labeled S/T histograms with either the Barthel analytic fit or GMM Gaussian components

State update
------------
Reverts any temporary detuning override, then (if the fit succeeded) updates the
integration-weights angle and discrimination threshold on the corresponding sensor dot.
"""


node = QualibrationNode[Parameters, Quam](
    name="06d_PSB_search_opx_fixed_detuning",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Local debugging-only parameter overrides."""
    # You can get type hinting in your IDE by typing node.parameters.
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the fixed-point labeled-IQ QUA program."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Select which qubits participate and resolve the paired PSB readout dots
    qubits, qubit_dot_pairs = resolve_qubits_and_dot_pairs(node)
    node.namespace["qubits"] = qubits
    node.namespace["qubit_dot_pairs"] = qubit_dot_pairs

    # Number of shots at the fixed measurement point
    n_avg = node.parameters.num_shots

    # Temporary detuning override tracking (reverted in update_state)
    node.namespace["tracked_original_detunings"] = {}
    for _, dot_pair in qubit_dot_pairs:
        modify_and_track_point(dot_pair, node.parameters.detuning, node.namespace["tracked_original_detunings"])

    # The only streamed axis is the shot index.
    node.namespace["sweep_axes"] = {
        "n_runs": xr.DataArray(np.arange(n_avg), attrs={"long_name": "shot"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Allocate real-time variables on the OPX:
        #   I_no_pi_st, Q_no_pi_st : buffers collecting the I/Q of the no-pi-pulse arm before transfer to PC
        #   I_pi_st, Q_pi_st : buffers collecting the I/Q of the pi-pulse arm before transfer to PC
        #   n            : shot counter
        #   n_st         : stream reporting shot index to PC (progress bar)
        n = declare(int)
        n_st = declare_output_stream()
        I_no_pi_st = {q.name: declare_output_stream() for q in qubits}
        Q_no_pi_st = {q.name: declare_output_stream() for q in qubits}
        I_pi_st = {q.name: declare_output_stream() for q in qubits}
        Q_pi_st = {q.name: declare_output_stream() for q in qubits}

        # ── OUTER LOOP: repeat the full sweep n_avg times ──
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)  # tell the PC which shot we are on

            for qubit, dot_pair in qubit_dot_pairs:
                # ── ARM 1: measure without a pi pulse ─────────────────────

                # Perform the initialize macro
                dot_pair.initialize()
                align()

                # Perform the measure macro (ramps to the measure point)
                i_no_pi, q_no_pi, _ = dot_pair.measure(return_iq=True)

                # Add this run to the no-pi-pulse stream buffer
                save(i_no_pi, I_no_pi_st[qubit.name])
                save(q_no_pi, Q_no_pi_st[qubit.name])

                # Make sure the outputs are ramped to zero at the end of the arm
                align()
                dot_pair.voltage_sequence.ramp_to_zero()

                # ── ARM 2: measure after an x180 pulse ─────────────────────

                # Perform the initialize macro
                dot_pair.initialize()

                # Perform the x180 macro
                align()
                qubit.x180()
                align()

                # Perform the measure macro (ramps to the measure point)
                i_pi, q_pi, _ = dot_pair.measure(return_iq=True)

                # Add this run to the pi-pulse stream buffer
                save(i_pi, I_pi_st[qubit.name])
                save(q_pi, Q_pi_st[qubit.name])

                # Make sure the outputs are ramped to zero at the end of the arm
                align()
                dot_pair.voltage_sequence.ramp_to_zero()

        # ── Post-processing on the OPX before data reaches the PC ──────────
        with stream_processing():
            n_st.save("n")  # expose shot counter as "n" in the fetched dataset
            for qubit in qubits:
                # Each save() above is one run.
                # .buffer(n_avg) : group points along the repetitions axis
                # Result : 1D trace I(n_avg), Q(n_avg) per qubit pair
                I_no_pi_st[qubit.name].buffer(n_avg).save(f"I_no_pi_{qubit.name}")
                Q_no_pi_st[qubit.name].buffer(n_avg).save(f"Q_no_pi_{qubit.name}")
                I_pi_st[qubit.name].buffer(n_avg).save(f"I_pi_{qubit.name}")
                Q_pi_st[qubit.name].buffer(n_avg).save(f"Q_pi_{qubit.name}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate a synthetic labeled two-arm ``ds_raw`` for offline analysis."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated fixed-detuning PSB dataset generated successfully.")


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.use_simulated_data
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute QUA, fetch the shot streams, and assemble the labeled-arm ``ds_raw`` dataset."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    qubits = node.namespace["qubits"]

    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())

    node.results["ds_raw"] = assemble_labeled_ds_raw(dataset, qubits)


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
    node.namespace["tracked_original_detunings"] = {}


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Convert raw no-pi/pi IQ streams into the labeled dataset used downstream."""
    node.results["ds_processed"] = process_raw_dataset(node.results["ds_raw"], node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit the labeled IQ shots with the selected Barthel or GMM model."""
    node.results["ds_fit"], fit_results = fit_fixed_detuning_raw_data(node)
    node.results["fit_results"] = {str(name): asdict(result) for name, result in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Generate all node figures via the fixed-detuning plotting API."""
    node.results["figures"] = plot_all(
        node.results["ds_processed"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        fit_results=node.results["fit_results"],
        analysis_model=node.parameters.analysis_model,
    )
    if not node.modes.external:
        plt.show()


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

    fit_results = node.results.get("fit_results")
    if not fit_results:
        return

    with node.record_state_updates():
        for qubit, dot_pair in node.namespace["qubit_dot_pairs"]:
            fit_result = fit_results.get(qubit.name)
            if fit_result is None or not fit_result["success"]:
                continue

            sensor_dot = dot_pair.sensor_dots[0]
            op_name = f"readout_{dot_pair.name}"
            operation = sensor_dot.readout_resonator.operations[op_name]
            operation.integration_weights_angle -= float(fit_result["iw_angle"])

            pair_ids = {getattr(dot_pair, "id", None), getattr(dot_pair, "name", None)} - {None, ""}
            for pair_id in pair_ids:
                sensor_dot._add_readout_params(pair_id, threshold=float(fit_result["I_threshold"]))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist node results to storage."""
    node.save()

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.experiment import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import QubitQuam as Quam

from calibration_utils.psb_search_sweep_detuning_vs_buffer import (
    Parameters,
    assemble_ds_raw,
    fit_detuning_vs_buffer_raw_data,
    generate_simulated_dataset,
    log_fitted_results,
    plot_all,
    process_raw_dataset,
    validate_and_build_arrays,
    extract_vgs_id,
)


# %% {Node initialisation}
description = """
PAULI SPIN BLOCKADE SEARCH — Sweep detuning vs buffer duration (OPX)

This node sweeps the PSB measure-point detuning and the pre-readout buffer duration
to map where the readout contrast is strongest. Each sweep point prepares the chosen
state, ramps to the target detuning together with a barrier-gate offset, waits for
the swept buffer duration, and then performs sensor readout.

Prerequisites
-------------
- QUAM configured and loaded (``quam_config/populate_quam_state_*.py``).
- Sensor-dot readout calibrated (resonator calibration nodes completed).
- Empty / initialize / measure macros defined on the dot pair.

Datasets
--------
- ``ds_raw``: shot-level ``I`` and ``Q`` vs ``detuning`` and ``buffer_duration``
  (dims: ``qubit_pair``, ``n_runs``, ``detuning``, ``buffer_duration``).
- ``ds_processed``: processed copy of ``ds_raw`` used by the analysis/plotting pipeline.
- ``ds_fit``: 2D PCA-derived contrast maps (currently ``pc1_std`` and ``iq_trace``).
- ``fit_results``: per-pair scalar results (serialized dataclass) for logging.

Results
-------
For each qubit pair, the node selects the detuning / buffer-duration point that maximizes
the chosen exploratory contrast metric.

Figures
-------
- 2D heatmap of the selected PCA-derived metric vs detuning and buffer duration

State update
------------
Updates the measure-point detuning and persists the optimal ``buffer_duration`` on
the pair ``measure`` macro when supported.
"""


node = QualibrationNode[Parameters, Quam](
    name="06e_PSB_search_opx_sweep_detuning_vs_buffer",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the sweep axes and the QUA pulse sequence."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Select which qubit-pairs participate in this calibration
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    # Ensure that the machine is set up to track the integrated voltage
    node.machine.reset_voltage_sequence(extract_vgs_id(qubit_pairs), track_integrated_voltage=True)

    # Number of shots per detuning point
    n_avg = node.parameters.num_shots

    # Validate the 2D sweep and build both the OPX clock-cycle axis detuning axis
    detuning_array, buffer_cc_array, buffer_ns_array = validate_and_build_arrays(node)
    node.namespace["detuning_array"] = detuning_array
    node.namespace["buffer_ns_array"] = buffer_ns_array

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "n_runs": xr.DataArray(np.arange(n_avg), attrs={"long_name": "shot"}),
        "detuning": xr.DataArray(
            detuning_array,
            attrs={"long_name": "detuning", "units": "V"},
        ),
        "buffer_duration": xr.DataArray(
            buffer_ns_array.astype(float),
            attrs={"long_name": "buffer duration", "units": "ns"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Allocate real-time variables on the OPX:
        #   I[i], Q[i]   : demodulated quadratures for qubit_pair i
        #   I_st[i], Q_st[i] : buffers collecting I/Q before transfer to PC
        #   n            : shot counter
        #   n_st         : stream reporting shot index to PC (progress bar)
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=len(qubit_pairs))

        # Real-time variables holding the detuning value and the buffer length,
        # in integer clock cycles.
        detuning = declare(fixed)
        buffer_cc = declare(int)

        # ── OUTER LOOP: repeat the full sweep n_avg times ─────────────────
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)  # tell the PC which shot we are on

            # Perform the selected pairs sequentially for now
            for i, qubit_pair in enumerate(qubit_pairs):
                # Start with global align
                align()
                # Extract the dot pair associated with the qubit pair
                dot_pair = qubit_pair.quantum_dot_pair
                # Extract the dot-pair specific readout infomation before the loops
                sensor = dot_pair.sensor_dots[0]
                rr = sensor.readout_resonator
                op_name = f"readout_{dot_pair.name}"
                readout_length = rr.operations[op_name].length

                # ── MIDDLE LOOP: step detuning value ──────────────────────
                with for_(*from_array(detuning, detuning_array)):

                    # ── INNER LOOP: sweep the buffer duration ─────────────
                    with for_(*from_array(buffer_cc, buffer_cc_array)):

                        # ── STEP 1 - SETUP & INITIALIZE: Setup the sweep and initialize ──────────

                        # Choose to initialize a particular qubit pair. If None,
                        # then by default initialize the same pair that is measured
                        if node.parameters.qubit_pair_to_initialize is not None:
                            init_pair = node.machine.qubit_pairs[node.parameters.qubit_pair_to_initialize]
                            init_pair.quantum_dot_pair.macros[node.parameters.initialization_macro].apply()
                        else:
                            dot_pair.macros[node.parameters.initialization_macro].apply()

                        # Optionally drive a qubit using the x180 macro.
                        if node.parameters.qubit_to_pulse is not None:
                            q = node.machine.qubits[node.parameters.qubit_to_pulse]

                            # Since the qubit's xy component is a separate element in QUA, ensure that this is exactly aligned to the VoltageSequence
                            align(q.xy.id, dot_pair.physical_channel.id)
                            q.x180()
                            align(q.xy.id, dot_pair.physical_channel.id)

                        # Align the start of the resonator's wait command to the END of the initialization macro
                        align(rr.id, dot_pair.physical_channel.id)

                        # ── STEP 2 - RAMP & WAIT: Ramp to the readout point and wait for a buffer duration ──────────

                        # Ramp to the requested detuning / barrier point and wait for the buffer duration and the readout length
                        dot_pair.ramp_to_voltages(
                            {
                                dot_pair.name: detuning,
                                dot_pair.barrier_gate.name: node.parameters.barrier_gate_voltage,
                            },
                            ramp_duration=node.parameters.ramp_duration,
                            duration=readout_length + buffer_cc * 4,
                        )

                        # ── STEP 3 - MEASURE: Send the readout pulse and demodulate ──────────

                        # Resonator will be sat idle during the ramp + buffer. wait() function argument is in clock cycles, hence the division by 4
                        rr.wait(node.parameters.ramp_duration // 4 + buffer_cc)
                        # Resonator will measure after waiting for ramp + buffer, means that this corresponds to the readout_length wait time in line 178
                        # Save the measured values in the QUA variables for I and Q
                        rr.measure(op_name, qua_vars=(I[i], Q[i]))

                        # Add this run to the stream buffers
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

                        # Apply the compensation pulse via the voltage sequence. This both steps to 0 before, and goes back to 0 after
                        # Compensation begins after the ramp + buffer + readout_length, so all should be synchronised nicely
                        dot_pair.voltage_sequence.apply_compensation_pulse(go_to_zero=True, return_to_zero=True)

                        align()

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")  # expose shot counter as "n" in the fetched dataset
            for i, qp in enumerate(qubit_pairs):
                # Each save() above is one 2D sweep point.
                # .buffer(len(buffer_cc_array)) : group points along buffer duration
                # .buffer(len(detuning_array))  : group points along detuning
                # .buffer(n_avg)                : group points along repetitions
                # Result : 3D trace I(buffer, detuning, n_avg), Q(buffer, detuning, n_avg)
                I_st[i].buffer(len(buffer_cc_array)).buffer(len(detuning_array)).buffer(n_avg).save(f"I_{qp.name}")
                Q_st[i].buffer(len(buffer_cc_array)).buffer(len(detuning_array)).buffer(n_avg).save(f"Q_{qp.name}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        # "samples": samples,
    }


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate a synthetic shot-level ``ds_raw`` for offline analysis."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated detuning-vs-buffer PSB dataset generated successfully.")


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.use_simulated_data
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
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

    # Reshape the per-pair streams into a qubit_pair-indexed ds with I/Q variables.
    pair_names = [pair.name for pair in node.namespace["qubit_pairs"]]
    node.results["ds_raw"] = assemble_ds_raw(dataset, pair_names)


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_processed"] = process_raw_dataset(node.results["ds_raw"].copy(deep = True), node)
    node.results["ds_fit"], fit_results = fit_detuning_vs_buffer_raw_data(node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Generate all node figures via the shared plotting API."""
    node.results["figures"] = plot_all(
        node.results["ds_fit"],
        metric_name=node.parameters.pca_metric,
        fit_results=node.results["fit_results"],
    )
    node.results["figure"] = node.results["figures"]["detuning_vs_buffer_pca_map"]
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor data analysis was successful."""
    fit_results = node.results.get("fit_results")
    if not fit_results:
        return

    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            fit_result = fit_results[qp.name]
            if not fit_result["success"]:
                continue

            dot_pair = qp.quantum_dot_pair
            point_name = dot_pair._create_point_name("measure")
            point = dot_pair.voltage_sequence.gate_set.get_macros()[point_name]

            point.voltages[dot_pair.name] = float(fit_result["optimal_detuning"])
            point.voltages[dot_pair.barrier_gate.name] = node.parameters.barrier_gate_voltage

            measure_macro = dot_pair.macros.get("measure")
            if measure_macro is not None:
                optimal_ns = int(round(float(fit_result["optimal_buffer_duration"])))
                try:
                    measure_macro.update(buffer_duration=optimal_ns)
                except TypeError:
                    node.log(
                        f"Skipping measure macro buffer_duration update for {dot_pair.id!r}: "
                        "macro.update does not accept buffer_duration."
                    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist node results to storage."""
    node.save()

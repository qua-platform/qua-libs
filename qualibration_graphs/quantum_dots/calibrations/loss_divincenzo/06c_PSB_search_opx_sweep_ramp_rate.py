# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array

from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam
from calibration_utils.psb_search_sweep_ramp_rate import (
    Parameters,
    fit_sweep_rate_raw_data,
    generate_simulated_dataset,
    process_raw_dataset,
    log_fitted_results,
    plot_all,
    prepare_dot_pairs,
    modify_and_track_point,
    validate_and_build_ramp_sweep,
    extract_vgs_id,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.experiment import get_qubit_pairs


# %% {Node initialisation}
description = """
PAULI SPIN BLOCKADE SEARCH — Sweep ramp duration to measure (OPX)

This node probes PSB readout contrast while sweeping the ramp duration (ns) used to reach
the PSB measurement point. For a fixed voltage trajectory, shorter ramps correspond to
higher effective ramp rates on the OPX fast lines.

The sequence matches the detuning-sweep PSB nodes (06a/06b) except the swept axis is the
ramp duration: preparation via ``initialization_macro`` (default ``empty``), then for each
ramp duration a ``ramp_to_point('measure', ...)`` is executed, followed by resonator readout.
An optional detuning override (``parameters.detuning``) can be applied temporarily.

Prerequisites
-------------
- QUAM configured and loaded (``quam_config/populate_quam_state_*.py``).
- Sensor-dot readout calibrated (resonator calibration nodes completed).
- Empty / initialize / measure macros defined on the dot pair.
- Prefer running 06a/06b first to set a reasonable measure detuning; this node can optionally
  override detuning temporarily via ``parameters.detuning``.

Datasets
--------
- ``ds_raw``: shot-level ``I`` and ``Q`` vs ramp duration (dims: ``qubit_pair``, ``n_runs``, ``ramp_duration``).
- ``ds_fit``: readout metrics vs ramp duration (PCA + two-Gaussian EM per sweep point).
- ``fit_results``: per-pair scalar results (serialized dataclass) for logging and state updates.

Results
-------
For each qubit pair, the node selects an optimal ramp duration (fidelity or visibility)
and extracts the readout axis and threshold used for PSB discrimination at that optimum.

Figures
-------
- Fidelity and visibility vs ramp duration
- Sweep summary (fidelity + visibility on twin axes)
- Shot histograms vs ramp duration (projected readout axis; normalized by sweep value)
- Rotated IQ density at the optimal ramp duration with the chosen threshold

State update
------------
Reverts temporary detuning override, then (if the fit succeeded) persists the optimal ramp
duration on the pair ``measure`` macro when supported, and updates integration-weights angle
and discrimination threshold on the sensor dot (readout pulse length is not changed).
"""


node = QualibrationNode[Parameters, Quam](
    name="06c_PSB_search_opx_sweep_ramp_rate",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the sweep axes and the QUA pulse sequence."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Select which qubit-pairs participate in this calibration, and save the information in the namespace
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    node.namespace["dot_pairs"] = [qp.quantum_dot_pair for qp in qubit_pairs]

    # Ensure that the machine is set up to track the integrated voltage
    node.machine.reset_voltage_sequence(extract_vgs_id(qubit_pairs), track_integrated_voltage=True)

    # Number of shots per sweep point
    n_avg = node.parameters.num_shots

    # Temporary detuning override tracking (reverted in update_state)
    node.namespace["tracked_original_detunings"] = {}
    for qubit_pair in qubit_pairs:
        modify_and_track_point(
            qubit_pair,
            node.parameters.detuning,
            node.namespace["tracked_original_detunings"],
        )

    # Build the ramp-duration sweep
    ramp_duration_array = validate_and_build_ramp_sweep(node)
    ramp_durations_cc = ramp_duration_array // 4
    buffer_duration_cc = node.parameters.buffer_duration // 4

    # The swept axes. Buffer order is (ramp_duration) then (n_runs).
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([qp.name for qp in qubit_pairs]),
        "n_runs": xr.DataArray(np.arange(n_avg), attrs={"long_name": "shot"}),
        node.parameters.sweep_name: xr.DataArray(
            ramp_duration_array.astype(float),
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        # Allocate real-time variables on the OPX:
        #   I_st[name], Q_st[name] : buffers collecting I/Q before transfer to PC
        #   n            : shot counter
        #   n_st         : stream reporting shot index to PC (progress bar)
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=len(qubit_pairs))

        # Real time variable for the ramp duration
        ramp_d = declare(int)

        # ── OUTER LOOP: repeat the full sweep n_avg times ──
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)  # tell the PC which shot we are on

            # Loop over the qubit pairs involved in this experiment
            for i, qubit_pair in enumerate(qubit_pairs):
                dot_pair = qubit_pair.quantum_dot_pair

                # Use the first sensor associated to each dot pair
                sensor = dot_pair.sensor_dots[0]
                rr = sensor.readout_resonator

                # Extract the readout pulse name and the readout length
                op_name = f"readout_{dot_pair.name}"
                readout_len = rr.operations[op_name].length

                # ── INNER LOOP: sweep ramp duration ───────────────────────────
                with for_(*from_array(ramp_d, ramp_durations_cc)):

                    # ── STEP 0 - RESET: ensure settling between sweep points ──────────
                    wait(node.parameters.reset_wait_time // 4)
                    align()  # Start loop with a global align after the reset time. This ensures that all the elements will start here

                    # ── STEP 1 - INITIALIZE: preparation macro (empty or initialize) ─
                    dot_pair.macros[node.parameters.initialization_macro].apply()

                    # Align the readout resonator to the end of the initialize
                    align(rr.id, dot_pair.physical_channel.id)

                    # ── STEP 2 - RAMP: ramp to measure with the swept duration ───────
                    dot_pair.ramp_to_point(
                        "measure",
                        ramp_duration=ramp_d * 4,  # Convert back from clock cycles to nanoseconds
                        duration=buffer_duration_cc * 4 + readout_len,
                    )

                    # ── STEP 3 - MEASURE: resonator readout at the PSB point ─────────
                    # Resonator sits idle for ramp duration + buffer duration
                    rr.wait(ramp_d + buffer_duration_cc)  # This wait command is in clock cycles

                    # Measure the demodulated measurement and save into the QUA variables
                    rr.measure(op_name, qua_vars=(I[i], Q[i]))

                    # Append this sweep point's I/Q to the stream buffer
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

                    # Apply the compensation pulse via the voltage sequence
                    dot_pair.voltage_sequence.apply_compensation_pulse(
                        go_to_zero=True,
                        return_to_zero=True,
                    )

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            n_r = len(ramp_duration_array)
            for i, qp in enumerate(qubit_pairs):
                # Each save() above is one sweep point.
                # .buffer(n_r)   : group points along the ramp_duration axis
                # .buffer(n_avg) : group points along the repetitions axis
                # Result : 2D trace I(ramp_duration, n_runs), Q(ramp_duration, n_runs) per qubit pair
                I_st[i].buffer(n_r).buffer(n_avg).save(f"I_{qp.name}")
                Q_st[i].buffer(n_r).buffer(n_avg).save(f"Q_{qp.name}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
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
    node.log("[sim] Simulated PSB ramp-duration dataset generated successfully.")


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
        job.wait_until("Done")
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Reshape the per-pair streams into a qubit_pair-indexed ds with I/Q variables.
    pair_names = [pair.name for pair in node.namespace["qubit_pairs"]]
    I_arr = xr.concat([dataset[f"I_{p}"] for p in pair_names], dim="qubit_pair")
    Q_arr = xr.concat([dataset[f"Q_{p}"] for p in pair_names], dim="qubit_pair")
    I_arr = I_arr.assign_coords(qubit_pair=pair_names)
    Q_arr = Q_arr.assign_coords(qubit_pair=pair_names)
    node.results["ds_raw"] = xr.Dataset({"I": I_arr, "Q": Q_arr})


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw dataset into an analysis-ready form (keeps ``ds_raw`` immutable)."""
    node.results["ds_processed"] = process_raw_dataset(node.results["ds_raw"], node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_fit"], fit_results = fit_sweep_rate_raw_data(node)
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
    # s and alpha are relevant kwargs for plotting a scatter plot.
    # Hard coded here as 4 and 0.15, since they should not be exposed as node parameters.
    sweep_name = node.parameters.sweep_name
    node.results["figures"] = plot_all(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        node.results["ds_fit"],
        sweep_name=sweep_name,
        fit_results=node.results["fit_results"],
        plot_kde=node.parameters.plot_kde,
        s=4,
        alpha=0.15,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor data analysis was successful."""
    for qubit_pair in node.namespace["qubit_pairs"]:
        dot_pair = qubit_pair.quantum_dot_pair
        if dot_pair.name in node.namespace.get("tracked_original_detunings", {}):
            dot_pair_gate_set = dot_pair.voltage_sequence.gate_set
            point_name = dot_pair._create_point_name("measure")
            point = dot_pair_gate_set.get_macros()[point_name]
            point.voltages[dot_pair.name] = node.namespace["tracked_original_detunings"][dot_pair.name]

    fit_results = node.results.get("fit_results")
    if not fit_results:
        return

    with node.record_state_updates():
        op_name = node.parameters.operation
        for qp in node.namespace["qubit_pairs"]:
            fit_result = fit_results[qp.name]
            if not fit_result["success"]:
                continue

            dot_pair = qp.quantum_dot_pair
            sensor_dot = dot_pair.sensor_dots[0]
            operation = sensor_dot.readout_resonator.operations[op_name]

            optimal_ns = int(round(float(fit_result["optimal_sweep_value"])))

            measure_macro = dot_pair.macros.get("measure")
            if measure_macro is not None:
                try:
                    measure_macro.update(ramp_duration=optimal_ns)
                except TypeError:
                    node.log(
                        f"Skipping measure macro ramp_duration update for {dot_pair.id!r}: "
                        "macro.update does not accept ramp_duration."
                    )

            operation.integration_weights_angle -= float(fit_result["iw_angle"])

            pair_ids = {
                getattr(dot_pair, "id", None),
                getattr(dot_pair, "name", None),
            } - {None, ""}
            for pair_id in pair_ids:
                sensor_dot._add_readout_params(pair_id, threshold=float(fit_result["I_threshold"]))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist node results to storage."""
    node.save()

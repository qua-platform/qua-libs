# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from qualibration_libs.parameters.experiment import get_qubit_pairs
from quam_config import QubitQuam as Quam
from calibration_utils.psb_search_sweep_measure_duration import (
    Parameters,
    process_raw_dataset,
    build_psb_readout_sweep,
    fit_measure_duration_raw_data,
    generate_simulated_dataset,
    log_fitted_results,
    plot_all,
    modify_and_track_point,
    modify_and_track_readout_pulse,
    validate_readout,
    prepare_dot_pairs,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Node initialisation}
description = """
PAULI SPIN BLOCKADE SEARCH — Fixed detuning, sweep readout length (OPX)

This node probes PSB readout contrast while sweeping the resonator integration time
(readout pulse length / accumulated demodulation segments) at a fixed measure-point
detuning (optionally overridden via node parameters).

Because the sequence uses ``measure_accumulated``, the readout pulse length is constrained
to an integer number of integration chunks:

    ``pulse_length = N * 4 * segment_length``  (ns)

Prerequisites
-------------
- QUAM configured and loaded (``quam_config/populate_quam_state_*.py``).
- Sensor-dot readout calibrated (resonator calibration nodes completed).
- Empty / initialize / measure macros defined on the dot pair.
- Prefer running 06a first to set a reasonable measure detuning; this node can optionally
  override detuning temporarily via ``parameters.detuning``.

Datasets
--------
- ``ds_raw``: shot-level ``I`` and ``Q`` vs ``readout_length`` (dims: ``qubit_pair``, ``n_runs``, ``readout_length``).
- ``ds_fit``: readout metrics vs readout length (PCA + two-Gaussian EM per sweep point).
- ``fit_results``: per-pair scalar results (serialized dataclass) for logging and state updates.

Results
-------
For each qubit pair, the node selects an optimal readout length (fidelity or visibility)
and extracts the readout axis and threshold used for PSB discrimination at that optimum.

Figures
-------
- Fidelity and visibility vs readout length
- Sweep summary (fidelity + visibility on twin axes)
- Shot histograms vs readout length (projected readout axis; normalized by sweep value)
- Rotated IQ density at the optimal readout length with the chosen threshold

State update
------------
Reverts temporary detuning/pulse-length overrides, then (if the fit succeeded) persists the
optimal readout ``length``, integration-weights angle, and discrimination threshold on the
pair's sensor dot (same pattern as 05c length + 06a readout calibration).
"""


node = QualibrationNode[Parameters, Quam](
    name="06b_PSB_search_opx_fixed_detuning_measure_duration",
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

    # Number of shots per detuning point
    n_avg = node.parameters.num_shots

    # Validate dot pairs and extract the max readout, in-case none is given
    readout_max = prepare_dot_pairs(node)

    # Build the sweep. This program uses measure_accumulated, so the sweep must be carefully constructed
    sweep = build_psb_readout_sweep(
        node.parameters.readout_length_min,
        readout_max,
        node.parameters.readout_length_points,
    )

    # The array size will be used in the stream processing
    array_size = sweep["array_size"]

    # Save the sweep params in the namespace
    node.namespace["readout_sweep"] = sweep

    # measure_accumulated returns differently depending on ReadoutResonator type:
    #      ReadoutResonatorSingle - returns a single IQ pair, since we have a single analog input
    #      ReadoutResonatorIQ - returns 4 IQs, since we have a dual analog input
    # Therefore, the class must be consistent. The returned readout_cls can be either "single" or "dual"
    readout_cls = validate_readout(qubit_pairs, sweep)

    # Temporary changes to the Quam for the sake of the program. These are:
    #      - The readout pulse length is temporarily changed to match the sweep. This is because this node sweeps the readout length.
    #      - If a different detuning value is desired for this sweep, then this temporarily adds this detuning point to the "measure" point.
    node.namespace["tracked_original_detunings"] = {}
    node.namespace["tracked_resonators"] = []
    for qubit_pair in qubit_pairs:
        modify_and_track_readout_pulse(qubit_pair, sweep["pulse_length"], node.namespace["tracked_resonators"])
        modify_and_track_point(qubit_pair, node.parameters.detuning, node.namespace["tracked_original_detunings"])

    # The swept axes. Buffer order is (measure_duration) then (n_runs).
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([qp.name for qp in qubit_pairs]),
        "n_runs": xr.DataArray(np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}),
        node.parameters.sweep_name: xr.DataArray(
            sweep["sweep_coord"], attrs={"long_name": "readout length", "units": "ns"}
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Allocate real-time variables on the OPX:
        #   I_st[i], Q_st[i] : buffers collecting I/Q before transfer to PC
        #   n            : shot counter
        #   n_st         : stream reporting shot index to PC (progress bar)
        #   tmp_i, tmp_q : run-time variables to fill up the stream buffers
        n = declare(int)
        n_st = declare_output_stream()
        I_st = [declare_output_stream() for _ in qubit_pairs]
        Q_st = [declare_output_stream() for _ in qubit_pairs]
        tmp_i = declare(fixed)
        tmp_q = declare(fixed)

        # Real time variable to iterate over the measure_accumulated array
        idx = declare(int)

        # ── OUTER LOOP: repeat the full sweep n_avg times ──
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)  # tell the PC which shot we are on

            # Loop over the qubit_pairs
            for i, qubit_pair in enumerate(qubit_pairs):
                align() # Initial global align

                # ── STEP 1 - SETUP & INITIALIZE: Setup the sweep and initialize ──────────
                # Extract the underlying quantum_dot_pair and its readout name
                dot_pair = qubit_pair.quantum_dot_pair
                op_name = "readout" + f"_{dot_pair.name}"

                # Use the first sensor dot in the list of sensors associated with the quantum dot pair
                sensor = dot_pair.sensor_dots[0]
                rr = sensor.readout_resonator
                readout_length = rr.operations[op_name].length

                # Perform the chosen initialize macro
                dot_pair.macros[node.parameters.initialization_macro].apply()

                # Align the start of the resonator's wait command to the END of the initialization macro
                align(rr.id, dot_pair.physical_channel.id)

                # ── STEP 2 - RAMP: Ramp to the measure point ──────────
                dot_pair.ramp_to_point(
                    "measure",
                    ramp_duration=node.parameters.ramp_duration,
                    duration=node.parameters.buffer_duration + readout_length,
                )

                # Resonator will be sat idle during the ramp + buffer. wait() function argument is in clock cycles, hence the division by 4
                rr.wait((node.parameters.ramp_duration + node.parameters.buffer_duration)//4)


                # ── STEP 3 - MEASURE: Perform the measurement ──────────
                
                # Play the "readout_{quantum_dot_pair.name}" pulse and cumulatively integrate I/Q
                # measure_accumulated returns differently depending on ReadoutResonator type:
                #      ReadoutResonatorSingle - returns a single IQ pair, since we have a single analog input
                #      ReadoutResonatorIQ - returns 4 IQs, since we have a dual analog input
                if readout_cls == "single":
                    I_acc, Q_acc = rr.measure_accumulated(op_name, segment_length=sweep["segment_length"])
                else:
                    II_a, IQ_a, QI_a, QQ_a = rr.measure_accumulated(op_name, segment_length=sweep["segment_length"])

                # Apply the compensation pulse via the voltage sequence. This both steps to 0 before, and goes back to 0 after
                dot_pair.voltage_sequence.apply_compensation_pulse(go_to_zero=True, return_to_zero=True)

                # Measure_accumulated returns arrays, where each value cumulatively integrates
                # Now we loop over the arrays, iteratively saving the value into the streams
                if readout_cls == "single":
                    with for_(idx, 0, idx < array_size, idx + 1):
                        save(I_acc[idx], I_st[i])
                        save(Q_acc[idx], Q_st[i])
                else:
                    with for_(idx, 0, idx < array_size, idx + 1):
                        assign(tmp_i, II_a[idx] - QQ_a[idx])
                        assign(tmp_q, IQ_a[idx] + QI_a[idx])
                        save(tmp_i, I_st[i])
                        save(tmp_q, Q_st[i])

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for i, qubit_pair in enumerate(qubit_pairs):
                # Each save() above is one voltage point.
                # .buffer(array_size) : group points along the readout_length axis
                # .buffer(n_avg) : group points along the repetitions axis
                # Result : 2D trace I(detuning, n_avg), Q(detuning, n_avg) per qubit pair
                I_st[i].buffer(array_size).buffer(n_avg).save(f"I_{qubit_pair.name}")
                Q_st[i].buffer(array_size).buffer(n_avg).save(f"Q_{qubit_pair.name}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect(timeout=600)
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated PSB readout-length dataset generated successfully.")


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
    node.namespace["dot_pairs"] = [qp.quantum_dot_pair for qp in node.namespace["qubit_pairs"]]


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw dataset into a plotting/analysis-ready dataset (keeps ds_raw immutable)."""
    node.results["ds_processed"] = process_raw_dataset(node.results["ds_raw"], node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit PCA + two-Gaussian readout model at each readout length (same stack as 06a)."""
    node.results["ds_fit"], fit_results = fit_measure_duration_raw_data(node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot all node figures via the shared plotting API."""
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
    """Revert temporary patches, then persist optimal readout length and readout calibration."""
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    for dot_pair in node.namespace["dot_pairs"]:
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
            op_name = "readout" + f"_{dot_pair.name}"
            sensor_dot = dot_pair.sensor_dots[0]
            operation = sensor_dot.readout_resonator.operations[op_name]

            optimal_ns = int(round(float(fit_result["optimal_sweep_value"])))
            operation.length = optimal_ns

            operation.integration_weights_angle -= float(fit_result["iw_angle"])
            print(
                f"For sensor {sensor_dot.name}, pair {dot_pair.name}, threshold calculated to be {fit_result['I_threshold']} and angle {float(fit_result['iw_angle'])}"
            )
            sensor_dot._add_readout_params(dot_pair.name, threshold=float(fit_result["I_threshold"]))
            sensor_dot.readout_thresholds[dot_pair.name] = float(fit_result["I_threshold"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

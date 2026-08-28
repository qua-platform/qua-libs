# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam

from calibration_utils.charge_state_readout_time_optimization import (
    Parameters,
    analyse_raw_data,
    process_raw_dataset,
    plot_all,
    generate_simulated_dataset,
    get_dot_pairs,
    get_dot_pair_sensors,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

# %% {Node initialisation}
description = """
CHARGE STATE READOUT TIME OPTIMIZATION

This sequence measures the integration-time dependence of charge-state readout SNR for
selected quantum-dot pairs. For each pair, the node initializes the system, acquires the
(1,1) reference distribution, ramps to a configurable ``detuning_02`` value to prepare
the (0,2) readout state, and acquires accumulated IQ chunks throughout one long readout
pulse. The accumulated chunks are analyzed to determine the first integration time that
reaches the target SNR.

Prerequisites:
    - Readout resonators biased to their sensitive frequency.
    - Sensor dots calibrated for the selected quantum-dot pairs.
    - A suitable (1,1) operating point and ``detuning_02`` readout point identified.
    - Pair-specific readout operations ``readout_<dot_pair_name>`` available on the sensor resonators.

Datasets:
    - ``ds_raw``: untouched accumulated IQ chunks for the (1,1) and (0,2) reference states.
    - ``ds_fit``: fitted SNR traces and derived readout metrics for each dot-pair/sensor combination.

Results (``node.results["fit_results"][<dot_pair>_<sensor>]``):
    - ``optimal_integration_time``: first integration time that reaches the target SNR.
    - ``iw_angle``: rotation angle aligning the discrimination axis.
    - ``I_threshold``: discrimination threshold in the rotated I frame.
    - ``used_double_gaussian``: whether the (0,2) state was treated as T1-limited.

Figures (``node.results["figures"]``):
    - ``"iq_histogram"``: 2D IQ histograms at the longest integration time.
    - ``"snr_vs_integration_time"``: SNR traces with the selected operating point.
    - ``"projected_histogram"``: rotated-I histograms at the chosen integration time.

State update:
    - The readout pulse length is set to the optimal integration time.
    - The integration-weights angle is updated so (1,1) maps to the higher rotated-I value.
    - The rotated-I threshold is stored for the relevant dot-pair IDs.
"""

node = QualibrationNode[Parameters, Quam](
    name="05d_charge_state_readout_time_optimization",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Extract the dot pairs and sensors
    node.namespace["quantum_dot_pairs"], _ = quantum_dot_pairs, vgs_id = get_dot_pairs(node)
    node.namespace["all_sensors"] = all_sensors = get_dot_pair_sensors(node)

    # Ensure that the machine is set up to track the integrated voltage
    node.machine.reset_voltage_sequence(vgs_id, track_integrated_voltage=True)

    # Number of averages
    n_avg = node.parameters.num_shots

    # Set up the integration time array
    integration_times = np.arange(
        node.parameters.integration_time_start,
        node.parameters.integration_time_stop,
        node.parameters.integration_time_step,
    )
    # measure_accumulated takes the number of samples, which is determined by the integration time step
    samples_per_chunk = node.parameters.integration_time_step // 4
    array_size = len(integration_times)

    # Temporarily set the readout pulse length to cover the full integration time range.
    # measure_accumulated chunks data within the readout pulse, so the pulse must be at
    # least as long as the maximum integration time. Reverted in update_state.
    node.namespace["tracked_resonators"] = []
    unique_sensors = {s.name: s for pair in quantum_dot_pairs for s in pair.sensor_dots}
    for s in unique_sensors.values():
        for pair in quantum_dot_pairs:
            if s.name not in [sd.name for sd in pair.sensor_dots]:  # compare by name, not identity
                continue
            op_name = f"readout_{pair.name}"
            resolved_op_name = op_name if op_name in s.readout_resonator.operations else "readout"

            with tracked_updates(s.readout_resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
                resonator.operations[resolved_op_name].length = node.parameters.integration_time_stop
                node.namespace["tracked_resonators"].append(resonator)

    # Register the sweep axes to be added to the dataset when fetching data.
    node.namespace["sweep_axes"] = {
        "repetition": xr.DataArray(np.arange(n_avg)),
        "integration_time": xr.DataArray(
            np.arange(1, array_size + 1) * samples_per_chunk * 4,
            attrs={"long_name": "integration time", "units": "ns"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Real-time variables:
        # n  : shot counter
        # idx : a integer index to loop over the QUA array to save the variable

        # Streams:
        # I/Q_st_11 : streams to store the accumulated measurement of the 11 state.
        # I/Q_st_02 : streams to store the accumulated measurement of the 02 state.
        # n_st : stores the shot counter n, allowing the PC to track the progress

        n = declare(int)
        idx = declare(int)
        progress = declare(int)

        I_st_11 = {dp.name: {s.name: declare_stream() for s in dp.sensor_dots} for dp in quantum_dot_pairs}
        Q_st_11 = {dp.name: {s.name: declare_stream() for s in dp.sensor_dots} for dp in quantum_dot_pairs}
        I_st_02 = {dp.name: {s.name: declare_stream() for s in dp.sensor_dots} for dp in quantum_dot_pairs}
        Q_st_02 = {dp.name: {s.name: declare_stream() for s in dp.sensor_dots} for dp in quantum_dot_pairs}
        n_st = declare_stream()

        # Loop over the dot pairs chosen.
        for dp_idx, dot_pair in enumerate(quantum_dot_pairs):
            seq = dot_pair.voltage_sequence

            # TODO: Do we need to declare the QUA variables?
            I_11 = {}
            Q_11 = {}
            I_02 = {}
            Q_02 = {}

            readout_pulse_name = "readout" + f"_{dot_pair.name}"

            # ── OUTER LOOP: average over shots ───────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                assign(progress, dp_idx * n_avg + n)  # Update QUA variable for progress counter
                save(progress, n_st)  # Tell the PC which shot we are on

                align()  # Initial global align

                # ── STEP 1: Initialize ───────────────────────
                dot_pair.initialize(
                    target_state=node.parameters.target_state,
                    max_loops=node.parameters.max_loops,
                )

                align()

                # ── STEP 2: Measure the 11 state ───────────────────────
                for batch in all_sensors[dot_pair.name].batch():
                    for s in batch.values():
                        rr = s.readout_resonator
                        I_11[s.name], Q_11[s.name] = rr.measure_accumulated(
                            readout_pulse_name,
                            segment_length=samples_per_chunk,
                        )

                # ── STEP 3: Ramp to the 02 charge state ───────────────────────
                align()

                seq.ramp_to_voltages(
                    {dot_pair.name: node.parameters.detuning_02},
                    duration=node.parameters.wait_time + node.parameters.integration_time_stop,
                    ramp_duration=node.parameters.ramp_duration,
                )

                # ── STEP 4: Measure the 02 state ───────────────────────
                for batch in all_sensors[dot_pair.name].batch():
                    for s in batch.values():
                        rr = s.readout_resonator
                        rr.wait((node.parameters.wait_time + node.parameters.ramp_duration) // 4)
                        I_02[s.name], Q_02[s.name] = rr.measure_accumulated(
                            readout_pulse_name,
                            segment_length=samples_per_chunk,
                        )

                align()

                # ── STEP 5: Apply the compensation pulse ───────────────────────
                seq.apply_compensation_pulse(return_to_zero=True, go_to_zero=True)

                align()

                # ── STEP 6: Loop over the dicts and QUA arrays to save the data to the streams ───────────────────────
                for batch in all_sensors[dot_pair.name].batch():
                    for s in batch.values():
                        with for_(idx, 0, idx < array_size, idx + 1):
                            save(I_11[s.name][idx], I_st_11[dot_pair.name][s.name])
                            save(Q_11[s.name][idx], Q_st_11[dot_pair.name][s.name])
                            save(I_02[s.name][idx], I_st_02[dot_pair.name][s.name])
                            save(Q_02[s.name][idx], Q_st_02[dot_pair.name][s.name])

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for dp in quantum_dot_pairs:
                for batch in all_sensors[dp.name].batch():
                    for s in batch.values():
                        I_st_11[dp.name][s.name].buffer(array_size).buffer(n_avg).save(f"I_11_{dp.name}_{s.name}")
                        Q_st_11[dp.name][s.name].buffer(array_size).buffer(n_avg).save(f"Q_11_{dp.name}_{s.name}")
                        I_st_02[dp.name][s.name].buffer(array_size).buffer(n_avg).save(f"I_02_{dp.name}_{s.name}")
                        Q_st_02[dp.name][s.name].buffer(array_size).buffer(n_avg).save(f"Q_02_{dp.name}_{s.name}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the OPX and simulate the QUA program."""
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


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.use_simulated_data
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the OPX, execute the QUA program, and fetch raw IQ chunks into ``ds_raw``."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots * len(node.namespace["quantum_dot_pairs"]),
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated IQ data so the full analysis pipeline can run without hardware."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id

    node.namespace["quantum_dot_pairs"], _ = get_dot_pairs(node)
    node.namespace["all_sensors"] = get_dot_pair_sensors(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process ``ds_raw``, fit the integration-time sweep, and store processed outputs."""
    node.namespace["ds_processed"] = ds_processed = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
    (
        node.results["ds_fit"],
        node.results["fit_results"],
        node.outcomes,
    ) = analyse_raw_data(ds_processed, node, log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Build the node figures from the raw and fitted readout-time optimization data."""
    node.results["figures"] = plot_all(
        node.results["ds_raw"],
        node.namespace["all_sensors"],
        node.namespace["quantum_dot_pairs"],
        ds_fit=node.results.get("ds_fit"),
        fit_results=node.results.get("fit_results"),
        show=False,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor_name data analysis was successful."""

    # Revert the readout length change done at the beginning of the node
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    with node.record_state_updates():
        quantum_dot_pairs = node.namespace["quantum_dot_pairs"]
        all_sensors = node.namespace["all_sensors"]
        for dp in quantum_dot_pairs:
            for sensor in all_sensors[dp.name]:
                key = f"{dp.name}_{sensor.name}"
                fit_result = node.results["fit_results"][key]
                if not fit_result["success"]:
                    continue
                optimal_time = int(fit_result["optimal_integration_time"])

                op_name = "readout" + f"_{dp.name}"
                operation = sensor.readout_resonator.operations.get(op_name, None)
                if operation is None:
                    operation = sensor.readout_resonator.operations["readout"]

                operation.length = optimal_time
                operation.integration_weights_angle -= float(fit_result["iw_angle"])
                pair_ids = {
                    getattr(dp, "id", None),
                    getattr(dp, "name", None),
                } - {None, ""}
                for pair_id in pair_ids:
                    sensor._add_readout_params(pair_id, threshold=float(fit_result["I_threshold"]))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the node results and any recorded state updates."""
    node.save()

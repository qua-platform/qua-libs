# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict
from itertools import combinations

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.experiment import (
    get_dots,
    _make_batchable_list_from_multiplexed,
)
from calibration_utils.charge_state_readout_time_optimization import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_iq_histogram,
    plot_snr_vs_integration_time,
    plot_projected_histogram,
    generate_simulated_dataset,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

# %% {Node initialisation}
description = """
        CHARGE STATE READOUT TIME OPTIMIZATION
This measurement aims to characterise the minimum integration time necessary to achieve SNR = 1 for readout. In this node,
a double-quantum-dot is ramped from charge configuration (1,1) to state (0,2), in order to characterise the charge state readout
fidelity. The aim is to characterise the integration time necessary to reach SNR = 1, for use with PSB readout. The measured IQ
blobs in the IQ state distribution map is analysed, and the SNR is extracted through the relevant axis.


Prerequisites:
    - Having calibrated the resonator to the most sensitive frequency.
    - Having calibrated the relevant sensor dots.
    - Having identified the (1,1) (operation) and (0,2) (readout) points on your charge stability diagram.

State update:
    - The readout pulse length (operation.length) set to the optimal integration time.
    - The integration_weights_angle corrected so (1,1) maps to the higher rotated I value.
    - The readout threshold in the rotated I frame for each dot-pair ID.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="05c_charge_state_readout_time_optimization",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.quantum_dots = ["virtual_dot_3", "virtual_dot_4"]
    node.parameters.num_shots = 10000
    # node.parameters.simulate = True
    # node.parameters.simulation_duration_ns = 200_000
    node.parameters.integration_time_stop = 10000
    node.parameters.integration_time_start = 16
    node.parameters.integration_time_step = 1000
    node.parameters.threshold_SNR = 20
    # node.parameters.wait_time = 0
    node.parameters.use_simulated_data = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.use_simulated_data
)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    node.namespace["quantum_dots"] = quantum_dots = get_dots(node)

    if len(quantum_dots) < 2:
        raise ValueError(
            f"At least 2 Quantum Dots required. Received {len(quantum_dots)}"
        )

    # Find all the existing quantum dot pair names within the list of quantum dots provided.
    quantum_dot_pair_names = [
        pair
        for dot1, dot2 in combinations([k.id for k in quantum_dots], 2)
        if (pair := node.machine.find_quantum_dot_pair(dot1, dot2)) is not None
    ]
    node.log(
        f"Found {len(quantum_dot_pair_names)} quantum dot pairs: {quantum_dot_pair_names}"
    )
    node.namespace["quantum_dot_pairs"] = quantum_dot_pairs = [
        node.machine.get_component(k) for k in quantum_dot_pair_names
    ]

    # Extract the sensors from the quantum dot pairs
    node.namespace["all_sensors"] = all_sensors = {
        pair.name: _make_batchable_list_from_multiplexed(
            pair.sensor_dots, node.parameters.multiplexed
        )
        for pair in quantum_dot_pairs
    }

    # Temporarily set the readout pulse length to cover the full integration time range.
    # measure_accumulated chunks data within the readout pulse, so the pulse must be at
    # least as long as the maximum integration time. Reverted in update_state.
    node.namespace["tracked_resonators"] = []
    unique_sensors = {s.name: s for pair in quantum_dot_pairs for s in pair.sensor_dots}
    for s in unique_sensors.values():
        for pair in quantum_dot_pairs:
            with tracked_updates(
                s.readout_resonator, auto_revert=False, dont_assign_to_none=True
            ) as resonator:
                resonator.operations[f"readout_{pair.name}"].length = (
                    node.parameters.integration_time_stop
                )
                node.namespace["tracked_resonators"].append(resonator)

    # Extract the sweep parameters and axes from the node parameters
    n_reps = node.parameters.num_shots

    integrations_times = np.arange(
        node.parameters.integration_time_start,
        node.parameters.integration_time_stop,
        node.parameters.integration_time_step,
    )
    samples_per_chunk = node.parameters.integration_time_step // 4
    array_size = len(integrations_times)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        # "sensors": xr.DataArray(sensors.get_names()),
        "quantum_dot_pairs": xr.DataArray([qdp.name for qdp in quantum_dot_pairs]),
        "repetition": xr.DataArray(np.arange(n_reps)),
        "integration_time": xr.DataArray(
            np.arange(1, array_size + 1) * samples_per_chunk * 4,
            attrs={"long_name": "integration time", "units": "ns"},
        ),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:

        n = declare(int)
        idx = declare(int)

        I_st_11 = {
            dp.name: {s.name: declare_stream() for s in dp.sensor_dots}
            for dp in quantum_dot_pairs
        }
        Q_st_11 = {
            dp.name: {s.name: declare_stream() for s in dp.sensor_dots}
            for dp in quantum_dot_pairs
        }
        I_st_02 = {
            dp.name: {s.name: declare_stream() for s in dp.sensor_dots}
            for dp in quantum_dot_pairs
        }
        Q_st_02 = {
            dp.name: {s.name: declare_stream() for s in dp.sensor_dots}
            for dp in quantum_dot_pairs
        }
        n_st = declare_stream()
        heralded_and_return_n_loops = getattr(node.parameters, "return_n_loops", False)
        n_loops_st = (
            {dp.name: declare_stream() for dp in quantum_dot_pairs}
            if heralded_and_return_n_loops
            else {}
        )

        for dot_pair in quantum_dot_pairs:
            readout_pulse_name = "readout" + f"_{dot_pair.name}"
            align()
            with for_(n, 0, n < n_reps, n + 1):
                save(n, n_st)

                align()

                # ---------------------------------------------------------
                # Step 1a: Empty - step to empty point (fixed duration)
                # ---------------------------------------------------------
                # dot_pair.empty()
                # Requires the dot pair object to have the empty macro, in addition to the qubits
                # Equivalet step to the lvl_init

                # align()
                # ---------------------------------------------------------
                # Step 2: Initialize - load electron into dots (fixed duration)
                # ---------------------------------------------------------
                n_init = dot_pair.initialize()
                if heralded_and_return_n_loops:
                    save(n_init, n_loops_st[dot_pair.name])
                # Requires the dot pair object to have the initialize macro, in addition to the qubits

                align()
                dot_pair.voltage_sequence.step_to_voltages(
                    voltages={}, duration=node.parameters.integration_time_stop
                )

                for batch in all_sensors[dot_pair.name].batch():
                    for batch_idx, s in batch.items():
                        I_11, Q_11 = s.readout_resonator.measure_accumulated(
                            readout_pulse_name,
                            segment_length=samples_per_chunk,
                        )

                # ---------------------------------------------------------
                # Step 3a: Go to Measure Point
                # ---------------------------------------------------------
                align()
                dot_pair.voltage_sequence.step_to_point(f"{dot_pair.name}_measure")
                align()

                # ---------------------------------------------------------
                # Step 3b: Wait - ensure that it is a singlet state.
                # ---------------------------------------------------------
                dot_pair.voltage_sequence.step_to_voltages(
                    voltages={}, duration=node.parameters.wait_time
                )
                align()

                dot_pair.voltage_sequence.step_to_voltages(
                    voltages={}, duration=node.parameters.integration_time_stop
                )
                for batch in all_sensors[dot_pair.name].batch():
                    for batch_idx, s in batch.items():
                        I_02, Q_02 = s.readout_resonator.measure_accumulated(
                            readout_pulse_name,
                            segment_length=samples_per_chunk,
                        )

                align()

                dot_pair.voltage_sequence.apply_compensation_pulse(
                    return_to_zero=True, go_to_zero=True
                )

                align()

                for batch in all_sensors[dot_pair.name].batch():
                    for batch_idx, s in batch.items():
                        with for_(idx, 0, idx < array_size, idx + 1):
                            save(I_11[idx], I_st_11[dot_pair.name][s.name])
                            save(Q_11[idx], Q_st_11[dot_pair.name][s.name])
                            save(I_02[idx], I_st_02[dot_pair.name][s.name])
                            save(Q_02[idx], Q_st_02[dot_pair.name][s.name])

        with stream_processing():
            n_st.save("n")
            for dp in quantum_dot_pairs:
                for batch in all_sensors[dp.name].batch():
                    for batch_idx, s in batch.items():
                        I_st_11[dp.name][s.name].buffer(array_size).buffer(n_reps).save(
                            f"I_11_{dp.name}_{s.name}"
                        )
                        Q_st_11[dp.name][s.name].buffer(array_size).buffer(n_reps).save(
                            f"Q_11_{dp.name}_{s.name}"
                        )
                        I_st_02[dp.name][s.name].buffer(array_size).buffer(n_reps).save(
                            f"I_02_{dp.name}_{s.name}"
                        )
                        Q_st_02[dp.name][s.name].buffer(array_size).buffer(n_reps).save(
                            f"Q_02_{dp.name}_{s.name}"
                        )
                if heralded_and_return_n_loops:
                    n_loops_st[dp.name].buffer(n_reps).average().save(
                        f"n_loops_{dp.name}"
                    )


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
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.simulate
    or node.parameters.use_simulated_data
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect(timeout = 500)
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        job.wait_until("Done")
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            # progress_counter_with_log(
            #     int(data_fetcher.get("n", 0)),
            #     node.parameters.num_shots,
            #     start_time=data_fetcher.t_start,
            # ,
            #     node=node
            # )
            pass
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
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

    quantum_dot_pairs = node.namespace["quantum_dot_pairs"]
    node.namespace["all_sensors"] = {
        pair.name: _make_batchable_list_from_multiplexed(
            pair.sensor_dots, node.parameters.multiplexed
        )
        for pair in quantum_dot_pairs
    }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        key: ("successful" if fit_result["success"] else "failed")
        for key, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot IQ histograms and SNR vs integration time."""
    fig_iq = plot_iq_histogram(
        node.results["ds_raw"],
        node.namespace["all_sensors"],
        node.namespace["quantum_dot_pairs"],
        fit_results=node.results["fit_results"],
    )
    fig_snr = plot_snr_vs_integration_time(
        node.results["ds_fit"],
        node.namespace["all_sensors"],
        node.namespace["quantum_dot_pairs"],
        fit_results=node.results["fit_results"],
    )
    fig_proj = plot_projected_histogram(
        node.results["ds_raw"],
        node.namespace["all_sensors"],
        node.namespace["quantum_dot_pairs"],
        fit_results=node.results["fit_results"],
    )
    plt.show()
    node.results["figures"] = {
        "iq_histogram": fig_iq,
        "snr_vs_integration_time": fig_snr,
        "projected_histogram": fig_proj,
    }
    annotate_node_figures(node)


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
    node.save()

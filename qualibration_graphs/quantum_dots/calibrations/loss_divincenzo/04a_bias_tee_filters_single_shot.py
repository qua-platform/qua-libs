# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_sensors
from calibration_utils.bias_tee_filters_single_shot import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_all,
    generate_simulated_dataset,
    get_elements,
)
from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot

# %% {Node initialisation}
description = """
BIAS TEE FILTERS CHARACTERIZATION WITH SINGLE SHOT

This sequence characterizes the device-level bias-tee response of one or more swept
elements by applying a single DC step and measuring the sensor response as a function of
time. The readout is performed with sliced demodulation, so the time resolution is set by
``integration_time``. The resulting transient is fitted with an exponential decay and
used to derive the corresponding OPX exponential-filter parameter.


Prerequisites:
    - Having calibrated the resonator to the most sensitive frequency.
    - Having calibrated the relevant sensor dots.
    - Having identified a Coulomb peak on the plunger dot gate voltage.

Datasets:
    - ds_raw: Raw IQ data (I_{el}_{i}, Q_{el}_{i}) vs time for each element/sensor pair.
    - ds_fit: Processed dataset with amplitude_{el}_{i}, fit_{el}_{i}, and
      amplitude_corrected_{el}_{i} variables per element/sensor pair.

Results:
    - fit_results: Dict of {el_name}_{sensor_name} → FitParameters (amplitude,
      time_constant_ns, cutoff_frequency_Hz, offset, success).

Figures:
    - signal_vs_time: IQ amplitude vs time after the step with the fitted exponential
      decay and correction overlay per element/sensor pair.

State update:
    - exponential_filter on each element's OPX output port: [(1.0, tau_ns)].
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="04a_bias_tee_filters_single_shot",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # node.parameters.use_simulated_data = True
    # node.parameters.estimated_bias_tee_tau_ns = 20000  # ns
    # node.parameters.simulate = True
    # node.parameters.sensor_names = ["virtual_sensor_1"]
    # node.parameters.measurement_time = 10000
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

    # ── Experiment parameters (Python side) ──────────────────────────────

    # First extract the relevant elements and sensors given in the node parameters
    node.namespace["elements"], _ = elements, vgs_id = get_elements(node)
    node.namespace["sensors"] = sensors = get_sensors(node)
    num_sensors = len(sensors)

    # Number of shots per single shot measurement
    n_avg = node.parameters.num_shots
    
    # TODO: Can we not average the bias tee value from the multiple sensors? 
    if len(sensors) != 1:
        raise ValueError(
            "04a_bias_tee_filters_single_shot requires exactly one sensor "
            "because it writes one output filter per element."
        )

    # Extend the readout pulse so one acquisition covers the full sliced-demodulation window.
    tracked_resonators = []
    for sensor in sensors:
        with tracked_updates(sensor.readout_resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            resonator.operations["readout"].length = node.parameters.measurement_time
            tracked_resonators.append(resonator)
    node.namespace["tracked_resonators"] = tracked_resonators

    # Set up the sweep. The total measurement time will be split into num_chunks sections
    num_chunks = node.parameters.measurement_time // node.parameters.integration_time
    if num_chunks < 1:
        raise ValueError("measurement_time must be at least integration_time.")

    time_array = (np.arange(num_chunks) + 0.5) * node.parameters.integration_time

    # Wait briefly before measuring so the sliced demodulation starts after
    # the leading edge transient has settled.
    wait_time = node.parameters.wait_time_after_pulse

    # Add margin so the sliced demodulation window stays fully inside the played pulse.
    readout_len = int(np.round(node.parameters.measurement_time * 1.2 / 4) * 4)

    # The swept axes. Buffer along the time axis and average over n_avg
    node.namespace["sweep_axes"] = {
        "time": xr.DataArray(
            time_array,
            attrs={"long_name": "time", "units": "ns"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        seq = node.machine.voltage_sequences[vgs_id]

        # Real-time variables:
        # n      : shot counter
        # ind    : index over the sliced-demodulation chunks
        # I_all/Q_all : per-element, per-sensor arrays of accumulated IQ chunks
        # I_st/Q_st   : streams used to transfer the chunked IQ data back to the host
        n = declare(int)
        n_st = declare_stream()
        ind = declare(int)

        I_all = {
            el.name: [declare(fixed, size=num_chunks) for _ in sensors]
            for el in elements
        }
        Q_all = {
            el.name: [declare(fixed, size=num_chunks) for _ in sensors]
            for el in elements
        }
        I_st_all = {el.name: [declare_stream() for _ in sensors] for el in elements}
        Q_st_all = {el.name: [declare_stream() for _ in sensors] for el in elements}

        # Outer python loop over all the elements in the measurement
        for el in elements:
            I = I_all[el.name]
            Q = Q_all[el.name]
            I_st = I_st_all[el.name]
            Q_st = Q_st_all[el.name]

            # Inner python loop over the sensors involved in this measurement
            for multiplexed_sensors in sensors.batch():

                # ── OUTER QUA LOOP: repeat the measurement n_avg times ───────────────────────
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st) # tell the PC which shot we are on
                    # Optionally wait at the start of the averaging loop
                    if node.parameters.reset_wait_time > 0: 
                        wait(node.parameters.reset_wait_time // 4)

                    # Align everything first
                    align()

                    # Apply one step on the selected element and keep it in place
                    # long enough to cover the entire readout window.
                    seq.step_to_voltages(
                        voltages={el.name: node.parameters.step_amplitude},
                        duration=readout_len + wait_time,
                    )

                    # Measure the response with sliced demodulation. Each chunk integrates over
                    # integration_time and is saved as one sample on the time axis.
                    for i, s in multiplexed_sensors.items():
                        # Extract the readout resonator from the SensorDot
                        rr = s.readout_resonator
                        # Resonator sits idle for the wait time
                        rr.wait(wait_time // 4)
                        # Measure for measurement_time, and slice the measurement into chunks, saving each into a QUA array
                        I[i], Q[i] = rr.measure_sliced(
                            pulse_name="readout",
                            num_segments=num_chunks,
                        )

                    # Return to zero and apply the compensation pulse before the next shot.
                    seq.apply_compensation_pulse(
                        return_to_zero=True, go_to_zero=True
                    )

                    # For each sensor, loop over the number of elements in the chunked array and save them individually. 
                    for i, s in multiplexed_sensors.items():
                        with for_(ind, 0, ind < num_chunks, ind + 1):
                            save(I[i][ind], I_st[i])
                            save(Q[i][ind], Q_st[i])

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for el in elements:
                for i in range(num_sensors):
                    # .buffer(len(time_array)) : group points along the time axis
                    # .average() : group points along the repetitions axis
                    # Result : 2D trace I(time_array, n_avg), Q(time_array, n_avg) per sensor per element
                    I_st_all[el.name][i].buffer(len(time_array)).average().save(
                        f"I_{el.name}_{i + 1}"
                    )
                    Q_st_all[el.name][i].buffer(len(time_array)).average().save(
                        f"Q_{el.name}_{i + 1}"
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
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        # "samples": samples,
    }
    for resonator in node.namespace.pop("tracked_resonators", []):
        resonator.revert_changes()


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.simulate
    or node.parameters.use_simulated_data
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
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset
    for resonator in node.namespace.pop("tracked_resonators", []):
        resonator.revert_changes()


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated IQ data so the full analysis pipeline can run without hardware."""
    node.namespace["elements"] = get_elements(node)
    node.namespace["sensors"] = get_sensors(node)
    if len(node.namespace["sensors"]) != 1:
        raise ValueError(
            "04a_bias_tee_filters_single_shot requires exactly one sensor "
            "because it writes one output filter per element."
        )
    num_chunks = node.parameters.measurement_time // node.parameters.integration_time
    if num_chunks < 1:
        raise ValueError("measurement_time must be at least integration_time.")
    time_array = (np.arange(num_chunks) + 0.5) * node.parameters.integration_time
    node.namespace["sweep_axes"] = {
        "time": xr.DataArray(
            time_array,
            attrs={"long_name": "time", "units": "ns"},
        ),
    }
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    for resonator in node.namespace.pop("tracked_resonators", []):
        resonator.revert_changes()
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    if node.parameters.elements is None:
        node.parameters.elements = list(node.machine.quantum_dots.keys())
    node.namespace["elements"] = [
        node.machine.get_component(el) for el in node.parameters.elements
    ]
    node.namespace["sensors"] = get_sensors(node)
    if len(node.namespace["sensors"]) != 1:
        raise ValueError(
            "04a_bias_tee_filters_single_shot requires exactly one sensor "
            "because it writes one output filter per element."
        )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit an exponential decay to extract the bias tee time constant."""
    ds_processed = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        name: ("successful" if fit_result["success"] else "failed")
        for name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot amplitude vs time with the fitted exponential decay."""
    ds_plot = node.results.get("ds_fit", node.results["ds_raw"])
    node.results["figures"] = plot_all(
        ds_plot,
        node.namespace["elements"],
        node.namespace["sensors"],
        fit_results=node.results.get("fit_results"),
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the OPX output filter parameters on each element's physical channel.

    Sets the exponential compensation filter using the fitted time constant.
    For a bias tee high-pass distortion s(t) = exp(-t/tau), the compensation
    uses exponential_filter = [(1.0, tau_ns)].

    See https://docs.quantum-machines.co/latest/docs/Guides/output_filter/
    """
    for resonator in node.namespace.pop("tracked_resonators", []):
        resonator.revert_changes()

    elements = node.namespace["elements"]
    sensor = node.namespace["sensors"][0]

    with node.record_state_updates():
        for el in elements:
            fit_key = f"{el.name}_{sensor.name}"
            fit_result = node.results["fit_results"].get(fit_key)
            if fit_result is None or not fit_result["success"]:
                node.log(f"Skipping filter update for {el.name}: no successful fit")
                continue

            tau_ns = fit_result["time_constant_ns"]
            port = el.physical_channel.opx_output

            if hasattr(port, "exponential_filter"):
                port.exponential_filter = [(1.0, tau_ns)]
                node.log(
                    f"Updated {el.physical_channel.id} exponential_filter: "
                    f"[(1.0, {tau_ns:.1f})] (τ = {tau_ns:.1f} ns, "
                    f"f_c = {fit_result['cutoff_frequency_Hz']:.1f} Hz)"
                )
            else:
                node.log(
                    f"Port type for {el.physical_channel.id} does not support "
                    f"exponential_filter. Fitted τ = {tau_ns:.1f} ns — "
                    f"configure feedback_filter manually."
                )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the node results and any recorded state updates."""
    node.save()

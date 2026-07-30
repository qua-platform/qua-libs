# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_sensors
from calibration_utils.bias_tee_filters import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_all,
    generate_simulated_dataset,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot

# %% {Node initialisation}
description = """
        BIAS TEE FILTERS CHARACTERIZATION
This measurement aims to characterize the bias tees at the device level, in order to extract the relevant digital
filter coefficients. This calibration is performed by tuning the sensor, and tuning the plunger dot gate voltage
on top of a Coulomb peak. A square wave is sent through the plunger at varying frequencies, and the response of the
sensor is measured. High-frequency square waves pass through to the device undistorted, whereas lower frequency square
waves decay with time. This manifests in the integrated signal measured by the sensor.


Prerequisites:
    - Having calibrated the resonator to the most sensitive frequency.
    - Having calibrated the relevant sensor dots.
    - Having identified a Coulomb peak on the plunger dot gate voltage.

Datasets:
    - ds_raw: Raw IQ data (I_{el}_{i}, Q_{el}_{i}) vs frequency for each element/sensor pair.
    - ds_fit: Processed dataset with amplitude_{el}_{i}, fit_{el}_{i}, and
      amplitude_corrected_{el}_{i} variables per element/sensor pair.

Results:
    - fit_results: Dict of {el_name}_{sensor_name} → FitParameters (amplitude,
      time_constant_ns, cutoff_frequency_Hz, offset, success).

Figures:
    - signal_vs_frequency: IQ amplitude vs square-wave frequency with the fitted
      high-pass curve and correction overlay per element/sensor pair.

State update:
    - exponential_filter on each element's OPX output port: [(1.0, tau_ns)].
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="04b_bias_tee_filters",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # node.parameters.use_simulated_data = True
    # node.parameters.estimated_bias_tee_tau_ns = 320  # ns
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
    n_avg = node.parameters.num_shots
    if node.parameters.elements is None:
        node.parameters.elements = list(node.machine.quantum_dots.keys())
    node.namespace["elements"] = elements = [
        node.machine.get_component(el) for el in node.parameters.elements
    ]
    node.namespace["sensors"] = sensors = get_sensors(node)
    if len(sensors) != 1:
        raise ValueError(
            "04b_bias_tee_filters requires exactly one sensor because it writes "
            "one output filter per element."
        )

    u = unit(coerce_to_integer=True)
    node.namespace["frequencies"] = frequencies = np.arange(
        node.parameters.square_wave_frequency_start_MHz * u.MHz,
        node.parameters.square_wave_frequency_stop_MHz * u.MHz,
        node.parameters.square_wave_frequency_step_MHz * u.MHz,
    )
    if len(frequencies) == 0:
        raise ValueError("Frequency sweep is empty; adjust start/stop/step.")
    node.namespace["sweep_axes"] = {
        "frequency": xr.DataArray(
            frequencies,
            attrs={"long_name": "frequency", "units": "Hz"},
        ),
    }
    num_sensors = len(sensors)

    # TODO: Add a check for this. Possible to perform this node for dots in different gate sets?
    vgs_id = elements[0].voltage_sequence.gate_set.id

    half_periods_ns = np.round(1e9 / (2 * np.flip(frequencies)) / 4).astype(int) * 4
    total_length_ns = int(
        max([s.readout_resonator.operations["readout"].length for s in sensors]) * 5
    )
    num_periods = np.maximum(
        np.ceil(total_length_ns / (half_periods_ns * 2)).astype(int), 1
    )
    amp_val = node.parameters.square_wave_amplitude / 2
    max_periods = int(num_periods.max())
    amp_array = np.tile([amp_val, -amp_val], max_periods).tolist()

    with program() as node.namespace["qua_program"]:
        seq = node.machine.voltage_sequences[vgs_id]

        n = declare(int)
        n_st = declare_stream()

        half_period = declare(int)
        n_periods = declare(int)
        square_wave_idx = declare(int)

        I_all = {el.name: [declare(fixed) for _ in sensors] for el in elements}
        Q_all = {el.name: [declare(fixed) for _ in sensors] for el in elements}
        I_st_all = {el.name: [declare_stream() for _ in sensors] for el in elements}
        Q_st_all = {el.name: [declare_stream() for _ in sensors] for el in elements}

        amp = declare(fixed)
        amp_array_qua = declare(fixed, value=amp_array)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_each_(
                (half_period, n_periods),
                (half_periods_ns.tolist(), num_periods.tolist()),
            ):
                for el in elements:
                    I = I_all[el.name]
                    Q = Q_all[el.name]
                    I_st = I_st_all[el.name]
                    Q_st = Q_st_all[el.name]

                    # Align everything first
                    align()

                    # Dispatch measurements to sensor elements (runs concurrently on different elements)
                    for multiplexed_sensors in sensors.batch():
                        for i, s in multiplexed_sensors.items():
                            rr = s.readout_resonator
                            rr.wait(100)
                            I[i], Q[i] = rr.measure("readout")

                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

                    with for_(
                        square_wave_idx,
                        0,
                        square_wave_idx < n_periods * 2,
                        square_wave_idx + 1,
                    ):
                        assign(amp, amp_array_qua[square_wave_idx])
                        seq.step_to_voltages(
                            voltages={el.name: amp}, duration=half_period
                        )
                    seq.ramp_to_zero(ramp_duration=16)
                    align()
                seq.ramp_to_zero()

        with stream_processing():
            n_st.save("n")
            for el in elements:
                for i in range(num_sensors):
                    I_st_all[el.name][i].buffer(len(frequencies)).average().save(
                        f"I_{el.name}_{i + 1}"
                    )
                    Q_st_all[el.name][i].buffer(len(frequencies)).average().save(
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
                node=node,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated IQ data so the full analysis pipeline can run without hardware."""
    if node.parameters.elements is None:
        node.parameters.elements = list(node.machine.quantum_dots.keys())
    node.namespace["elements"] = [
        node.machine.get_component(el) for el in node.parameters.elements
    ]
    node.namespace["sensors"] = get_sensors(node)
    if len(node.namespace["sensors"]) != 1:
        raise ValueError(
            "04b_bias_tee_filters requires exactly one sensor because it writes "
            "one output filter per element."
        )
    u = unit(coerce_to_integer=True)
    node.namespace["frequencies"] = frequencies = np.arange(
        node.parameters.square_wave_frequency_start_MHz * u.MHz,
        node.parameters.square_wave_frequency_stop_MHz * u.MHz,
        node.parameters.square_wave_frequency_step_MHz * u.MHz,
    )
    if len(frequencies) == 0:
        raise ValueError("Frequency sweep is empty; adjust start/stop/step.")
    node.namespace["sweep_axes"] = {
        "frequency": xr.DataArray(
            frequencies,
            attrs={"long_name": "frequency", "units": "Hz"},
        ),
    }
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
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
            "04b_bias_tee_filters requires exactly one sensor because it writes "
            "one output filter per element."
        )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit the high-pass transfer function to extract the bias tee time constant."""
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
    """Plot amplitude vs frequency with the fitted high-pass transfer function."""
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
    node.save()

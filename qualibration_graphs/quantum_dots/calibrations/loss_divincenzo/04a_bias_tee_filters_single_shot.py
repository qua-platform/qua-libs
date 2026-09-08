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
    - ``ds_raw``: untouched I/Q traces (``I_{el}_{i}``, ``Q_{el}_{i}``) vs time for each
      element/sensor pair.
    - ``ds_fit``: processed dataset with ``amplitude_{el}_{i}``, ``fit_{el}_{i}``, and
      ``amplitude_corrected_{el}_{i}`` variables per element/sensor pair. Used by ``plot_data``.

Results:
    - ``node.results["fit_results"][<element>_<sensor>]``: per-sensor fit result with
      ``success``, ``time_constant_ns``, ``cutoff_frequency_Hz``, ``amplitude``, and ``offset``.
    - ``node.results["fit_results"][<element>]``: aggregated per-element result derived from the
      successful sensor fits and used by ``update_state``.

Figures (``node.results["figures"]``):
    - ``"signal_vs_time"``: IQ amplitude vs time after the step with the fitted exponential decay
      and correction overlay per element/sensor pair.

State update:
    - The exponential filter on each element's OPX output port:
      ``element.physical_channel.opx_output.exponential_filter = [(1.0, tau_ns)]``
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
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 1D time sweep and the QUA pulse sequence."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Elements driven by the voltage step
    elements, vgs_id = get_elements(node)
    node.namespace["elements"] = elements

    # Sensors used for readout (each has its own resonator line)
    node.namespace["sensors"] = sensors = get_sensors(node)
    num_sensors = len(sensors)

    # Ensure that the machine is tracking the integrated voltage, for compensation
    node.machine.reset_voltage_sequence(vgs_id, track_integrated_voltage=True)

    n_avg = node.parameters.num_shots  # number of repetitions averaged for each time trace

    # Extend the readout pulse so one acquisition covers the full sliced-demodulation window.
    tracked_resonators = []
    for sensor in sensors:
        with tracked_updates(sensor.readout_resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            resonator.operations["readout"].length = node.parameters.measurement_time
            tracked_resonators.append(resonator)
    node.namespace["tracked_resonators"] = tracked_resonators

    # Time axis: split the measurement into chunks set by the integration time
    num_chunks = node.parameters.measurement_time // node.parameters.integration_time
    if num_chunks < 1:
        raise ValueError("measurement_time must be at least integration_time.")
    time_array = (np.arange(num_chunks) + 0.5) * node.parameters.integration_time

    # Wait briefly before measuring so the sliced demodulation starts after
    # the leading edge transient has settled.
    wait_time = node.parameters.wait_time_after_pulse

    # Add 20% margin so the sliced demodulation window stays fully inside the played pulse.
    readout_len = int(np.round(node.parameters.measurement_time * 1.2 / 4) * 4)

    # Metadata for data fetching: labels the saved I/Q arrays when results come back from the OPX
    node.namespace["sweep_axes"] = {
        "time": xr.DataArray(
            time_array,
            attrs={"long_name": "time", "units": "ns"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        seq = node.machine.voltage_sequences[vgs_id]

        # Allocate real-time variables on the OPX:
        #   n                    : shot counter
        #   n_st                 : stream reporting shot index to PC (progress bar)
        #   ind                  : index over the sliced-demodulation chunks
        #   I_all/Q_all          : demodulated quadratures per element and sensor
        #   I_st_all/Q_st_all    : stream buffers collecting I/Q chunks before transfer to PC
        n = declare(int)
        n_st = declare_stream()
        ind = declare(int)

        I_all = {el.name: [declare(fixed, size=num_chunks) for _ in sensors] for el in elements}
        Q_all = {el.name: [declare(fixed, size=num_chunks) for _ in sensors] for el in elements}
        I_st_all = {el.name: [declare_output_stream() for _ in sensors] for el in elements}
        Q_st_all = {el.name: [declare_output_stream() for _ in sensors] for el in elements}

        # Elements are handled one-by-one so each step can update a single output filter.
        for el in elements:
            I = I_all[el.name]
            Q = Q_all[el.name]
            I_st = I_st_all[el.name]
            Q_st = Q_st_all[el.name]

            # If several sensors share the same AWG resources, they are grouped into batches
            for multiplexed_sensors in sensors.batch():

                # ── OUTER LOOP: repeat the full time-domain acquisition n_avg times ──
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)  # tell the PC which shot we are on

                    # Optionally wait at the start of the averaging loop
                    if node.parameters.reset_wait_time > 0:
                        wait(node.parameters.reset_wait_time // 4)

                    align()  # sync all channels before starting the step-and-readout sequence

                    # Apply one step on the selected element and keep it in place
                    # long enough to cover the entire readout window.
                    seq.step_to_voltages(
                        voltages={el.name: node.parameters.step_amplitude},
                        duration=readout_len + wait_time,
                    )

                    # Measure the response with sliced demodulation. Each chunk integrates over
                    # integration_time and is saved as one sample on the time axis.
                    for i, sensor in multiplexed_sensors.items():
                        rr = sensor.readout_resonator
                        rr.wait(wait_time // 4)

                        # Play the "readout" pulse and integrate I/Q into chunked QUA arrays
                        I[i], Q[i] = rr.measure_sliced(
                            pulse_name="readout",
                            num_segments=num_chunks,
                        )

                    # Return to zero and apply the compensation pulse before the next shot.
                    seq.apply_compensation_pulse(return_to_zero=True, go_to_zero=True)

                    # Append each time chunk's I/Q to the stream buffer
                    for i, sensor in multiplexed_sensors.items():
                        with for_(ind, 0, ind < num_chunks, ind + 1):
                            save(I[i][ind], I_st[i])
                            save(Q[i][ind], Q_st[i])

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")  # expose shot counter as "n" in the fetched dataset
            for el in elements:
                for i in range(num_sensors):
                    # Each save() is one time chunk.
                    # .buffer(len(time_array)) : group points along the time axis
                    # .average()               : average over all shots (n_avg repetitions)
                    # Result: 1D trace I(time), Q(time) per element/sensor pair
                    I_st_all[el.name][i].buffer(len(time_array)).average().save(f"I_{el.name}_{i + 1}")
                    Q_st_all[el.name][i].buffer(len(time_array)).average().save(f"Q_{el.name}_{i + 1}")


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
    for resonator in node.namespace.pop("tracked_resonators", []):
        resonator.revert_changes()


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
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    for resonator in node.namespace.pop("tracked_resonators", []):
        resonator.revert_changes()
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    elements, _ = get_elements(node)
    node.namespace["elements"] = elements
    # Get the active sensors from the loaded node parameters
    node.namespace["sensors"] = get_sensors(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit an exponential decay to extract the bias tee time constant."""
    node.namespace["ds_processed"] = ds_processed = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        el.name: ("successful" if node.results["fit_results"][el.name]["success"] else "failed")
        for el in node.namespace["elements"]
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot amplitude vs time with the fitted exponential decay."""
    node.results["figures"] = plot_all(
        node.results["ds_fit"],
        node.namespace["elements"],
        node.namespace["sensors"],
        node.results["fit_results"],
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

    with node.record_state_updates():
        for el in elements:
            fit_result = node.results["fit_results"].get(el.name)
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
                    f"f_c = {fit_result['cutoff_frequency_Hz']:.1f} Hz, "
                    f"sensors used = {fit_result['n_sensors_used']})"
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

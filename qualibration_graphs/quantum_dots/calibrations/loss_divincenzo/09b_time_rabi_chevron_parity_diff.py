# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.time_rabi_chevron_parity_diff import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.common_utils.experiment import get_sensors, get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

# %% {Node initialisation}
description = """
        TIME RABI CHEVRON PARITY DIFFERENCE
This sequence performs a 2D chevron measurement with parity difference to characterize qubit coherence and
coupling as a function of both pulse duration and frequency detuning. The measurement involves sweeping both
the duration of a qubit control pulse (typically an X180 pulse) and the frequency detuning while measuring
the parity state before and after the pulse using charge sensing via RF reflectometry or DC current sensing.

The sequence uses voltage gate sequences to navigate through a triangle in voltage space (empty -
initialization - measurement) using OPX channels on the fast lines of the bias-tees. At each combination
of pulse duration and frequency detuning, the parity is measured before (P1) and after (P2) the qubit pulse,
and the parity difference (P_diff) is calculated. When P1 == P2, P_diff = 0; otherwise P_diff = 1.

The 2D chevron pattern in the parity difference signal reveals the qubit coupling strength as a function
of both time and frequency, creating a characteristic chevron shape. This measurement is particularly useful
for characterizing two-qubit gates, understanding the dynamics of coupled quantum dots, and identifying
optimal operating points for qubit control.

Prerequisites:
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the voltage points (empty - initialization - measurement), including sensor the dot bias.
    - Rough guess of the qubit pulse calibration (X180 pulse amplitude and frequency).

State update:
    - The qubit x180 operation duration and frequency.
"""


node = QualibrationNode[Parameters, Quam](
    name="09b_time_rabi_chevron_parity_diff", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    u = unit(coerce_to_integer=True)

    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # The number of averages
    # state_discrimination = node.parameters.use_state_discrimination
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    pulse_durations = np.arange(
        node.parameters.min_wait_time_in_ns,
        node.parameters.max_wait_time_in_ns,
        node.parameters.time_step_in_ns,
    )
    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "pulse_duration": xr.DataArray(pulse_durations, attrs={"long_name": "qubit pulse duration", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        # Declare QUA variables using machine's method
        t = declare(int)
        df = declare(int)
        n = declare(int)

        # Additional variables for pre/post measurement comparison
        # Use int instead of bool so we can average in stream processing
        p1 = declare(int, size=num_qubits)
        p2 = declare(int, size=num_qubits)
        # Streams keyed by qubit name; each qubit appears in exactly one batch.
        p1_st = {qubit.name: declare_stream() for qubit in qubits}
        p2_st = {qubit.name: declare_stream() for qubit in qubits}
        pdiff_st = {qubit.name: declare_stream() for qubit in qubits}
        n_st = declare_stream()

        # Main experiment loop
        for batched_qubits in qubits.batch():
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(df, dfs)):
                    with for_(*from_array(t, pulse_durations // 4)):
                        # ---------------------------------------------------------
                        # Pre-measurement: Check initial state
                        # ---------------------------------------------------------
                        for i, qubit in batched_qubits.items():
                            # Set drive frequency for this iteration
                            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)

                        # ---------------------------------------------------------
                        # Step 1: Empty - step to empty point (fixed duration)
                        # ---------------------------------------------------------
                        align()
                        for i, qubit in batched_qubits.items():
                            qubit.empty()

                        align()
                        for i, qubit in batched_qubits.items():
                            # Measure initial state (includes step_to('readout'))
                            # Cast bool to int for stream averaging
                            assign(p1[i], Cast.to_int(qubit.measure()))

                        # ---------------------------------------------------------
                        # Step 2: Initialize - load electron into dot (variable duration)
                        # ---------------------------------------------------------
                        align()

                        for i, qubit in batched_qubits.items():
                            qubit.initialize(duration=4 * t + node.parameters.gap_wait_time_in_ns)

                        # ---------------------------------------------------------
                        # Step 3: X180 - apply pi pulse
                        # ---------------------------------------------------------
                        for i, qubit in batched_qubits.items():
                            # X180 macro handles X180 pulse
                            qubit.x180(duration=t)

                        # Synchronize before measurement
                        align()

                        # ---------------------------------------------------------
                        # Step 4: Measure - move to PSB and measure
                        # ---------------------------------------------------------
                        for i, qubit in batched_qubits.items():
                            # Measure macro handles step_to('readout') + measurement
                            # Cast bool to int for stream averaging
                            assign(p2[i], Cast.to_int(qubit.measure()))

                        # Synchronize before compensation
                        align()

                        # ---------------------------------------------------------
                        # Step 5: Apply compensation pulse to reset DC bias
                        # ---------------------------------------------------------
                        for i, qubit in batched_qubits.items():
                            qubit.voltage_sequence.apply_compensation_pulse()

                        # ---------------------------------------------------------
                        # Save results
                        # ---------------------------------------------------------
                        for i, qubit in batched_qubits.items():
                            save(p1[i], p1_st[qubit.name])
                            save(p2[i], p2_st[qubit.name])

                            # Calculate state difference
                            with if_(p1[i] == p2[i]):
                                save(0, pdiff_st[qubit.name])
                            with else_():
                                save(1, pdiff_st[qubit.name])

        # Stream processing
        with stream_processing():
            n_st.save("n")

            n_durations = len(pulse_durations)
            n_freqs = len(dfs)

            for qubit in qubits:
                p1_st[qubit.name].buffer(n_freqs, n_durations).average().save(f"p1_{qubit.name}")
                p2_st[qubit.name].buffer(n_freqs, n_durations).average().save(f"p2_{qubit.name}")
                pdiff_st[qubit.name].buffer(n_freqs, n_durations).average().save(f"pdiff_{qubit.name}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
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


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    ds = process_raw_dataset(node.results["ds_raw"], node)
    ds_fit, fit_results = fit_raw_data(ds, node)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubits"],
        node.results.get("fit_results", {}),
    )
    node.results["figure"] = fig


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if not node.results["fit_results"][qubit.name]["success"]:
                continue

            fit_result = node.results["fit_results"][qubit.name]
            qubit.xy.operations[node.parameters.operation].length = fit_result["optimal_duration"]
            qubit.xy.intermediate_frequency = fit_result["optimal_frequency"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

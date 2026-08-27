# %% {Imports}
import logging
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.ramsey_chevron import (
    Parameters,
    analyse_raw_data,
    log_fitted_results,
    plot_all,
)
from qualibration_libs.parameters import get_qubits
from calibration_utils.measurement_utils.measurement_streams import (
    declare_streams,
    save_measurement,
    buffer_streams,
    process_streams,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.sweep import get_idle_times_in_clock_cycles

# %% {Node initialisation}
description = """
        RAMSEY CHEVRON PARITY DIFFERENCE
This sequence performs a Ramsey measurement with parity difference to characterize the qubit detuning and idle time.
The measurement involves sweeping the detuning frequency of the qubit, and performing a sequence of
two π/2 rotations with a swept idle time in between to create a 2D measurement. PSB is used to measure the
parity of the resulting state.

The sequence uses voltage sequences to navigate through a triangle in voltage space (empty -
initialization - measurement) using OPX channels on the fast lines of the bias-tees. At each pulse duration,
the parity is measured before (P1) and after (P2) the qubit manipulation; joint-outcome streams are
saved and reduced to conditional readout expectations in post-processing.

The analysis signal (default: conditional second parity given first parity) reveals Ramsey oscillations as a function of pulse duration and as a function of
pulse detuning, which can be used to extract the qubit coupling strength, coherence time, and optimal pulse parameters.

Prerequisites:
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the voltage points (empty - initialization - measurement).
    - Qubit pulse calibration (X90 pulse amplitude and frequency).

State update:
    - The qubit Larmor frequency.
    - The qubit  T2* (Ramsey) time.
"""


node = QualibrationNode[Parameters, Quam](
    name="11c_ramsey_chevron",
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
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # A class for unit conversion
    u = unit(coerce_to_integer=True)

    # Select which qubits participate in this calibration
    node.namespace["qubits"] = qubits = get_qubits(node)

    # Number of shots per detuning point
    n_avg = node.parameters.num_shots

    # Idle time sweep (in clock cycles of 4ns)
    tau_values = get_idle_times_in_clock_cycles(node.parameters)

    # Construct the array of frequency detunings
    detuning_values = np.arange(
        -node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_step_in_mhz * u.MHz,
    )

    # Register the sweep axes to be added to the dataset when fetching data.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            detuning_values,
            attrs={"long_name": "frequency detuning", "units": "Hz"},
        ),
        "tau": xr.DataArray(
            tau_values * 4,
            attrs={"long_name": "idle time", "units": "ns"},
        ),
    }
    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        # Real-time variables:
        # t      : QUA variable representing the idle time
        # df     : QUA variable representing the frequency detuning
        # n      : shot counter
        # p_post, p_pre : post- / pre-manipulation measurement outcomes
        t = declare(int)
        df = declare(int)
        n = declare(int)
        p_post, p_pre, streams = declare_streams(node, qubits)
        n_st = declare_output_stream()

        # Python loop over the relevant qubits
        for qubit in qubits:
            # Extract the current IF of the qubit's XY component
            intermediate_frequency = qubit.xy.intermediate_frequency

            # ── OUTER LOOP: average over shots ───────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)  # tell the PC which shot we are on

                # ── MIDDLE LOOP: sweep all detuning values ───────────────────────
                with for_(*from_array(df, detuning_values)):

                    # ── INNER LOOP: sweep idle times t ───────────────────────
                    with for_each_(t, tau_values):

                        # Reset the drive frequency and frame before each Ramsey shot, for initialization
                        qubit.xy.update_frequency(intermediate_frequency)
                        reset_frame(qubit.xy.name)

                        # Start with global align
                        align()

                        # Optional pre-measurement parity readout for conditional analysis
                        if node.parameters.parity_measurement:
                            qubit.empty()
                            a1 = qubit.measure()

                        # Perform the initialize macro
                        qubit.initialize()

                        # Update the frequency after the initialize macro
                        qubit.xy.update_frequency(intermediate_frequency + df)

                        # Global align before and after the operate -> wait -> operate -> measure shot
                        align()

                        with strict_timing_():
                            # Apply the Ramsey π/2 – idle – π/2 sequence for this detuning/τ point.
                            qubit.x90()
                            wait(t, qubit.xy.name)
                            qubit.x90()

                        align()

                        # Measure the post-sequence state
                        a2 = qubit.measure()

                        # Just in-case there is any residual output, ramp everything down to zero
                        qubit.voltage_sequence.ramp_to_zero()
                        align()

                        # Cast the bool output of the measurement to an int (0 or 1) for averaging purposes
                        assign(p_post, Cast.to_int(a2))

                        # Optionally cast the pre-parity measurement to an int too
                        if node.parameters.parity_measurement:
                            assign(p_pre, Cast.to_int(a1))

                        # Save the measurements to the relevant streams, dependent on whether node.parameters.parity_measurement
                        save_measurement(node, qubit.name, p_pre, p_post, streams)

            # Reset the XY freqency to the initially stored IF
            qubit.xy.update_frequency(intermediate_frequency)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            n_detuning = len(detuning_values)
            n_tau = len(tau_values)
            for qubit in qubits:
                # Save order per stream: for each qubit, sweep all tau values.
                # n_tau axis - group points along the idle times axis
                # n_detuning - group points along the detuning values axis
                # Result: 2D joint-outcome counts vs (n_detuning, n_tau) per qubit
                buffer_streams(node, qubit.name, streams, n_detuning, n_tau)


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        # "samples": samples,
    }


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            # Display the progress bar
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome streams."""
    node.results["ds_raw"] = process_streams(
        node.results["ds_raw"],
        [q.name for q in node.namespace["qubits"]],
        parity_measurement=node.parameters.parity_measurement,
        sweep_dims=("detuning", "tau"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data to extract frequency offset and T2*."""
    node.results["ds_fit"], fit_results, node.namespace["_fit_results_full"] = analyse_raw_data(
        node.results["ds_raw"], node
    )
    node.results["fit_results"] = fit_results
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""
    fit_with_diag = node.namespace.get("_fit_results_full", node.results.get("fit_results", {}))
    node.results["figures"] = plot_all(
        node.results["ds_raw"],
        node.namespace["qubits"],
        ds_fit=node.results.get("ds_fit"),
        fit_results=fit_with_diag,
        analysis_signal=node.parameters.analysis_signal,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if not node.results["fit_results"][qubit.name]["success"]:
                continue

            fit_result = node.results["fit_results"][qubit.name]
            qubit.larmor_frequency = qubit.larmor_frequency + fit_result["freq_offset"]
            qubit.T2ramsey = fit_result["t2_star"] * 1e-9


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the node results and any recorded state updates."""
    node.save()

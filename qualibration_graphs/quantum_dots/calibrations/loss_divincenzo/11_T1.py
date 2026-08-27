# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.T1 import (
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

# %% {Node initialisation}
description = """
        T1 RELAXATION TIME MEASUREMENT - using standard QUA (pulse > 16ns and 4ns granularity)
The goal of this script is to measure the longitudinal (spin-lattice) relaxation time T1 of the qubit.
T1 characterizes how quickly an excited qubit state decays back to the ground state (thermal equilibrium of ensemble) 
due to energy exchange with the environment. 
This sets the fundamental upper limit for qubit coherence and readout fidelity.

The QUA program is divided into three sections:
    1) step between the initialization point and the operation point using sticky elements (long timescale).
    2) apply a pi pulse to excite the qubit, then wait for a variable idle time (short timescale).
    3) measure the state of the qubit using RF reflectometry via parity readout.

The measurement sequence is:
    - Initialize qubit to ground state (with optional conditional pi pulse for active reset).
    - Apply a pi pulse to flip the spin to the excited state.
    - Wait for variable delay time tau.
    - Measure the qubit state via parity readout.

The excited state population decays exponentially as P(t) = exp(-t/T1), and fitting this decay curve
yields the T1 relaxation time. Longer T1 times indicate better isolation from environmental noise sources
such as phonons, charge noise, and Johnson noise from the measurement circuit.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point).
    - Setting the DC offsets of the external DC voltage source.
    - Connecting the OPX to the fast line of the plunger gates.
    - Having calibrated the initialization and readout point from the charge stability map.
    - Having calibrated the pi pulse parameters (amplitude and duration) from Rabi measurements.

Analysis:
    - Fits P(τ) = offset + A·exp(−τ/T₁) via profiled differential
      evolution (1-D search over T₁, linear solve for offset and A).

Before proceeding to the next node:
    - Verify T₁ is sufficiently long for intended gate sequences.

State updates:
    - qubit.T1
"""


node = QualibrationNode[Parameters, Quam](name="11_T1", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    node.parameters.simulate = True
    node.parameters.tau_min = 200
    node.parameters.simulation_duration_ns = 200000
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Select which qubits participate in this calibration
    node.namespace["qubits"] = qubits = get_qubits(node)

    # Number of shots per detuning point
    n_avg = node.parameters.num_shots

    # Construct the array of tau times
    tau_values = np.arange(
        node.parameters.tau_min,
        node.parameters.tau_max,
        node.parameters.tau_step,
    )

    # Register the sweep axes to be added to the dataset when fetching data.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(tau_values, attrs={"long_name": "idle time", "units": "ns"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        # Real-time variables:
        # t      : QUA variable representing the idle time
        # n      : shot counter
        # p_post, p_pre : post- / pre-manipulation measurement outcomes
        t = declare(int)
        n = declare(int)
        p_post, p_pre, streams = declare_streams(node, qubits)
        n_st = declare_output_stream()

        # Python loop over the relevant qubits
        for qubit in qubits:
            
            # ── OUTER LOOP: average over shots ───────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)  # tell the PC which shot we are on

                # ── INNER LOOP: sweep idle times t ───────────────────────
                with for_(*from_array(t, tau_values // 4)):

                    # Optional pre-measurement parity readout for conditional analysi
                    if node.parameters.parity_measurement:
                        qubit.empty()
                        a1 = qubit.measure()

                    # Perform the initialize macro
                    qubit.initialize(max_loops=1)

                    # Global align before and after the operate -> wait -> measure shot
                    align()

                    # Prepare the excited state and wait for the variable idle time
                    qubit.x180()
                    qubit.idle(t)
                    # Align drive and plunger before readout so the idle window is complete.
                    align(qubit.xy.name, qubit.physical_channel.name)
                    # Measure the post-sequence state
                    a2 = qubit.measure()

                    # Just in-case there is any residual output, ramp everything down to zero
                    qubit.voltage_sequence.ramp_to_zero()

                    # Cast the bool output of the measurement to an int (0 or 1) for averaging purposes
                    assign(p_post, Cast.to_int(a2))

                    align()

                    # Optionally cast the pre-parity measurement to an int too
                    if node.parameters.parity_measurement:
                        assign(p_pre, Cast.to_int(a1))

                    # Save the measurements to the relevant streams, dependent on whether node.parameters.parity_measurement
                    save_measurement(node, qubit.name, p_pre, p_post, streams)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            n_tau = len(tau_values)
            for qubit in qubits:
                # Save order per stream: for each qubit, sweep all tau values.
                # n_tau axis - group points along the idle time axis
                # Result: 2D joint-outcome counts vs (n_tau) per qubit
                buffer_streams(node, qubit.name, streams, n_tau)


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
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


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program, and fetch raw joint-outcome data into ``ds_raw``."""
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
            # Display the progress bar
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


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome streams."""
    node.results["ds_raw"] = process_streams(
        node.results["ds_raw"],
        [q.name for q in node.namespace["qubits"]],
        parity_measurement=node.parameters.parity_measurement,
        sweep_dims=("tau",),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data."""
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
            qubit.T1 = fit_result["T1"] * 1e-9


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the node results and any recorded state updates."""
    node.save()

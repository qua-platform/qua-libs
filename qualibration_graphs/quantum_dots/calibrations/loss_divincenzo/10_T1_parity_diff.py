# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.T1_parity_diff import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.common_utils.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        T1 RELAXATION TIME MEASUREMENT - using standard QUA (pulse > 16ns and 4ns granularity)
The goal of this script is to measure the longitudinal (spin-lattice) relaxation time T1 of the qubit.
T1 characterizes how quickly an excited qubit state decays back to the ground state (thermal equilibrium of ensemble) due to energy exchange
with the environment. This sets the fundamental upper limit for qubit coherence and readout fidelity.

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


node = QualibrationNode[Parameters, Quam](name="10_T1_parity_diff", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1"]
    # node.parameters.num_shots = 10
    # node.parameters.tau_min = 16
    # node.parameters.tau_max = 10000
    # node.parameters.tau_step = 52
    # node.parameters.frequency_min_in_mhz = -0.5
    # node.parameters.frequency_max_in_mhz = 0.525
    # node.parameters.frequency_step_in_mhz = 0.025
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    tau_values = np.arange(
        node.parameters.tau_min,
        node.parameters.tau_max,
        node.parameters.tau_step,
    )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(tau_values, attrs={"long_name": "idle time", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        t = declare(int)
        n = declare(int)

        p1 = declare(int, size=num_qubits)
        p2 = declare(int, size=num_qubits)

        p1_st = {qubit.name: declare_stream() for qubit in qubits}
        p2_st = {qubit.name: declare_stream() for qubit in qubits}
        pdiff_st = {qubit.name: declare_stream() for qubit in qubits}
        n_st = declare_stream()

        for batched_qubits in qubits.batch():
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(t, tau_values // 4)):
                    # ---------------------------------------------------------
                    # Step 1: Empty - step to empty point (fixed duration)
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        qubit.empty()

                    align()
                    for i, qubit in batched_qubits.items():
                        assign(p1[i], Cast.to_int(qubit.measure()))

                    # ---------------------------------------------------------
                    # Step 2: Initialize - load electron into dot (fixed duration)
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        op_length = qubit.macros["x180"].duration
                        qubit.initialize(duration=node.parameters.gap_wait_time_in_ns + op_length + 4 * t)

                    # ---------------------------------------------------------
                    # Step 3: X180 - apply pi pulse and idle
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        qubit.x180()
                        qubit.xy.wait(t)

                    # ---------------------------------------------------------
                    # Step 4: Measure - move to PSB and measure
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        assign(p2[i], Cast.to_int(qubit.measure()))

                    # ---------------------------------------------------------
                    # Step 5: Apply compensation pulse to reset DC bias
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        qubit.voltage_sequence.apply_compensation_pulse()

                    # ---------------------------------------------------------
                    # Save results
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        save(p1[i], p1_st[qubit.name])
                        save(p2[i], p2_st[qubit.name])

                        with if_(p1[i] == p2[i]):
                            save(0, pdiff_st[qubit.name])
                        with else_():
                            save(1, pdiff_st[qubit.name])

        with stream_processing():
            n_st.save("n")

            n_tau = len(tau_values)
            for qubit in qubits:
                p1_st[qubit.name].buffer(n_tau).average().save(f"p1_{qubit.name}")
                p2_st[qubit.name].buffer(n_tau).average().save(f"p2_{qubit.name}")
                pdiff_st[qubit.name].buffer(n_tau).average().save(f"pdiff_{qubit.name}")


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
    """Fit exponential decay P(τ) = offset + A·exp(−τ/T₁) to extract T₁."""
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: {kk: vv for kk, vv in v.items() if kk != "_diag"} for k, v in fit_results.items()}
    node.namespace["_fit_results_full"] = fit_results
    log_fitted_results(node.results["fit_results"], log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot parity-difference vs idle time with exponential fit."""
    fit_with_diag = node.namespace.get("_fit_results_full", node.results.get("fit_results", {}))
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubits"],
        fit_with_diag,
    )
    plt.show()
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
            qubit.T1 = fit_result["T1"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

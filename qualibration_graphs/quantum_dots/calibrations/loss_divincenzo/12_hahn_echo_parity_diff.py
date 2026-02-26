# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.hahn_echo_parity_diff import (
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
        HAHN ECHO (SPIN ECHO) T2 MEASUREMENT - using standard QUA (pulse > 16ns and 4ns granularity)
The goal of this script is to measure the spin-spin relaxation time T2 using the Hahn echo (spin echo) technique.
Unlike the Ramsey experiment which measures T2* (sensitive to low-frequency noise and inhomogeneous broadening),
the Hahn echo refocuses static dephasing, yielding the intrinsic T2 coherence time which is always >= T2*.

The QUA program is divided into three sections:
    1) step between the initialization point and the operation point using sticky elements.
    2) apply the Hahn echo pulse sequence: pi/2 - tau - pi - tau - pi/2.
    3) measure the state of the qubit using RF reflectometry via parity readout.

The Hahn echo sequence works by:
    - First pi/2 pulse (x90): Creates superposition, placing qubit on Bloch sphere equator.
    - First wait period (tau): Qubit dephases due to noise and field inhomogeneities.
    - Pi pulse (x180): Flips the qubit state, reversing the accumulated phase.
    - Second wait period (tau): Previously accumulated phase is undone (refocused).
    - Final pi/2 pulse (x90): Projects the refocused state for measurement.

The echo amplitude decays as exp(-2*tau/T2), where T2 reflects irreversible dephasing from
high-frequency noise that cannot be refocused. This is the simplest dynamical decoupling sequence
and forms the basis for more advanced sequences (CPMG, XY-n) that extend coherence further.

The fitting uses profiled differential evolution: a 1-D global search over T2_echo with
the linear parameters (offset, amplitude) solved analytically at each step via least-squares.

Prerequisites:
    - Having run the Ramsey node to calibrate the qubit frequency and T2*, and the corresponding prerequisites.
    - Having calibrated pi and pi/2 pulse parameters from Rabi measurements.

Before proceeding to the next node:
    - Extract T2 from exponential fit of the echo decay curve.
    - Compare T2 to T2* to assess the contribution of low-frequency noise.
    - Consider dynamical decoupling sequences if longer coherence is needed.

State update:
    - T2echo
"""


node = QualibrationNode[Parameters, Quam](
    name="12_hahn_echo_parity_diff", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]

    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for the Hahn echo sequence.

    Pulse sequence per sweep point:
        empty → measure(p1) → initialise → x90 → wait(τ) → x180 → wait(τ) → x90 → measure(p2)
    """
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
        "tau": xr.DataArray(tau_values, attrs={"long_name": "per-arm idle time", "units": "ns"}),
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
                    # Step 2: Initialise - load electron into dot
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        op_x90_len = qubit.macros["x90"].duration
                        op_x180_len = qubit.macros["x180"].duration
                        echo_duration = 2 * op_x90_len + op_x180_len + 2 * 4 * t
                        qubit.initialize(duration=node.parameters.gap_wait_time_in_ns + echo_duration)

                    # ---------------------------------------------------------
                    # Step 3: Hahn echo — x90 · wait(τ) · x180 · wait(τ) · x90
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        qubit.x90()
                        qubit.xy.wait(t)
                        qubit.x180()
                        qubit.xy.wait(t)
                        qubit.x90()

                    # ---------------------------------------------------------
                    # Step 4: Measure — move to PSB and measure
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
    """Fit an exponential decay to the Hahn echo data for each qubit.

    Stores ``fit_results`` dict and a fitted xarray Dataset ``ds_fit``.
    """
    ds_raw = node.results["ds_raw"]
    qubits = node.namespace["qubits"]

    fit_results = fit_raw_data(ds_raw, qubits)
    node.results["fit_results"] = fit_results

    log_fitted_results(fit_results, node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted Hahn echo data."""
    ds_raw = node.results["ds_raw"]
    fit_results = node.results["fit_results"]
    qubits = node.namespace["qubits"]

    fig = plot_raw_data_with_fit(ds_raw, fit_results, qubits)
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
            qubit.T2echo = fit_result["T2_echo"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

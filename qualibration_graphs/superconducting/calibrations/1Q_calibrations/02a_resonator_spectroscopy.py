# %% {Imports}
from dataclasses import asdict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibration_libs.core import tracked_updates
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from qualibrate import QualibrationNode

from quam_config import Quam

from calibration_utils.resonator_spectroscopy import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_amplitude_with_fit,
    plot_raw_phase,
    plot_detrended_phase,
    plot_iq_circle,
)
from calibration_utils.resonator_spectroscopy.escalation import (
    plan_span_escalation,
    plan_lo_recenter,
)


# %% {Node initialisation}
description = """
        1D RESONATOR SPECTROSCOPY
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies for all the active qubits.
The data is then post-processed to determine the resonator resonance frequency.
This frequency is used to update the readout frequency in the state.

Prerequisites:
    - Having calibrated the IQ mixer/Octave connected to the readout line (node 01a_mixer_calibration.py).
    - Having calibrated the time of flight, offsets, and gains (node 01a_time_of_flight.py).
    - Having initialized the QUAM state parameters for the readout pulse amplitude and duration, and the resonators depletion time.
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The readout frequency: qubit.resonator.f_01 & qubit.resonator.RF_frequency
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="02a_resonator_spectroscopy",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]) -> None:
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["qA1"] # XXX
    pass


# Stash the re-fit overrides before load_from_id() can overwrite node.parameters
_stored_re_fit_resonators = node.parameters.re_fit_resonators
_stored_re_fit_centers_ghz = node.parameters.re_fit_centers_ghz
_stored_re_fit_span_mhz = node.parameters.re_fit_span_mhz


# %% {Program_helpers}
def _setup_sweep_and_program(node: QualibrationNode[Parameters, Quam], span_hz: float) -> None:
    """Build the sweep axes + QUA program for a symmetric ±span/2 scan.

    Shared by the initial pass and the no-dip span-escalation retries, so both
    measure through the identical sequence (only the span differs).
    """
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    qubits = node.namespace["qubits"]
    num_qubits = len(qubits)
    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span_hz / 2, +span_hz / 2, step)
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the readout frequency

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        rr = qubit.resonator
                        # Update the resonator frequencies for all resonators
                        rr.update_frequency(df + rr.intermediate_frequency)
                        # Measure the resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        # wait for the resonator to deplete
                        rr.wait(rr.depletion_time * u.ns)
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


def _execute_and_fetch(node: QualibrationNode[Parameters, Quam]) -> None:
    """Connect, run the prepared QUA program, and store "ds_raw" (shared with the retries)."""
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


def _analyse(node: QualibrationNode[Parameters, Quam]) -> None:
    """Process + fit + log + outcomes (shared between the main pass and the retries)."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]) -> None:
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = get_qubits(node)
    _setup_sweep_and_program(node, node.parameters.frequency_span_in_mhz * u.MHz)


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]) -> None:
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
def execute_qua_program(node: QualibrationNode[Parameters, Quam]) -> None:
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    _execute_and_fetch(node)


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]) -> None:
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset (this overwrites node.parameters)
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Restore the re-fit overrides that were set by the user for this re-analysis run
    node.parameters.re_fit_resonators = _stored_re_fit_resonators
    node.parameters.re_fit_centers_ghz = _stored_re_fit_centers_ghz
    node.parameters.re_fit_span_mhz = _stored_re_fit_span_mhz
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]) -> None:
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    _analyse(node)


# %% {Escalate_no_dip}
@node.run_action(
    skip_if=node.parameters.simulate
    or node.parameters.load_data_id is not None
    or not node.parameters.escalate_on_no_dip
)
def escalate_no_dip(node: QualibrationNode[Parameters, Quam]) -> None:
    """Bring-up span-escalation: when no significant dip was found, re-measure with
    the span doubled (up to ``max_escalation_span_in_mhz``), re-centering each
    affected readout LO when the wider sweep would exceed the IF reach (same
    tracked-LO pattern as node 09b; reverted after the retry ladder).

    All active qubits are re-measured together at the wider span (homogeneous
    dataset -> plots and ds_fit stay consistent); already-found dips simply
    re-fit at the wider span. Every rung is recorded in results["escalation"].
    """
    u = unit(coerce_to_integer=True)
    span = node.parameters.frequency_span_in_mhz * u.MHz
    max_span = node.parameters.max_escalation_span_in_mhz * u.MHz
    audit = []
    tracked = node.namespace.setdefault("tracked_lo_qubits", [])

    while True:
        plan = plan_span_escalation(node.results["fit_results"], span, max_span)
        if not plan["retry"]:
            break
        span = plan["new_span_hz"]
        node.log(
            f"escalation: no dip on {plan['qubits']} -> re-measuring all active qubits "
            f"at span {span / 1e6:.0f} MHz"
        )
        # Re-center LOs where the wider sweep would push the IF out of reach.
        lo_moves = {}
        for q in node.namespace["qubits"]:
            rr = q.resonator
            lo_plan = plan_lo_recenter(
                rf_hz=rr.RF_frequency,
                lo_hz=rr.opx_output.upconverter_frequency,
                span_hz=span,
                band=getattr(rr.opx_output, "band", None),
            )
            if lo_plan["error"]:
                node.log(f"escalation: {q.name}: {lo_plan['error']} — keeping current LO")
                continue
            if lo_plan["shift"]:
                with tracked_updates(q, auto_revert=False, dont_assign_to_none=False) as q_upd:
                    q_upd.resonator.opx_output.upconverter_frequency = lo_plan["new_lo_hz"]
                    tracked.append(q_upd)
                lo_moves[q.name] = lo_plan["new_lo_hz"]
                node.log(f"escalation: {q.name}: LO re-centered to {lo_plan['new_lo_hz'] / 1e9:.4f} GHz")
        audit.append(dict(span_hz=float(span), retried=plan["qubits"], lo_moves=lo_moves))
        _setup_sweep_and_program(node, span)
        _execute_and_fetch(node)
        _analyse(node)

    node.results["escalation"] = audit
    # Revert the LO moves so the state file is untouched by the scan mechanics
    # (the resonator frequency itself is written by update_state from the fit).
    for q_upd in tracked:
        q_upd.revert_changes()
    node.namespace["tracked_lo_qubits"] = []


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]) -> None:
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location.

    Amplitude+fit and detrended phase are the primary diagnostics and always shown.
    Raw phase and the IQ circle are secondary/troubleshooting views, gated behind
    show_raw_phase_plot / show_iq_circle_plot to keep the default output focused.
    """
    figures = {
        "amplitude": plot_raw_amplitude_with_fit(
            node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
        ),
        "detrended_phase": plot_detrended_phase(
            node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
        ),
    }
    if node.parameters.show_raw_phase_plot:
        figures["phase"] = plot_raw_phase(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    if node.parameters.show_iq_circle_plot:
        figures["iq_circle"] = plot_iq_circle(
            node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
        )
    plt.show()
    # Store the generated figures
    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]) -> None:
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            q.resonator.f_01 = float(node.results["fit_results"][q.name]["frequency"])
            q.resonator.RF_frequency = float(node.results["fit_results"][q.name]["frequency"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]) -> None:
    # Safety net: revert any escalation LO shift that survived an aborted retry
    # ladder (e.g. an exception between shift and revert) before saving state.
    for q_upd in node.namespace.get("tracked_lo_qubits", []) or []:
        q_upd.revert_changes()
    node.save()

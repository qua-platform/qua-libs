# pylint: disable=duplicate-code
"""Coupler zero-point calibration."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.coupler_zero_point import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Description}
description = """
        COUPLER ZERO-INTERACTION CALIBRATION
This calibration program determines the flux bias point for tunable couplers that
results in zero effective coupling (g ≈ 0) between pairs of flux-tunable qubits.
This is a crucial step for architectures relying on dynamically tunable coupling
to implement high-fidelity two-qubit gates and isolate qubits during single-qubit operations.

The method performs a 2D sweep of:
    - The coupler flux bias (around its idle point).
    - The qubit control flux (to bring qubit frequencies closer to resonance).

Each point in this sweep involves initializing the control qubit in the excited state and applying
concurrent flux pulses to both the control qubit and the coupler. The resulting excitation in the
target qubit is measured either using state discrimination or IQ integration, depending on the
configuration. The aim is to identify the coupler bias point at which the residual interaction vanishes.

From the data, the optimal coupler flux (yielding minimal excitation transfer) and corresponding
control qubit flux (yielding maximal excitation retention) are extracted. These values are used
to update the coupler’s `decouple_offset` and the estimated qubit `detuning`.

This procedure ensures precise decoupling between qubits during idle or single-qubit operations, helping
mitigate unwanted crosstalk and residual ZZ interactions.

Prerequisites:
    - Coupler hardware model with known calibration structure.
    - Qubit frequencies, flux tuning models (quadratic term at least).
    - Active reset routines for fast initialization (optional).
    - Calibrated readout and XY pulses on the control and target qubits.
    - Initial coupler `decouple_offset` set near its expected g ≈ 0 point.

State update:
    - Coupler zero-point flux: `coupler.decouple_offset`
    - Control qubit detuning: `qubit_pair.detuning`

Additional notes:
    - Supports both simulation and hardware execution.
    - Results are visualized in a 2D map with overlays for idle and calibrated zero-g coupler flux points.
    - If enabled, detuning is also plotted on a secondary axis for interpretation.

This calibration is essential for optimizing gate scheduling, minimizing idling errors,
and preparing the system for entangling gate calibration.
"""

node = QualibrationNode[Parameters, Quam](
    name="30_coupler_zero_point_coarse",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    node.parameters.qubit_pairs = ["coupler_q2_q3"]
    node.parameters.cz_or_iswap = "cz"
    node.parameters.coupler_flux_step = 0.01
    node.parameters.qubit_flux_step = 0.01
    node.parameters.use_state_discrimination = True
    #pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


def _compute_fluxes_qp(node: QualibrationNode[Parameters, Quam], qubit_pairs, fluxes_qubit):
    """Return per-pair qubit flux sweep arrays centred on the estimated detuning."""
    fluxes_qp = {}
    for qp in qubit_pairs:
        if node.parameters.use_saved_detuning:
            est_flux_shift = qp.detuning
        elif node.parameters.cz_or_iswap == "iswap":
            est_flux_shift = np.sqrt(
                -(qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency)
                / qp.qubit_control.freq_vs_flux_01_quad_term
            )
        elif node.parameters.cz_or_iswap == "cz":
            est_flux_shift = np.sqrt(
                -(qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency + qp.qubit_target.anharmonicity)
                / qp.qubit_control.freq_vs_flux_01_quad_term
            )
        else:
            raise ValueError(f"Invalid cz_or_iswap value: {node.parameters.cz_or_iswap}")
        fluxes_qp[qp.name] = fluxes_qubit + est_flux_shift
    return fluxes_qp


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(
    node: QualibrationNode[Parameters, Quam],
):  # pylint: disable=too-many-branches,too-many-statements
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    n_avg = node.parameters.num_shots
    operation = node.parameters.operation

    # Coupler flux sweep relative to the coupler set point
    fluxes_coupler = np.arange(
        node.parameters.coupler_flux_min,
        node.parameters.coupler_flux_max,
        node.parameters.coupler_flux_step,
    )
    # Qubit flux detuning sweep relative to the estimated detuning between the pair
    fluxes_qubit = np.arange(
        -node.parameters.qubit_flux_span / 2,
        node.parameters.qubit_flux_span / 2,
        node.parameters.qubit_flux_step,
    )
    fluxes_qp = _compute_fluxes_qp(node, qubit_pairs, fluxes_qubit)
    node.namespace["fluxes_qp"] = fluxes_qp

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "coupler_flux": xr.DataArray(fluxes_coupler, attrs={"long_name": "coupler flux", "units": "V"}),
        "qubit_flux": xr.DataArray(fluxes_qubit, attrs={"long_name": "qubit flux", "units": "V"}),
    }

    for qp in qubit_pairs:
        node.log(f"Pair {qp.name}: control={qp.qubit_control.name}, target={qp.qubit_target.name}")
        if "coupler_qubit_crosstalk" not in qp.extras:
            node.log(f"No crosstalk compensation for {qp.name}")

    with program() as node.namespace["qua_program"]:
        flux_coupler = declare(fixed)
        flux_qubit = declare(fixed)
        comp_flux_qubit = declare(fixed)

        if node.parameters.use_state_discrimination:
            n = declare(int)
            n_st = declare_output_stream()
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
        else:
            I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
            I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU for pairs in this batch
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            for ii, qp in multiplexed_qubit_pairs.items():
                has_crosstalk = "coupler_qubit_crosstalk" in qp.extras
                crosstalk = qp.extras["coupler_qubit_crosstalk"] if has_crosstalk else 0.0
                wait(1000)
                
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(flux_coupler, fluxes_coupler)):
                        with for_(*from_array(flux_qubit, fluxes_qp[qp.name])):
                            # Qubit initialization
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()

                            # Flux pulse amplitudes (with optional coupler-qubit crosstalk compensation)
                            if has_crosstalk:
                                assign(comp_flux_qubit, flux_qubit + crosstalk * flux_coupler)
                            else:
                                assign(comp_flux_qubit, flux_qubit)
                            qp.align()

                            # State preparation
                            qp.qubit_control.xy.play("x180")
                            if node.parameters.cz_or_iswap == "cz":
                                qp.qubit_target.xy.play("x180")
                            align()
                            # Coupler and qubit flux pulses
                            qp.macros[operation].apply(
                                amplitude_scale_qubit=comp_flux_qubit / qp.macros[operation].flux_pulse_qubit.amplitude,
                                amplitude_scale_coupler=flux_coupler / qp.macros[operation].coupler_flux_pulse.amplitude,
                            )
                            align()
                            wait(20)

                            # Qubit readout
                            if node.parameters.use_state_discrimination:
                                if node.parameters.cz_or_iswap == "cz":
                                    qp.qubit_control.readout_state_gef(state_c[ii])
                                else:
                                    qp.qubit_control.readout_state(state_c[ii])
                                qp.qubit_target.readout_state(state_t[ii])
                                save(state_c[ii], state_c_st[ii])
                                save(state_t[ii], state_t_st[ii])
                            else:
                                qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                                qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                save(I_c[ii], I_c_st[ii])
                                save(Q_c[ii], Q_c_st[ii])
                                save(I_t[ii], I_t_st[ii])
                                save(Q_t[ii], Q_t_st[ii])
            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_c_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(
                        f"state_control{i}"
                    )
                    state_t_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(
                        f"state_target{i}"
                    )
                else:
                    I_c_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_control{i}")
                    Q_c_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_control{i}")
                    I_t_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_target{i}")
                    Q_t_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i}")


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
    """
    Connect to the QOP, execute the QUA program and fetch the raw data
    and store it in a xarray dataset called "ds_raw".
    """
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
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    fluxes_qubit = np.arange(
        -node.parameters.qubit_flux_span / 2,
        node.parameters.qubit_flux_span / 2,
        node.parameters.qubit_flux_step,
    )
    node.namespace["fluxes_qp"] = _compute_fluxes_qp(node, node.namespace["qubit_pairs"], fluxes_qubit)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """
    Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary.
    """
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_pair_name: ("successful" if fit_result["success"] else "failed")
        for qubit_pair_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    figs_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["fit_results"]
    )
    plt.show()
    node.results["figures"] = {
        "raw_fit_target": figs_raw_fit["target"],
        "raw_fit_control": figs_raw_fit["control"],
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            qp.coupler.decouple_offset = node.results["fit_results"][qp.name]["optimal_coupler_flux"]
            qp.detuning = node.results["fit_results"][qp.name]["optimal_qubit_flux"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%

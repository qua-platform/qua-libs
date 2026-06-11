# pylint: disable=duplicate-code
"""Bootstrap CZ / iSWAP flux operating points for a tunable-coupler qubit pair."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_iswap_flux_bootstrap import (
    Parameters,
    estimate_qubit_flux_shift,
    fit_raw_data,
    log_fitted_results,
    QubitRoles,
    verify_moving_qubit,
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
CZ / iSWAP flux operating-point bootstrap

First step in the two-qubit flux calibration chain. Finds coarse moving-qubit and coupler
flux biases for the selected macro (`operation`, e.g. `cz_unipolar` or `iswap_unipolar`)
before finer tuning. Set `cz_or_iswap` for preparation/readout/analysis; set `operation`
to the macro you want to calibrate and update in state.

Method
------
2D sweep of coupler flux (relative to `coupler.decouple_offset`) and moving-qubit flux
(centred via `estimate_qubit_flux_shift`). At each point: prepare |11⟩ (CZ) or |10⟩ (iSWAP),
play ``macros[operation]`` with scaled flux amplitudes, measure both qubits
(state discrimination or IQ).

Prerequisites
-------------
- Tunable coupler pair with ``macros[operation]`` defined in QUAM.
- CZ: GEF readout on both qubits; iSWAP: standard state readout on both qubits.
- Moving qubit `freq_vs_flux_01_quad_term` (09a_ramsey_vs_flux); partner anharmonicity for CZ estimate.
- Coupler sweep spans should be wide enough to cover the idle plateau and the first interaction fringe.

State update
------------
- `coupler.decouple_offset`, `qubit_pair.detuning`, and ``macros[operation]`` pulse amplitudes.

Notes
-----
- ``analysis_debug`` — optional 1D contrast-cut figure in ``plot_data`` (diagnostic only).
- ``analysis_fit_preset`` — ``default`` (normal sweep/SNR), ``noisy`` (poor SNR), or
  ``coarse`` (wide exploratory coupler scan).

"""

node = QualibrationNode[Parameters, Quam](
    name="30_cz_iswap_flux_bootstrap",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # node.parameters.qubit_pairs = ["q1_q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


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
    fluxes_qp = {}
    for qp in qubit_pairs:
        est_flux_shift = estimate_qubit_flux_shift(node.parameters, qp, log_callable=node.log)
        fluxes_qp[qp.name] = fluxes_qubit + est_flux_shift
    node.namespace["fluxes_qp"] = fluxes_qp

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "coupler_flux": xr.DataArray(fluxes_coupler, attrs={"long_name": "coupler flux", "units": "V"}),
        "qubit_flux": xr.DataArray(fluxes_qubit, attrs={"long_name": "qubit flux", "units": "V"}),
    }

    # Verify qp.moving_qubit against recalculation and precompute roles for QUA program loops.
    # Logs a warning and corrects qp.moving_qubit in-memory if they disagree; state is persisted
    # at the end of the node.
    qubit_roles_map = {}
    for qp in qubit_pairs:
        verify_moving_qubit(qp, node.parameters.cz_or_iswap, log_callable=node.log)
        qubit_roles_map[qp.name] = QubitRoles.resolve(qp, node.parameters.cz_or_iswap)
        if "coupler_qubit_crosstalk" not in qp.extras:
            node.log(f"Pair {qp.name}: no crosstalk compensation configured")
    node.namespace["qubit_roles_map"] = qubit_roles_map

    with program() as node.namespace["qua_program"]:
        flux_coupler = declare(fixed)
        flux_qubit = declare(fixed)
        comp_flux_qubit = declare(fixed)

        if node.parameters.use_state_discrimination:
            n = declare(int)
            n_st = declare_output_stream()
            state_mq = [declare(int) for _ in range(num_qubit_pairs)]
            state_sq = [declare(int) for _ in range(num_qubit_pairs)]
            state_mq_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
            state_sq_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
        else:
            I_m, I_m_st, Q_m, Q_m_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
            I_s, I_s_st, Q_s, Q_s_st, _, _ = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU for pairs in this batch
            for qp in multiplexed_qubit_pairs.values():
                qubit_role = qubit_roles_map[qp.name]; mq, sq = qubit_role.moving, qubit_role.stationary
                node.machine.initialize_qpu(target=mq)
                node.machine.initialize_qpu(target=sq)
            align()

            for ii, qp in multiplexed_qubit_pairs.items():
                qubit_role = qubit_roles_map[qp.name]; mq, sq = qubit_role.moving, qubit_role.stationary
                has_crosstalk = "coupler_qubit_crosstalk" in qp.extras
                crosstalk = qp.extras["coupler_qubit_crosstalk"] if has_crosstalk else 0.0
                wait(1000)

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(flux_coupler, fluxes_coupler)):
                        with for_(*from_array(flux_qubit, fluxes_qp[qp.name])):
                            # Qubit initialization
                            mq.reset(node.parameters.reset_type, node.parameters.simulate)
                            sq.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()

                            # Flux pulse amplitudes (with optional coupler-qubit crosstalk compensation)
                            if has_crosstalk:
                                assign(comp_flux_qubit, flux_qubit + crosstalk * flux_coupler)
                            else:
                                assign(comp_flux_qubit, flux_qubit)
                            qp.align()

                            # State preparation
                            mq.xy.play("x180")
                            if node.parameters.cz_or_iswap == "cz":
                                sq.xy.play("x180")
                            align()
                            # Coupler and qubit flux pulses
                            qp.macros[operation].apply(
                                amplitude_scale_qubit=comp_flux_qubit / qp.macros[operation].flux_pulse_qubit.amplitude,
                                amplitude_scale_coupler=flux_coupler
                                / qp.macros[operation].coupler_flux_pulse.amplitude,
                            )
                            align()
                            wait(20)

                            # Qubit readout
                            if node.parameters.use_state_discrimination:
                                if node.parameters.cz_or_iswap == "cz":
                                    mq.readout_state_gef(state_mq[ii])
                                    sq.readout_state_gef(state_sq[ii])
                                else:
                                    mq.readout_state(state_mq[ii])
                                    sq.readout_state(state_sq[ii])
                                save(state_mq[ii], state_mq_st[ii])
                                save(state_sq[ii], state_sq_st[ii])
                            else:
                                mq.resonator.measure("readout", qua_vars=(I_m[ii], Q_m[ii]))
                                sq.resonator.measure("readout", qua_vars=(I_s[ii], Q_s[ii]))
                                save(I_m[ii], I_m_st[ii])
                                save(Q_m[ii], Q_m_st[ii])
                                save(I_s[ii], I_s_st[ii])
                                save(Q_s[ii], Q_s_st[ii])
            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_mq_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(
                        f"state_moving{i}"
                    )
                    state_sq_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(
                        f"state_stationary{i}"
                    )
                else:
                    I_m_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_moving{i}")
                    Q_m_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_moving{i}")
                    I_s_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_stationary{i}")
                    Q_s_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_stationary{i}")


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
    # Register the raw dataset and the sweep/role data needed for reproducible re-analysis
    node.results["ds_raw"] = dataset
    node.results["fluxes_qp"] = node.namespace["fluxes_qp"]
    qubit_roles_map = node.namespace["qubit_roles_map"]
    node.results["qubit_roles"] = {
        name: {field: getattr(role, field).name for field in role._fields}
        for name, role in qubit_roles_map.items()
    }


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.namespace["analysis_debug"] = node.parameters.analysis_debug
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    node.namespace["fluxes_qp"] = {
        name: np.array(arr) for name, arr in node.results["fluxes_qp"].items()
    }
    node.namespace["qubit_roles_map"] = {
        name: QubitRoles(**{field: node.machine.qubits[qname] for field, qname in roles.items()})
        for name, roles in node.results["qubit_roles"].items()
    }


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
    """Plot control/target maps with fit markers; optional contrast-cut debug figure."""
    figs = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        node.results["fit_results"],
        qubit_roles_map=node.namespace["qubit_roles_map"],
        analysis_debug=node.namespace.get("analysis_debug", node.parameters.analysis_debug),
        cz_or_iswap=node.parameters.cz_or_iswap,
    )
    plt.show()
    node.results["figures"] = {
        "stationary": figs["stationary"],
        "moving": figs["moving"],
    }
    if "contrast_debug" in figs:
        node.results["figures"]["contrast_debug"] = figs["contrast_debug"]


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update coupler decouple, qubit detuning, and ``macros[operation]`` flux amplitudes."""

    operation = node.parameters.operation
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            fit_result = node.results["fit_results"][qp.name]
            if not fit_result.get("success", True):
                node.log(f"Skipping state update for {qp.name}: fit flagged unsuccessful.")
                continue
            qp.coupler.decouple_offset = fit_result["optimal_decouple_offset"]
            qp.detuning = fit_result["optimal_qubit_flux"]
            macro = qp.macros[operation]
            macro.flux_pulse_qubit.amplitude = fit_result["optimal_qubit_flux"]
            macro.coupler_flux_pulse.amplitude = (
                fit_result["optimal_cz_coupler_flux_total"] - fit_result["optimal_decouple_offset"]
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%

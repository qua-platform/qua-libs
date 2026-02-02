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
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Initialisation}
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

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="18a_coupler_zero_point",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubit_pairs = ["q1-2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots  # The number of averages

    # Loop parameters
    fluxes_coupler = np.arange(
        node.parameters.coupler_flux_min, node.parameters.coupler_flux_max, node.parameters.coupler_flux_step
    )
    fluxes_qubit = np.arange(
        -node.parameters.qubit_flux_span / 2,
        node.parameters.qubit_flux_span / 2,
        node.parameters.qubit_flux_step,
    )
    fluxes_qp = {}

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "coupler_flux": xr.DataArray(fluxes_coupler, attrs={"long_name": "coupler flux", "units": "V"}),
        "qubit_flux": xr.DataArray(fluxes_qubit, attrs={"long_name": "qubit flux", "units": "V"}),
    }
    for qp in qubit_pairs:
        # estimate the flux shift to get the control qubit to the target qubit frequency
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
        fluxes_qp[qp.name] = fluxes_qubit + est_flux_shift
    node.namespace["fluxes_qp"] = fluxes_qp

    assert (
        node.parameters.pulse_duration_ns % 4 == 0
    ), f"Expected pulse duration to be divisible by 4, got {node.parameters.pulse_duration_ns} ns"
    pulse_duration_ns = node.parameters.pulse_duration_ns
    reset_coupler_bias = False

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        flux_coupler = declare(float)
        flux_qubit = declare(float)
        comp_flux_qubit = declare(float)
        n_st = declare_stream()
        qua_pulse_duration = declare(int, value=pulse_duration_ns // 4)
        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
        else:
            I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
            I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()
        for qubit in node.machine.active_qubits:
            node.machine.initialize_qpu(target=qubit)
            align()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for ii, qp in multiplexed_qubit_pairs.items():
                print("qubit control: %s, qubit target: %s" % (qp.qubit_control.name, qp.qubit_target.name))
                # Bring the active qubits to the minimum frequency point
                if reset_coupler_bias:
                    qp.coupler.set_dc_offset(0.0)
                wait(1000)

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(flux_coupler, fluxes_coupler)):
                        with for_(*from_array(flux_qubit, fluxes_qp[qp.name])):
                            # Qubit initialization
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()
                            if "coupler_qubit_crosstalk" in qp.extras:
                                assign(
                                    comp_flux_qubit, flux_qubit + qp.extras["coupler_qubit_crosstalk"] * flux_coupler
                                )
                            else:
                                print("No crosstalk compensated")
                                assign(comp_flux_qubit, flux_qubit)
                            qp.align()
                            # setting both qubits ot the initial state
                            qp.qubit_control.xy.play("x180")
                            if node.parameters.cz_or_iswap == "cz":
                                qp.qubit_target.xy.play("x180")
                            align()
                            # wait(8)
                            qp.qubit_control.z.play(
                                "const",
                                amplitude_scale=comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude,
                                duration=qua_pulse_duration,
                            )
                            qp.coupler.play(
                                "const",
                                amplitude_scale=flux_coupler / qp.coupler.operations["const"].amplitude,
                                duration=qua_pulse_duration,
                            )
                            align()
                            wait(20)
                            # readout
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
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset."""
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
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    node.namespace["qubits"] = [qp.qubit_control for qp in qubit_pairs] + [qp.qubit_target for qp in qubit_pairs]
    node.namespace["qubit_pairs"] = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    fluxes_qubit = np.arange(
        -node.parameters.qubit_flux_span / 2,
        node.parameters.qubit_flux_span / 2,
        node.parameters.qubit_flux_step,
    )
    fluxes_qp = {}
    for qp in qubit_pairs:
        # estimate the flux shift to get the control qubit to the target qubit frequency
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
        fluxes_qp[qp.name] = fluxes_qubit + est_flux_shift
    node.namespace["fluxes_qp"] = fluxes_qp


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
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
    fig_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["fit_results"]
    )
    plt.show()
    node.results["figures"] = {"raw_fit": fig_raw_fit}


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
    node.save()


# %%

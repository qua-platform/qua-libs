"""Three-tone coupler spectroscopy vs coupler flux — 22b."""

# %%
from __future__ import annotations

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.common_utils.flux_distortions import plan_lo_shift_for_frequency_window
from calibration_utils.three_tone_coupler_spectroscopy_flux_pulse.parameters import (
    resolve_coupler_rf_centers_by_pair,
)
from calibration_utils.three_tone_coupler_spectroscopy_vs_coupler_flux import (
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

description = """
THREE-TONE COUPLER SPECTROSCOPY VS COUPLER FLUX

Maps coupler frequency vs coupler flux-pulse amplitude using three-tone
spectroscopy: a coupler ``const`` pulse sets the bias (relative to decouple_offset),
a strong control-qubit drive sweeps coupler frequency, and a weak target-qubit
probe maps the coupler response.

Prerequisites:
- Target-qubit readout calibrated (IQ blobs / state discrimination)
- Control- and target-qubit XY gates calibrated

Outputs:
- 2D dataset and heatmap of target response vs coupler flux pulse and drive frequency.
"""

node = QualibrationNode[Parameters, Quam](
    name="22b_three_tone_coupler_spectroscopy_vs_coupler_flux",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# %% {Custom_param}
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""


# %% {Create_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create sweep axes and generate the three-tone 2D QUA program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, span / 2, step, dtype=np.int64)
    fluxes = np.linspace(
        node.parameters.coupler_flux_min_in_v,
        node.parameters.coupler_flux_max_in_v,
        node.parameters.num_coupler_flux_points,
    )
    coupler_rf_centers = resolve_coupler_rf_centers_by_pair(
        qubit_pairs,
        node.parameters.rf_frequency_startpoint_in_hz,
        coupler_band=node.parameters.coupler_band,
        idle_detuning_hz=abs(float(node.parameters.coupler_idle_detuning_in_ghz)) * 1e9,
        log_callable=node.log,
    )
    node.namespace["coupler_rf_centers"] = coupler_rf_centers
    coupler_ifs = {
        qp.name: int(
            coupler_rf_centers[qp.name] - qp.qubit_control.xy.opx_output.upconverter_frequency
        )
        for qp in qubit_pairs
    }

    control_qubits = [qp.qubit_control for qp in qubit_pairs]
    lo_plan = plan_lo_shift_for_frequency_window(
        control_qubits,
        dfs,
        if_base_hz=[coupler_ifs[qp.name] for qp in qubit_pairs],
        log_callable=node.log,
    )
    if lo_plan.force_thermal_reset:
        node.parameters.reset_type = "thermal"
    if_update_by_pair = {qp.name: lo_plan.if_update[i] for i, qp in enumerate(qubit_pairs)}
    node.namespace["if_update_by_pair"] = if_update_by_pair
    node.namespace["tracked_qubits"] = lo_plan.tracked_qubits

    node.namespace["dfs"] = dfs
    node.namespace["fluxes"] = fluxes
    node.namespace["coupler_ifs"] = coupler_ifs

    flux_settle = node.parameters.coupler_flux_settle_in_ns // 4

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "flux": xr.DataArray(fluxes, attrs={"long_name": "coupler flux pulse amplitude", "units": "V"}),
        "freq": xr.DataArray(dfs, attrs={"long_name": "coupler drive detuning", "units": "Hz"}),
    }

    with program() as qua_prog:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubit_pairs)]
            state_st = [declare_stream() for _ in range(num_qubit_pairs)]
        flux_bias = declare(fixed)
        df = declare(int)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            control_durations = {}
            target_durations = {}
            for ii, qp in multiplexed_qubit_pairs.items():
                control = qp.qubit_control
                target = qp.qubit_target
                control_durations[ii] = (
                    node.parameters.control_pulse_duration_in_ns * u.ns
                    if node.parameters.control_pulse_duration_in_ns is not None
                    else control.xy.operations[node.parameters.control_drive_operation].length * u.ns
                ) // 4
                target_durations[ii] = (
                    node.parameters.target_pulse_duration_in_ns * u.ns
                    if node.parameters.target_pulse_duration_in_ns is not None
                    else target.xy.operations[node.parameters.target_drive_operation].length * u.ns
                ) // 4

            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(flux_bias, fluxes)):
                    with for_(*from_array(df, dfs)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            control = qp.qubit_control
                            target = qp.qubit_target
                            control.reset(node.parameters.reset_type, node.parameters.simulate)
                            target.reset(node.parameters.reset_type, node.parameters.simulate)
                            control.xy.update_frequency(
                                df + coupler_ifs[qp.name] - if_update_by_pair[qp.name]
                            )
                        align()

                        for ii, qp in multiplexed_qubit_pairs.items():
                            qp.coupler.play(
                                "const",
                                amplitude_scale=flux_bias / qp.coupler.operations["const"].amplitude,
                                duration=flux_settle + control_durations[ii] + target_durations[ii],
                            )
                        for ii, qp in multiplexed_qubit_pairs.items():
                            qp.qubit_control.xy.wait(flux_settle)
                        for ii, qp in multiplexed_qubit_pairs.items():
                            qp.qubit_control.xy.play(
                                node.parameters.control_drive_operation,
                                amplitude_scale=node.parameters.control_pulse_amplitude,
                                duration=control_durations[ii],
                            )
                        for ii, qp in multiplexed_qubit_pairs.items():
                            qp.qubit_target.xy.wait(flux_settle + control_durations[ii])
                        for ii, qp in multiplexed_qubit_pairs.items():
                            qp.qubit_target.xy.play(
                                node.parameters.target_drive_operation,
                                amplitude_scale=node.parameters.target_pulse_amplitude,
                                duration=target_durations[ii],
                            )
                        align()

                        for ii, qp in multiplexed_qubit_pairs.items():
                            target = qp.qubit_target
                            if node.parameters.use_state_discrimination:
                                target.readout_state(state[ii])
                                save(state[ii], state_st[ii])
                            else:
                                target.resonator.measure("readout", qua_vars=(I[ii], Q[ii]))
                                save(I[ii], I_st[ii])
                                save(Q[ii], Q_st[ii])
                        align()

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(dfs)).buffer(len(fluxes)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(dfs)).buffer(len(fluxes)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(dfs)).buffer(len(fluxes)).average().save(f"Q{i + 1}")

    node.namespace["qua_program"] = qua_prog


# %% {Simulate_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}
    plt.show()


# %% {Execute_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())

    if "qubit_pair" in dataset.dims:
        qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
        dataset = dataset.rename({"qubit_pair": "qubit"})
        dataset = dataset.assign_coords(qubit=qubit_pair_names)
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id

    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    if "freq" in node.results["ds_raw"].dims:
        node.namespace["dfs"] = node.results["ds_raw"].freq.values
    if "flux" in node.results["ds_raw"].dims:
        node.namespace["fluxes"] = node.results["ds_raw"].flux.values
    if "qubit_pair" in node.results["ds_raw"].dims:
        qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
        node.results["ds_raw"] = node.results["ds_raw"].rename({"qubit_pair": "qubit"})
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(qubit=qubit_pair_names)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw data and attach absolute frequency coordinates."""
    ds_proc = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_proc"] = ds_proc
    ds_fit, fit_results = fit_raw_data(ds_proc, node)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(fit_results, log_callable=node.log)

    qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
    node.outcomes = {
        pair_name: ("successful" if node.results["fit_results"].get(pair_name, {}).get("success", False) else "failed")
        for pair_name in qubit_pair_names
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot 2D three-tone spectroscopy heatmaps."""
    if "ds_fit" not in node.results:
        return
    fig = plot_raw_data_with_fit(node.results["ds_fit"], node.namespace["qubit_pairs"])
    node.results["figures"] = {"coupler_spectroscopy": fig}
    plt.show()


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results."""
    for qubit in node.namespace.get("tracked_qubits", []):
        try:
            qubit.revert_changes()
        except Exception:
            pass
    node.save()


# %%

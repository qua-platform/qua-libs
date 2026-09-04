"""Ramsey-based coupler flux long distortion characterization and filter design."""

# %%
from __future__ import annotations

import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.coupler_flux_long_distortion_ramsey import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from calibration_utils.common_utils.flux_distortions import update_coupler_filters
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam


# %%
description = """
Long coupler flux distortion characterization using Ramsey interferometry.

This protocol measures the effective coupler flux-line step response by playing a long
coupler flux pulse and probing the qubit frequency at variable delay times using a
Ramsey sequence with frame rotation.  A reference Ramsey measurement (without the long
flux pulse) is acquired for baseline subtraction.

Workflow:
For each qubit pair, sweep the frame rotation and the delay time after the onset of the
coupler flux pulse.  At each delay a Ramsey sequence (x90 – wait – frame_rotation – x90)
is played while a short flux probe pulse with amplitude `ramsey_flux_amplitude` is applied
during the Ramsey wait window.
Analysis: fit frame-rotation oscillations → phase(t); invert the reference
phase-vs-amp curve to get effective coupler flux; form the step response; fit a sum of
decaying exponentials.
State update (optional): write the fitted exponential filter to the coupler's
opx_output.exponential_filter.

Prerequisites
- A valid rotation angle and threshold if using state discrimination.
- Calibrated XY-Coupler delay.
- Calibrated x90 pulse.
- Sensible ``coupler_flux_amplitude_in_v`` / Ramsey probe amp (reference amp sweep covers the phase range).

Outputs and state updates
- Results: processed dataset (including intermediate phase, detuning, flux), fit results,
  and figures are saved under `node.results`.
- If `update_state=True` and fits succeed, the script updates the coupler at
  `coupler.opx_output.exponential_filter` with the cascade coefficients `(A_c, tau_c)`.
REMINDER: Adding digital filters will add a global delay — need to recalibrate IQ blobs
(rotation_angle & ge_threshold) and XY-Coupler delay.
"""

node = QualibrationNode[Parameters, Quam](
    name="21b_coupler_flux_long_distortion_ramsey",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)



# %% {Custom_param}
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # node.parameters.update_state = True
    pass


# Instantiate machine
stored_machine = Quam.load()

# Store fitting fractions set from GUI
loaded_n_exponentials = node.parameters.n_exponentials
stored_gui_update_flag = node.parameters.update_state_from_GUI


# %% {Create_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for Ramsey vs coupler flux measurement."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    node.namespace["qubits"] = measured_qubits

    # Time sweep — linear or log scale
    if node.parameters.time_axis == "linear":
        times = np.arange(
            node.parameters.min_wait_time_in_ns // 4,
            node.parameters.duration_in_ns // 4,
            max(node.parameters.time_step_in_ns, 4) // 4,
            dtype=np.int32,
        )
    else:
        times = np.logspace(
            np.log10(max(node.parameters.min_wait_time_in_ns // 4, 1)),
            np.log10(max(node.parameters.duration_in_ns // 4, 2)),
            max(node.parameters.time_step_num, 3),
            dtype=np.int32,
        )
        times = np.unique(times)

    settle_ns = node.parameters.flux_settle_time_in_ns
    max_delay_ns = int(4 * times.max())
    if settle_ns < max_delay_ns:
        warnings.warn(
            f"flux_settle_time_in_ns={settle_ns} is shorter than the longest delay probed "
            f"({max_delay_ns} ns). Components with tau near the sweep range are only charged to "
            f"1 - exp(-{settle_ns / max_delay_ns:.2f}) = {1 - np.exp(-settle_ns / max_delay_ns):.0%} "
            f"of their amplitude, and the fit caps tau at 20x the pulse length. Either raise "
            f"flux_settle_time_in_ns to >= duration_in_ns or lower duration_in_ns."
        )

    coupler_flux_amp = node.parameters.coupler_flux_amplitude_in_v
    frames = np.arange(0, 1, 1 / node.parameters.num_frame_rotations)
    ref_amplitudes = node.parameters.ramsey_flux_amplitude_in_v + np.linspace(
        -node.parameters.ramsey_flux_sweep_range_in_v,
        node.parameters.ramsey_flux_sweep_range_in_v,
        node.parameters.num_ramsey_flux_points,
    )
    node.namespace["ref_amplitudes"] = ref_amplitudes

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "frame": xr.DataArray(frames, attrs={"long_name": "frame rotation", "units": "2π"}),
        "time": xr.DataArray(4 * times, attrs={"long_name": "Ramsey sequence time", "units": "ns"}),
    }

    with program() as qua_prog:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        I_ref, I_st_ref, Q_ref, Q_st_ref, _, _ = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubit_pairs)]
            state_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_st_ref = [declare_stream() for _ in range(num_qubit_pairs)]

        frame = declare(fixed)
        t_delay = declare(int)
        a = declare(fixed)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)

                # Reference Ramsey (no long coupler flux pulse)
                with for_(*from_array(a, ref_amplitudes)):
                    with for_(*from_array(frame, frames)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist_qubit = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            protagonist_qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()

                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist_qubit = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            protagonist_qubit.xy.play("x90")
                            qp.coupler.align(protagonist_qubit.xy.name)
                            qp.coupler.wait(8)
                            qp.coupler.play(
                                "const",
                                amplitude_scale=a / qp.coupler.operations["const"].amplitude,
                                duration=node.parameters.ramsey_wait_time_in_ns // 4,
                            )
                            qp.coupler.wait(8)
                            qp.coupler.align(protagonist_qubit.xy.name)
                            protagonist_qubit.xy.frame_rotation_2pi(frame)
                            protagonist_qubit.xy.play("x90")
                            align()

                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist_qubit = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            if node.parameters.use_state_discrimination:
                                protagonist_qubit.readout_state(state[ii])
                                save(state[ii], state_st_ref[ii])
                            else:
                                protagonist_qubit.resonator.measure("readout", qua_vars=(I_ref[ii], Q_ref[ii]))
                                save(I_ref[ii], I_st_ref[ii])
                                save(Q_ref[ii], Q_st_ref[ii])
                    align()

                # Signal Ramsey (with long coupler flux pulse + variable delay)
                with for_(*from_array(frame, frames)):
                    with for_each_(t_delay, times):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist_qubit = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            protagonist_qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        # Extra wait to ensure long distortions have fully decayed between repetitions
                        wait(times.max())
                        align()

                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist_qubit = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            qp.coupler.play(
                                "const",
                                amplitude_scale=coupler_flux_amp / qp.coupler.operations["const"].amplitude,
                                duration=node.parameters.flux_settle_time_in_ns // 4,
                            )
                            protagonist_qubit.xy.wait(node.parameters.flux_settle_time_in_ns // 4 + t_delay)
                            protagonist_qubit.xy.play("x90")
                            qp.coupler.align(protagonist_qubit.xy.name)
                            qp.coupler.wait(8)
                            qp.coupler.play(
                                "const",
                                amplitude_scale=node.parameters.ramsey_flux_amplitude_in_v
                                / qp.coupler.operations["const"].amplitude,
                                duration=node.parameters.ramsey_wait_time_in_ns // 4,
                            )
                            qp.coupler.wait(8)
                            qp.coupler.align(protagonist_qubit.xy.name)
                            protagonist_qubit.xy.frame_rotation_2pi(frame)
                            protagonist_qubit.xy.play("x90")
                            align()

                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist_qubit = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            if node.parameters.use_state_discrimination:
                                protagonist_qubit.readout_state(state[ii])
                                save(state[ii], state_st[ii])
                            else:
                                protagonist_qubit.resonator.measure("readout", qua_vars=(I[ii], Q[ii]))
                                save(I[ii], I_st[ii])
                                save(Q[ii], Q_st[ii])
                        align()

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(times)).buffer(node.parameters.num_frame_rotations).average().save(
                        f"state{i + 1}"
                    )
                    state_st_ref[i].buffer(node.parameters.num_frame_rotations).buffer(
                        len(ref_amplitudes)
                    ).average().save(f"state_ref{i + 1}")
                else:
                    I_st[i].buffer(len(times)).buffer(node.parameters.num_frame_rotations).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(times)).buffer(node.parameters.num_frame_rotations).average().save(f"Q{i + 1}")
                    I_st_ref[i].buffer(node.parameters.num_frame_rotations).buffer(len(ref_amplitudes)).average().save(
                        f"I_ref{i + 1}"
                    )
                    Q_st_ref[i].buffer(node.parameters.num_frame_rotations).buffer(len(ref_amplitudes)).average().save(
                        f"Q_ref{i + 1}"
                    )

    node.namespace["qua_program"] = qua_prog


# %% {Simulate_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    debug = False
    if debug:
        from pathlib import Path
        from qm import generate_qua_script
        file_name = Path(__file__).stem
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", 'w') as sourceFile:
            print(generate_qua_script(node.namespace["qua_program"], config), file=sourceFile)
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}
    plt.show()


# %% {Execute_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()

    num_qp = len(node.namespace["sweep_axes"]["qubit_pair"])
    # Key the "qubit" dim by UNIQUE pair names (measured-qubit names repeat when
    # several pairs share a measured target — as on some chips — and would collapse under
    # .sel/groupby, yielding a 2-D slice that crashes _map_phase_to_amplitude). The
    # measured-qubit name is kept as a side coordinate for display only.
    qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
    measured_qubit_names = [q.name for q in node.namespace["measured_qubits"]]
    ref_handle_names = [f"{prefix}{i + 1}" for i in range(num_qp) for prefix in ("state_ref", "I_ref", "Q_ref")]
    frame_coords = node.namespace["sweep_axes"]["frame"].values
    ref_amplitudes = node.namespace["ref_amplitudes"]

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])

        # The XarrayDataFetcher snapshots `ignore_handles` in its __init__
        # (it builds `self.reduce_results_keys` and the underlying
        # `fetching_tool` once). So we must extend the CLASS attribute BEFORE
        # instantiation and restore it afterwards. The reference handles must
        # be excluded here, otherwise their differently-shaped data would
        # cause a shape-mismatch crash inside `update_dataset`.
        original_ignore_handles = list(XarrayDataFetcher.ignore_handles)
        XarrayDataFetcher.ignore_handles = original_ignore_handles + ref_handle_names
        try:
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        finally:
            XarrayDataFetcher.ignore_handles = original_ignore_handles
        # Belt-and-suspenders: also set the instance attribute so older
        # `qualibration_libs` versions that check `self.ignore_handles`
        # at iteration time (instead of snapshotting in __init__) also
        # skip the reference handles.
        data_fetcher.ignore_handles = original_ignore_handles + ref_handle_names

        for dataset in data_fetcher:
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
        node.log(job.execution_report())

        if "qubit_pair" in dataset.dims:
            dataset = dataset.rename({"qubit_pair": "qubit"})
            dataset = dataset.assign_coords(qubit=qubit_pair_names)
            dataset = dataset.assign_coords(measured_qubit_name=("qubit", measured_qubit_names))

        # On the cloud backend (CloudResultHandles), `handle.fetch_all()` only
        # returns data for streams that were registered with `fetching_tool`.
        # Because XarrayDataFetcher excluded the ref handles above, we open a
        # separate `fetching_tool` subscription here, with `wait_for_all`
        # mode so it blocks until the server has all values.
        from qualang_tools.results import fetching_tool as _fetching_tool
        _existing_keys = list(job.result_handles.keys())
        _present_ref_keys = [n for n in ref_handle_names if n in _existing_keys]
        _ref_data_map = {}
        if _present_ref_keys:
            try:
                _ref_fetcher = _fetching_tool(job, _present_ref_keys, mode="wait_for_all")
                _ref_results = _ref_fetcher.fetch_all()
                _ref_data_map = dict(zip(_present_ref_keys, _ref_results))
            except Exception as _rf_e:
                node.log(f"ref fetching_tool failed: {_rf_e}")

        for prefix in ("state_ref", "I_ref", "Q_ref"):
            arrays = []
            for i in range(num_qp):
                _val = _ref_data_map.get(f"{prefix}{i + 1}")
                if _val is not None:
                    arrays.append(np.asarray(_val))
            if arrays:
                dataset[prefix] = xr.DataArray(
                    np.stack(arrays, axis=0),
                    dims=["qubit", "a", "frame"],
                    coords={"qubit": qubit_pair_names, "a": ref_amplitudes, "frame": frame_coords},
                )

    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    node.namespace["qubits"] = measured_qubits

    if "qubit_pair" in node.results["ds_raw"].dims:
        # Unique pair-name coordinate; measured-qubit name kept as a side coordinate
        # (see execute_qua_program for why duplicate measured-target names break .sel).
        qubit_pair_names = [qp.name for qp in qubit_pairs]
        measured_qubit_names = [q.name for q in measured_qubits]
        node.results["ds_raw"] = node.results["ds_raw"].rename({"qubit_pair": "qubit"})
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(qubit=qubit_pair_names)
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(
            measured_qubit_name=("qubit", measured_qubit_names)
        )

    node.parameters.n_exponentials = loaded_n_exponentials
    node.parameters.update_state_from_GUI = stored_gui_update_flag
    if node.parameters.update_state_from_GUI:
        node.machine = stored_machine
        node.parameters.update_state = True
        print("State update from GUI is enabled")


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw dataset, extract Ramsey phase, compute flux response, and fit exponentials."""
    ds_proc = process_raw_dataset(node.results["ds_raw"], node)
    ds_fit, fit_results = fit_raw_data(ds_proc, node)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot flux response and exponential fits (plus debug raw/ref if enabled)."""
    if "ds_fit" not in node.results:
        return
    qubit_pairs = node.namespace.get("qubit_pairs", get_qubit_pairs(node))
    node.results["figures"] = plot_raw_data_with_fit(
        node.results["ds_fit"],
        qubit_pairs,
        node.results["fit_results"],
        ds_raw=node.results.get("ds_raw"),
        debug=node.parameters.debug_plots,
        log_scale=node.parameters.time_axis == "log",
    )
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update IIR filter tabs on coupler if fitting was successful."""
    if not node.parameters.update_state:
        return

    with node.record_state_updates():
        update_coupler_filters(
            node.namespace["qubit_pairs"],
            node.results["fit_results"],
            log_callable=node.log,
        )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results and persist state."""
    node.save()

# %%

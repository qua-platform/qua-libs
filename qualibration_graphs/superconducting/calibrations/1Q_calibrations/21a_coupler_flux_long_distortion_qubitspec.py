"""Pi vs coupler flux calibration for long flux distortion — spectroscopy-first cascade (spectroscopy -> Ramsey -> user-input fallback."""

# %%
from __future__ import annotations

import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.coupler_flux_long_distortion_qubitspec import (
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
from calibration_utils.common_utils.flux_distortions import (
    plan_lo_shift_for_frequency_window,
    resolve_coupler_flux_amplitudes,
    update_coupler_filters,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

description = """
Long cryoscope (π vs coupler flux) calibration.

The coupler flux amplitude and the freq→flux conversion in analysis are derived
per pair from a previously acquired dispersion curve via ``freq_to_flux_source``
(default ``auto``): qubit spectroscopy vs coupler flux (03c), then Ramsey vs
coupler flux (09b), and finally ``coupler_flux_amplitude_in_v``.

Workflow:
For each qubit pair, load the coupler-flux dispersion curve (played-relative
frame: 0 V = the coupler's parked decouple point), look up the qubit frequency
at that decouple point, and compute the coupler flux-pulse AMPLITUDE that achieves
the **signed** ``detuning_in_mhz`` from that reference frequency. Then sweep
XY-drive detuning vs coupler flux-pulse duration to characterise the coupler
flux-line step response.
Analysis: fit a sum of decaying exponentials; optionally write cascade coefficients
to coupler.opx_output.exponential_filter.

Prerequisites
- A valid rotation angle and threshold if using state discrimination
- Calibrated XY-Coupler delay
- A calibrated pi-pulse
- Completed 03c (qubit spectroscopy vs coupler flux) and/or 09b (Ramsey vs coupler
  flux) with ``save_load_id=True`` so run IDs land in qubit extras.

Outputs and state updates
- Results: processed dataset, fit results, and figures are saved under ``node.results``.
- If ``update_state=True`` and fits succeed, updates ``state.json`` for the coupler at
  ``coupler.opx_output.exponential_filter`` with cascade coefficients ``(A_c, tau_c)``.
REMINDER: Adding digital filters will add a global delay --> need to recalibrate IQ
blobs (rotation_angle & ge_threshold) and XY-Coupler delay.
"""

node = QualibrationNode[Parameters, Quam](
    name="21a_coupler_flux_long_distortion_qubitspec",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)



# %% {Custom_param}
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""


# Instantiate machine
stored_machine = Quam.load()

# store fitting parameter and GUI flag set from GUI
loaded_n_exponentials = node.parameters.n_exponentials
stored_gui_update_flag = node.parameters.update_state_from_GUI


# %% {Create_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for pi vs coupler flux measurement."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Extract the measured qubits based on measure_qubit parameter
    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    node.namespace["qubits"] = measured_qubits

    operation_name = node.parameters.operation
    for qubit in measured_qubits:
        if hasattr(qubit.xy.operations, operation_name):
            continue
        warnings.warn(f"Qubit {qubit.name} has no operation '{operation_name}', defaulting to 'x180'")
        operation_name = "x180"

    operation_amp_scale = node.parameters.operation_amplitude_factor or 1.0

    # Signed detuning from the qubit freq at the coupler's decouple_offset
    detuning_hz = node.parameters.detuning_in_mhz * 1e6  # signed
    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    step_hz = node.parameters.frequency_step_in_mhz * 1e6

    resolved = resolve_coupler_flux_amplitudes(
        qubit_pairs,
        measure_qubit=node.parameters.measure_qubit,
        detuning_hz=detuning_hz,
        freq_to_flux_source=node.parameters.freq_to_flux_source,
        fallback_amplitude_v=node.parameters.coupler_flux_amplitude_in_v,
        node=node,
        log_callable=node.log,
    )
    coupler_flux_amps = resolved.amplitudes

    node.namespace["coupler_flux_center"] = coupler_flux_amps[0]
    node.namespace["coupler_flux_amps"] = coupler_flux_amps
    node.results["coupler_flux_center"] = coupler_flux_amps[0]
    node.results["decouple_offsets"] = [qp.coupler.decouple_offset for qp in qubit_pairs]

    # Build dfs array: signed, centered at detuning relative to idle.
    # If freq_at_decouple differs from RF_frequency (idle), dfs must account
    # for that offset so the XY sweep is centered correctly.
    ref_qubit = measured_qubits[0]
    f_dec_ref = resolved.freq_at_decouple[0]
    idle_to_decouple_offset_hz = (
        f_dec_ref - ref_qubit.xy.RF_frequency if f_dec_ref is not None else 0.0
    )
    dfs_center_hz = detuning_hz + idle_to_decouple_offset_hz
    dfs = np.arange(
        dfs_center_hz - span_hz / 2,
        dfs_center_hz + span_hz / 2 + step_hz / 2,
        step_hz,
        dtype=np.int32,
    )

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

    # LO / IF reach: shift upconverter if dfs pushes outside usable MW-FEM window.
    lo_plan = plan_lo_shift_for_frequency_window(measured_qubits, dfs, log_callable=node.log)
    if lo_plan.force_thermal_reset:
        node.parameters.reset_type = "thermal"
    if_update = lo_plan.if_update
    tracked_qubits = lo_plan.tracked_qubits

    for i, (qp, qubit) in enumerate(zip(qubit_pairs, measured_qubits)):
        lo_hz = if_update[i]
        lo_txt = f"LO shifted by {lo_hz / 1e6:.1f} MHz" if lo_hz else "no LO shift"
        node.log(
            f"{qp.name} ({qubit.name}): coupler_flux={resolved.amplitudes[i]:.6f} V "
            f"({resolved.sources[i]}), RF={qubit.xy.RF_frequency / 1e9:.3f} GHz, {lo_txt}"
        )

    node.namespace["if_update"] = if_update
    node.namespace["tracked_qubits"] = tracked_qubits

    # buffer times
    buf_during_op = node.parameters.buffer_during_operation_in_ns // 4
    buf_after_op = node.parameters.buffer_after_operation_in_ns // 4

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "time": xr.DataArray(4 * times, attrs={"long_name": "Flux pulse duration", "units": "ns"}),
    }

    with program() as qua_prog:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubit_pairs)]
            state_st = [declare_stream() for _ in range(num_qubit_pairs)]

        df = declare(int)
        t_delay = declare(int)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
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
                            protagonist_qubit.xy.update_frequency(df + protagonist_qubit.xy.intermediate_frequency - if_update[ii])
                            align()
                            qp.coupler.play(
                                "const",
                                amplitude_scale=coupler_flux_amps[ii] / qp.coupler.operations["const"].amplitude,
                                duration=t_delay + buf_during_op,
                            )
                            protagonist_qubit.xy.wait(t_delay)
                            protagonist_qubit.xy.play(operation_name, amplitude_scale=operation_amp_scale)
                            protagonist_qubit.wait(buf_after_op)
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
                    state_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"Q{i + 1}")

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
    """Connect to the QOP, execute the QUA program and fetch the raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
        node.log(job.execution_report())
    # Rename qubit_pair dimension to qubit for compatibility with analysis functions.
    # Key the "qubit" dim by UNIQUE pair names (measured-qubit names repeat when
    # several pairs share a measured control/target and would collapse under
    # .sel, yielding a 2-D slice that crashes fitting/plotting). The measured-qubit
    # name is kept as a side coordinate for display only.
    if "qubit_pair" in dataset.dims:
        qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
        measured_qubit_names = [q.name for q in node.namespace["measured_qubits"]]
        dataset = dataset.rename({"qubit_pair": "qubit"})
        dataset = dataset.assign_coords(qubit=qubit_pair_names)
        dataset = dataset.assign_coords(measured_qubit_name=("qubit", measured_qubit_names))
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

    # Restore coupler_flux_center to namespace for traceability
    cfc = node.results.get("coupler_flux_center")
    if cfc is not None:
        node.namespace["coupler_flux_center"] = cfc

    # Overwrite the loaded node parameters with the ones defined from the GUI
    node.parameters.n_exponentials = loaded_n_exponentials
    node.parameters.update_state_from_GUI = stored_gui_update_flag
    if node.parameters.update_state_from_GUI:
        node.machine = stored_machine
        node.parameters.update_state = True
        print("State update from GUI is enabled")


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw data and fit exponential components to the flux response data."""
    ds_proc = process_raw_dataset(node.results["ds_raw"], node)
    ds_fit, fit_results = fit_raw_data(ds_proc, node)

    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot center freq, flux response, and exponential fits."""
    if "ds_fit" not in node.results:
        return
    qubit_pairs = node.namespace.get("qubit_pairs", get_qubit_pairs(node))
    measured_qubits = node.namespace.get("measured_qubits", [])
    node.results["figures"] = plot_raw_data_with_fit(
        node.results["ds_fit"],
        qubit_pairs,
        measured_qubits,
        node.results["fit_results"],
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
    """Save all node results, revert tracked qubit changes, and persist state."""
    for qubit in node.namespace.get("tracked_qubits", []):
        try:
            qubit.revert_changes()
        except Exception:
            pass
    node.save()


# %%

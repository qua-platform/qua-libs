"""Pi vs flux calibration for long flux distortion characterization and filter design."""

# %%
from __future__ import annotations

import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.qubit_flux_long_distortion_qubitspec import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
    resolve_flux_amplitudes,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from calibration_utils.common_utils.flux_distortions import (
    plan_lo_shift_for_frequency_window,
    update_filters,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

description = """
Long cryoscope (π vs flux) calibration.

This protocol measures the effective flux-line step response per qubit by sweeping the XY-drive detuning and the Z-flux pulse duration, then extracting the instantaneous qubit frequency versus time.
It then processes and fits the extracted flux response to model it as a sum of decaying exponentials and converts to usable filters.

Workflow:
For each qubit, sweep detuning over the configured span and flux-pulse duration over the configured time axis; play a constant Z pulse with amplitude `flux_amp`, then a chosen XY operation (default π), and measure I/Q or state.
Analysis: convert raw data to volts and extract the center frequency vs detuning at each time; map frequency to flux via the relation chosen by `freq_to_flux_source`; fit a sum of exponentials.
State update (optional): convert the fitted sum-of-exponentials to a cascade representation and write it to the state.json.


Prerequisites
- A valid rotation angle and threshold if using state discrimination
- Calibrated XYZ delay (16a)
- A calibrated pi-pulse
- Each qubit parked at its flux sweetspot. The Z pulse amplitude is derived as a magnitude and `f(Φ)` is assumed symmetric about idle, so either flux direction detunes downwards by the same amount and the side of the parabola is not exposed as a parameter.
- A frequency→voltage relation for each qubit, used both to pick the Z amplitude and to invert the measurement. `freq_to_flux_source="auto"` (default) takes the first available of:
    1. Ramsey vs flux (09a), run ID from `extras['ramsey_vs_flux_calibration_load_id']`
    2. Qubit spectroscopy vs flux (03b), run ID from `extras['qubit_spectroscopy_vs_flux_load_id']`
    3. `freq_vs_flux_01_quad_term` in the state
  Run 09a / 03b with `save_load_id=True` so their run IDs land in the state; no run ID is ever typed into this node. Set `freq_to_flux_source` to `"ramsey"`, `"spectroscopy"` or `"quad_term"` to force one source.

Outputs and state updates
- Results: processed dataset, fit results, and figures are saved under `node.results`.
- If `update_state=True` and fits succeed, the script updates `state.json` per qubit at `z.opx_output.exponential_filter` with the cascade coefficients `(A_c, tau_c)` derived from the fit.
REMINDER: Adding digital filters will add a global delay --> need to recalibrate IQ blobs (rotation_angle & ge_threshold) and (16a) XYZ_delay. It is also worth looking at (09a) Ramsey vs Flux as well
"""

node = QualibrationNode[Parameters, Quam](
    name="17a_qubit_flux_long_distortion_qubitspec",
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

# store n_exponentials set from GUI so the value picked at GUI submission time
# is preserved across the load_from_id() call (which would otherwise overwrite
# node.parameters with whatever the saved run used).
loaded_n_exponentials = node.parameters.n_exponentials
stored_gui_update_flag = node.parameters.update_state_from_GUI


# %% {Create_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for pi vs flux measurement."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    operation_names = {}
    for qubit in qubits:
        if hasattr(qubit.xy.operations, node.parameters.operation):
            operation_names[qubit.name] = node.parameters.operation
        else:
            warnings.warn(f"Qubit {qubit.name} has no operation '{node.parameters.operation}', defaulting to 'x180'")
            operation_names[qubit.name] = "x180"

    operation_amp_scale = node.parameters.operation_amplitude_factor or 1.0

    # Frequency sweep parameters: detuning + span → idle-referenced dfs (negative Hz)
    center_hz = node.parameters.detuning_in_mhz * 1e6
    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    step_hz = node.parameters.frequency_step_in_mhz * 1e6

    dfs = np.arange(
        -center_hz - span_hz / 2,
        -center_hz + span_hz / 2 + step_hz / 2,
        step_hz,
        dtype=np.int32,
    )

    # --- Per-qubit flux_amp derivation via the selected freq→flux relation ---
    resolved = resolve_flux_amplitudes(
        qubits,
        detuning_hz=center_hz,
        freq_to_flux_source=node.parameters.freq_to_flux_source,
        log_callable=node.log,
    )
    flux_amps = resolved.amplitudes

    # Time sweep linear of log scale
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

    # buffer time during operation
    buf_during_op = node.parameters.buffer_during_operation_in_ns // 4
    # buffer time after operation
    buf_after_op = node.parameters.buffer_after_operation_in_ns // 4

    # Sweep axes for data fetcher
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "time": xr.DataArray(4 * times, attrs={"long_name": "Flux pulse duration", "units": "ns"}),
    }

    # LO / IF reach: shift upconverter if dfs pushes outside usable MW-FEM window.
    lo_plan = plan_lo_shift_for_frequency_window(qubits, dfs, log_callable=node.log)
    if lo_plan.force_thermal_reset:
        node.parameters.reset_type = "thermal"
    if_update = lo_plan.if_update
    tracked_qubits = lo_plan.tracked_qubits

    for i, q in enumerate(qubits):
        lo_hz = if_update[i]
        lo_txt = f"LO shifted by {lo_hz / 1e6:.1f} MHz" if lo_hz else "no LO shift"
        node.log(
            f"{q.name}: flux_amp={flux_amps[i]:.6f} V ({resolved.sources[i]}), "
            f"RF={q.xy.RF_frequency / 1e9:.3f} GHz, {lo_txt}"
        )

    node.namespace["if_update"] = if_update
    node.namespace["tracked_qubits"] = tracked_qubits

    with program() as qua_prog:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        df = declare(int)
        t_delay = declare(int)

        for multiplexed_qubits in qubits.batch():
            # Place qubits to their respective flux point
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            # Averaging loop
            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                # Qubit spectroscopy frequency loop
                with for_(*from_array(df, dfs)):
                    # Time delay loop
                    with for_each_(t_delay, times):
                        # Reset the qubits
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            # Extra wait to ensure long distortions have fully decayed between experiment repetitions
                            qubit.wait(times.max())
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            # Step the qubit spectroscopy tone frequency
                            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency - if_update[i])
                            qubit.align()
                            # Play the flux pulse
                            qubit.z.play(
                                "const",
                                amplitude_scale=flux_amps[i] / qubit.z.operations["const"].amplitude,
                                duration=t_delay + buf_during_op,
                            )
                            # Wait for a variable time
                            qubit.xy.wait(t_delay)
                            # Play the qubit spectroscopy pulse
                            qubit.xy.play(operation_names[qubit.name], amplitude_scale=operation_amp_scale)
                            qubit.wait(buf_after_op)
                            qubit.align()

                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i, _ in enumerate(qubits):
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
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_id = node.parameters.load_data_id
    node.load_from_id(load_id)
    node.parameters.load_data_id = load_id
    node.namespace["qubits"] = get_qubits(node)

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
    qubits = node.namespace.get("qubits", get_qubits(node))
    node.results["figures"] = plot_raw_data_with_fit(
        node.results["ds_fit"],
        qubits,
        node.results["fit_results"],
        debug=node.parameters.debug_plots,
        log_scale=node.parameters.time_axis == "log",
    )
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update IIR filter tabs if fitting was successful."""
    if not node.parameters.update_state:
        return
    qubits = node.namespace["qubits"]

    with node.record_state_updates():
        update_filters(
            qubits,
            node.machine,
            node.results["fit_results"],
            update_iir=True,
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

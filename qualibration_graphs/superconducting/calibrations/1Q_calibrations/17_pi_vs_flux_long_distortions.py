"""Pi vs flux calibration for long flux distortion characterization and filter design."""

# %%
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.pi_flux import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_fit,
    process_raw_dataset,
    _to_python,
    _derive_flux_amp,
    _load_spectroscopy_curve,
    _load_ramsey_curve,
    plot_center_freqs,
    plot_flux_response,
    plot_iq_abs_heatmap,
    plot_phase_heatmap,
    plot_spectroscopy_curve,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

_OPPOSITE_BRANCH = {"left": "right", "right": "left"}

description = """
Long cryoscope (π vs flux) calibration.

This protocol measures the effective flux-line step response per qubit by sweeping the XY-drive detuning and the Z-flux pulse duration, then extracting the instantaneous qubit frequency versus time.
It then Processes and fits the extracted flux response to model it as a sum of decaying exponentials and converts to usable filters.

Workflow:
For each qubit, sweep detuning over the configured span and flux-pulse duration over the configured time axis; play a constant Z pulse with amplitude `flux_amp`, then a chosen XY operation (default π), and measure I/Q or state.
Analysis: convert raw data to volts and extract the center frequency vs detuning at each time; derive the flux response using each qubit’s `freq_vs_flux_01_quad_term`; fit a sum of exponentials and determine the best components and DC term.
State update (optional): convert the fitted sum-of-exponentials to a cascade representation and write it to the state.json.


Prerequisites
- A valid rotation angle and threshold if using state discrimination
- Calibrated XYZ delay
- A calibrated pi-pulse
- Each qubit must have a known `freq_vs_flux_01_quad_term` stored in the state (obtained via (09)Ramsey vs flux calibration).

Outputs and state updates
- Results: processed dataset, fit results, and figures are saved under `node.results`.
- If `update_state=True` and fits succeed, the script updates `state.json` per qubit at `z.opx_output.exponential_filter` with the cascade coefficients `(A_c, tau_c)` derived from the fit.
REMINDER: Adding digital filters will add a global delay --> need to recalibrate IQ blobs (rotation_angle & ge_threshold) and (15)XYZ_delay. It is also worth looking at (09) Ramsey vs Flux as well
"""

node = QualibrationNode[Parameters, Quam](
    name="17_pi_vs_flux_long_distortions",
    description=description,
    parameters=Parameters(),
)


# %% {Custom_param}
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""


# Instantiate machine
node.machine = stored_machine = Quam.load()

# store n_exponentials set from GUI so the value picked at GUI submission time
# is preserved across the load_from_id() call (which would otherwise overwrite
# node.parameters with whatever the saved run used).
loaded_n_exponentials = node.parameters.n_exponentials
stored_gui_update_flag = node.parameters.update_state_from_GUI


# %% {Create_qua_program}
# pylint: disable=too-many-branches,too-many-statements
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for pi vs flux measurement."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    operation_name = node.parameters.operation
    for qubit in qubits:
        if hasattr(qubit.xy.operations, operation_name):
            continue
        warnings.warn(f"Qubit {qubit.name} has no operation '{operation_name}', defaulting to 'x180'")
        operation_name = "x180"

    operation_amp_scale = node.parameters.operation_amplitude_factor or 1.0

    # Frequency sweep parameters: detuning + span → idle-referenced dfs (negative Hz)
    center_hz = node.parameters.detuning_in_mhz * 1e6
    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    step_hz = node.parameters.frequency_step_in_mhz * 1e6
    flux_branch = node.parameters.flux_branch

    dfs = np.arange(
        -center_hz - span_hz / 2,
        -center_hz + span_hz / 2 + step_hz / 2,
        step_hz,
        dtype=np.int32,
    )

    # --- Per-qubit flux_amp derivation with cascading fallback ---
    use_spec = getattr(node.parameters, "use_spectroscopy_data", False)
    spec_run_id = getattr(node.parameters, "spectroscopy_run_id", None)
    use_ramsey = getattr(node.parameters, "use_ramsey_data", False)

    flux_amps = []
    effective_branch = flux_branch
    for q in qubits:
        flux_amp_q = None
        source_label = None

        # --- Try spectroscopy curve ---
        spec_curve = None
        if use_spec and spec_run_id is not None:
            spec_curve = _load_spectroscopy_curve(spec_run_id, q.name, q.xy.RF_frequency)

        if spec_curve is not None:
            flux_amp_q = _derive_flux_amp(center_hz, q.xy.RF_frequency, spec_curve, flux_branch)
            if flux_amp_q is not None:
                source_label = f"spectroscopy #{spec_run_id} ({flux_branch})"
            else:
                opp = _OPPOSITE_BRANCH[flux_branch]
                warnings.warn(
                    f"{q.name}: target detuning not found on {flux_branch} branch "
                    f"of spectroscopy curve, trying {opp} branch"
                )
                flux_amp_q = _derive_flux_amp(center_hz, q.xy.RF_frequency, spec_curve, opp)
                if flux_amp_q is not None:
                    source_label = f"spectroscopy #{spec_run_id} ({opp}, fallback)"
                    effective_branch = opp

        # --- Try Ramsey curve ---
        ramsey_run_id_q = node.parameters.ramsey_run_id or (
            q.extras.get("ramsey_vs_flux_calibration_load_id") if hasattr(q, "extras") else None
        )
        if flux_amp_q is None and use_ramsey and ramsey_run_id_q is not None:
            ramsey_curve = _load_ramsey_curve(ramsey_run_id_q, q.name, q.xy.RF_frequency)
            if ramsey_curve is not None:
                flux_amp_q = _derive_flux_amp(center_hz, q.xy.RF_frequency, ramsey_curve, flux_branch)
                if flux_amp_q is not None:
                    source_label = f"Ramsey #{ramsey_run_id_q} ({flux_branch})"
                else:
                    opp = _OPPOSITE_BRANCH[flux_branch]
                    warnings.warn(
                        f"{q.name}: target detuning not found on {flux_branch} branch "
                        f"of Ramsey curve, trying {opp} branch"
                    )
                    flux_amp_q = _derive_flux_amp(center_hz, q.xy.RF_frequency, ramsey_curve, opp)
                    if flux_amp_q is not None:
                        source_label = f"Ramsey #{ramsey_run_id_q} ({opp}, fallback)"
                        effective_branch = opp

        # --- quad_term fallback ---
        if flux_amp_q is None:
            qt = getattr(q, "freq_vs_flux_01_quad_term", None)
            # pylint: disable-next=use-implicit-booleaness-not-comparison-to-zero
            if qt is not None and qt != 0 and np.isfinite(qt):
                sign = 1.0 if flux_branch == "right" else -1.0
                flux_amp_q = sign * float(np.sqrt(center_hz / abs(qt)))
                source_label = f"quad_term={qt:.3e}"

        if flux_amp_q is None:
            raise ValueError(
                f"Cannot derive flux_amp for {q.name}: no curve available and "
                f"freq_vs_flux_01_quad_term is missing or zero."
            )

        # Validation
        if abs(flux_amp_q) > 0.5:
            warnings.warn(
                f"{q.name}: derived flux_amp={flux_amp_q:.4f} V exceeds 0.5 V. "
                f"Verify detuning_in_mhz={node.parameters.detuning_in_mhz} is correct."
            )

        flux_amps.append(flux_amp_q)
        print(f"  {q.name}: flux_amp={flux_amp_q:.6f} V (source: {source_label})")

    # Store branch hint for analysis: sentinel guarantees correct use_upper_branch
    # comparison for all qubits regardless of their individual idle_flux values.
    # "right" → use_upper_branch=True, "left" → use_upper_branch=False
    node.namespace["flux_amp_for_detuning"] = 999.0 if effective_branch == "right" else -999.0

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

    # Track LO updates: shift upconverter if lowest dfs point pushes IF below -400 MHz
    tracked_qubits = []
    if_update = []

    for i, q in enumerate(qubits):
        detuning_hz = int((dfs.min() + dfs.max()) / 2)
        if (q.xy.intermediate_frequency + int(dfs.min())) < -400e6:
            node.parameters.reset_type = "thermal"  # Active reset will not work if the LO is changed
            warnings.warn(
                "Qubit LO has been changed to reach desired detuning, "
                "active reset will not work. Reset type changed to thermal."
            )
            if_update.append(detuning_hz)
            # track the LO and IF changes to revert later
            with tracked_updates(q, auto_revert=False, dont_assign_to_none=False) as q_upd:
                lo_frequency = q_upd.xy.opx_output.upconverter_frequency + detuning_hz
                if (q_upd.xy.opx_output.band == 3) and (lo_frequency < 6.5e9):
                    raise ValueError("Requested detuning is too large for the given MW FEM band")
                if (q_upd.xy.opx_output.band == 2) and (lo_frequency < 4.5e9):
                    raise ValueError("Requested detuning is too large for the given MW FEM band")
                print(f"Updating {q_upd.name} LO to {lo_frequency}")
                q_upd.xy.opx_output.upconverter_frequency = lo_frequency
                q_upd.xy.RF_frequency += detuning_hz
                tracked_qubits.append(q_upd)
        else:
            if_update.append(0)
            print(f"No LO update needed for {q.name}")

        print("======= SWEEP AXES =======")
        print(f"qubit: {q.name}")
        print(f"flux_amp: {flux_amps[i]:.6f} V (branch={flux_branch})")
        print(f"detuning_in_mhz: {node.parameters.detuning_in_mhz}")
        print(f"frequency_span_in_mhz: {node.parameters.frequency_span_in_mhz}")
        print(f"RF_frequency: {np.round(q.xy.RF_frequency*1e-9, 3)} GHz")
        print(f"dfs: {np.round(dfs[:2]*1e-6, 3)} MHz...{np.round(dfs[-2:]*1e-6, 3)} MHz")
        print(f"if_update: {if_update}")
        print(
            f"freqs: {np.round((dfs[:2]+q.xy.RF_frequency)*1e-9, 3)} GHz..."
            f"{np.round((dfs[-2:]+q.xy.RF_frequency)*1e-9, 3)} GHz"
        )
        print(f"times: {np.round(times[:2]*4, 3)} ns...{np.round(times[-2:]*4, 3)} ns")

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
                            qubit.xy.play(operation_name, amplitude_scale=operation_amp_scale)
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
    ds_in = process_raw_dataset(node.results["ds_raw"], node)
    ds, fit_results = fit_raw_data(ds_in, node)

    node.results["ds_fit"] = ds
    node.results["fit_results"] = {k: _to_python(v) for k, v in fit_results.items()}
    log_fitted_results(fit_results, log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot raw spectroscopy, center freq, flux response, and exponential fits."""
    if "ds_fit" not in node.results:
        return
    ds = node.results["ds_fit"]
    qubits = node.namespace.get("qubits", get_qubits(node))
    fit_results = node.results["fit_results"]

    figures = {
        "iq_abs_linear": plot_iq_abs_heatmap(ds, qubits, log_scale=False),
        "iq_abs_log": plot_iq_abs_heatmap(ds, qubits, log_scale=True),
        "phase": plot_phase_heatmap(ds, qubits),
        "center_freq_linear": plot_center_freqs(ds, qubits, log_scale=False),
        "center_freq_log": plot_center_freqs(ds, qubits, log_scale=True),
        "flux_response_linear": plot_flux_response(ds, qubits, log_scale=False),
        "flux_response_log": plot_flux_response(ds, qubits, log_scale=True),
        "fitted_data": plot_fit(ds, qubits, fit_results),
    }
    spec_fig = plot_spectroscopy_curve(ds, qubits)
    if spec_fig is not None:
        figures["spectroscopy_curve"] = spec_fig

    # Ramsey vs Z-flux reference curve (path 2)
    if node.parameters.use_ramsey_data:
        n_qubits = len(qubits)
        fig_r, axes = plt.subplots(1, n_qubits, figsize=(5 * n_qubits, 4), squeeze=False)
        used_run_ids = set()
        for ax, qubit in zip(axes[0], qubits):
            ramsey_run_id_q = node.parameters.ramsey_run_id or (
                qubit.extras.get("ramsey_vs_flux_calibration_load_id") if hasattr(qubit, "extras") else None
            )
            if ramsey_run_id_q is None:
                continue
            used_run_ids.add(int(ramsey_run_id_q))
            curve = _load_ramsey_curve(ramsey_run_id_q, qubit.name, qubit.xy.RF_frequency)
            if curve is not None:
                flux_bias, qubit_freq = curve
                ax.plot(flux_bias, np.array(qubit_freq) / 1e9, marker=".", linestyle="-")
                ax.set_xlabel("Z flux (V)")
                ax.set_ylabel("Qubit frequency (GHz)")
                ax.set_title(qubit.name)
        if used_run_ids:
            runs_txt = ", ".join(str(rid) for rid in sorted(used_run_ids))
            fig_r.suptitle(f"Ramsey vs Z-flux — run(s) from state: {runs_txt}")
        else:
            fig_r.suptitle("Ramsey vs Z-flux — no run IDs found in qubit extras")
        fig_r.tight_layout()
        figures["ramsey_curve"] = fig_r

    node.results["figures"] = figures
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update IIR filter tabs if fitting was successful."""
    if not node.parameters.update_state:
        return
    qubits = node.namespace["qubits"]

    for q in qubits:
        z_out = node.machine.qubits[q.name].z.opx_output
        if z_out.exponential_filter is None:
            z_out.exponential_filter = []

    with node.record_state_updates():
        for q in qubits:
            res = node.results["fit_results"][q.name]
            # Support dict or dataclass
            fit_success = res["fit_successful"]
            if not fit_success:
                continue
            best_a_dc = res["a_dc"]
            components = res["a_tau_tuple"]
            A_list = [amp / best_a_dc for amp, _ in components]
            tau_list = [tau for _, tau in components]
            node.machine.qubits[q.name].z.opx_output.exponential_filter.extend(list(zip(A_list, tau_list)))
            print(f"Updated {q.name} filter to: {node.machine.qubits[q.name].z.opx_output.exponential_filter}")


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

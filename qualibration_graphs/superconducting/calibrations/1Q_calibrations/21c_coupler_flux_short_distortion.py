"""Cryoscope calibration for coupler flux line — short-time IIR/FIR correction."""

# %%
from __future__ import annotations

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.coupler_flux_short_distortion import (
    Parameters,
    baked_coupler_waveform,
    fit_fir_data,
    fit_raw_data,
    log_fitted_results,
    plot_fir_figures,
    plot_raw_data,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from calibration_utils.common_utils.flux_distortions import (
    resolve_coupler_flux_amplitudes,
    update_coupler_filters,
)
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam


description = """
Short coupler flux distortion (cryoscope path).

Same idea as **17c** — see ``17c_qubit_flux_short_distortion.py`` for the physics.

Workflow:
For each qubit pair, resolve a coupler flux amplitude that places the measured qubit at
``detuning_in_mhz`` from its decouple-point frequency via ``freq_to_flux_source``, bake
1 ns flux segments, sweep pulse duration and frame, and reconstruct phase → frequency →
flux step response. Fit a sum of exponentials (IIR); optionally run FIR feedforward
analysis (``use_fir``).

Prerequisites:
- A valid rotation angle and threshold if using state discrimination
- Calibrated XY–coupler delay on the measured qubit (``measure_qubit``)
- Calibrated x90 pulse
- Completed 03c and/or 09b with ``save_load_id=True``

Outputs and state updates
- Results: processed dataset, fit results, and figures are saved under ``node.results``.
- Set ``update_state=True`` to write filters; use ``update_iir`` and/or ``update_fir``
  to choose which filters are committed.
- Re-load a prior run with ``load_data_id``, tune fit settings in the GUI, then set
  ``update_state_from_GUI=True`` to commit without re-acquiring data.
REMINDER: Adding digital filters will add a global delay --> need to recalibrate IQ
blobs (rotation_angle & ge_threshold) and XY–coupler delay.
"""

node = QualibrationNode[Parameters, Quam](
    name="21c_coupler_flux_short_distortion",
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

loaded_n_exponentials = node.parameters.n_exponentials
stored_gui_update_flag = node.parameters.update_state_from_GUI
loaded_fractions = node.parameters.exponential_fit_time_fractions
stored_use_fir = node.parameters.use_fir
stored_update_iir = node.parameters.update_iir
stored_update_fir = node.parameters.update_fir
stored_freq_to_flux_source = node.parameters.freq_to_flux_source
stored_debug_plots = node.parameters.debug_plots
stored_log_time_axis = node.parameters.log_time_axis


# %% {Create_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for coupler flux cryoscope measurement."""
    u = unit(coerce_to_integer=True)

    # --- Resolve qubit pairs and extract measured qubits --------------------
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Select which qubit in each pair acts as the Ramsey sensor
    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    # Set "qubits" for compatibility with cryoscope analysis functions
    node.namespace["qubits"] = measured_qubits

    n_avg = node.parameters.num_shots
    cryoscope_len = node.parameters.cryoscope_len

    # --- Amplitude resolution via freq_to_flux_source cascade ---
    detuning_hz = node.parameters.detuning_in_mhz * 1e6

    resolved = resolve_coupler_flux_amplitudes(
        qubit_pairs,
        measure_qubit=node.parameters.measure_qubit,
        detuning_hz=detuning_hz,
        freq_to_flux_source=node.parameters.freq_to_flux_source,
        fallback_amplitude_v=node.parameters.coupler_flux_amplitude_in_v,
        node=node,
        log_callable=node.log,
    )

    amplitudes = {}
    amplitude_resolution = {}
    for qp, amp, source in zip(qubit_pairs, resolved.amplitudes, resolved.sources):
        amplitudes[qp.coupler.name] = amp
        amplitude_resolution[qp.coupler.name] = {
            "amp_v": amp,
            "source": source,
            "detuning_mhz": node.parameters.detuning_in_mhz,
        }
        node.log(f"  {qp.coupler.name}: flux_amp={amp:.6f} V ({source})")

    # Record how each coupler amplitude was resolved for traceability.
    node.results["amplitude_resolution"] = amplitude_resolution

    cryoscope_time = np.arange(1, cryoscope_len + 1, 1)
    frames = np.linspace(0, 1, node.parameters.num_frames)

    baked_config = node.machine.generate_config()

    baked_signals = {
        qp.coupler.name: baked_coupler_waveform(baked_config, amplitudes[qp.coupler.name], qp.coupler, max_length=16)
        for qp in qubit_pairs
    }

    node.namespace["baked_config"] = baked_config

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "time": xr.DataArray(cryoscope_time, attrs={"long_name": "Cryoscope pulse duration", "units": "ns"}),
        "frame": xr.DataArray(frames, attrs={"long_name": "Frame rotation index"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubit_pairs)]
            state_st = [declare_stream() for _ in range(num_qubit_pairs)]
        t_left_ns = declare(int)
        t_cycles = declare(int)
        idx = declare(int)
        frame = declare(fixed)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(idx, 1, idx <= cryoscope_len, idx + 1):
                    with for_each_(frame, frames):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            protagonist.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        ################################################################################################
                        # The duration argument in the play command can only produce pulses with duration multiple of  #
                        # 4ns. To overcome this limitation we use the baking tool from the qualang-tools package to    #
                        # generate pulses with 1ns granularity. To avoid creating custom waveforms for each iteration  #
                        # we combine baked pulses with dynamically stretched (multiple of 4ns) pulses.                 #
                        ################################################################################################
                        with if_(idx <= 16):
                            with switch_(idx):
                                for j in range(1, 17):
                                    with case_(j):
                                        align()
                                        for ii, qp in multiplexed_qubit_pairs.items():
                                            protagonist = (
                                                qp.qubit_control
                                                if node.parameters.measure_qubit == "control"
                                                else qp.qubit_target
                                            )
                                            protagonist.xy.play("x90")
                                            qp.coupler.wait((protagonist.xy.operations["x90"].length + 16) // 4)
                                            baked_signals[qp.coupler.name][j - 1].run()
                                            protagonist.xy.wait((cryoscope_len + 16) >> 2)
                                            protagonist.xy.frame_rotation_2pi(frame)
                                            protagonist.xy.play("x90")
                        with else_():
                            assign(t_cycles, idx >> 2)
                            assign(t_left_ns, idx - (t_cycles << 2))
                            with switch_(t_left_ns):
                                with case_(0):
                                    align()
                                    for ii, qp in multiplexed_qubit_pairs.items():
                                        protagonist = (
                                            qp.qubit_control
                                            if node.parameters.measure_qubit == "control"
                                            else qp.qubit_target
                                        )
                                        protagonist.xy.play("x90")
                                        qp.coupler.wait((protagonist.xy.operations["x90"].length + 16) // 4)
                                        qp.coupler.play(
                                            "const",
                                            duration=t_cycles,
                                            amplitude_scale=amplitudes[qp.coupler.name]
                                            / qp.coupler.operations["const"].amplitude,
                                        )
                                        protagonist.xy.wait((cryoscope_len + 16) // 4)
                                        protagonist.xy.frame_rotation_2pi(frame)
                                        protagonist.xy.play("x90")
                                for j in range(1, 4):
                                    with case_(j):
                                        align()
                                        for ii, qp in multiplexed_qubit_pairs.items():
                                            protagonist = (
                                                qp.qubit_control
                                                if node.parameters.measure_qubit == "control"
                                                else qp.qubit_target
                                            )
                                            protagonist.xy.play("x90")
                                            qp.coupler.wait((protagonist.xy.operations["x90"].length + 16) // 4)
                                            qp.coupler.play(
                                                "const",
                                                duration=t_cycles,
                                                amplitude_scale=amplitudes[qp.coupler.name]
                                                / qp.coupler.operations["const"].amplitude,
                                            )
                                            baked_signals[qp.coupler.name][j - 1].run()
                                            protagonist.xy.wait((cryoscope_len + 16) // 4)
                                            protagonist.xy.frame_rotation_2pi(frame)
                                            protagonist.xy.play("x90")

                        align()
                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            if node.parameters.use_state_discrimination:
                                protagonist.readout_state(state[ii])
                                save(state[ii], state_st[ii])
                            else:
                                protagonist.resonator.measure("readout", qua_vars=(I[ii], Q[ii]))
                                save(I[ii], I_st[ii])
                                save(Q[ii], Q_st[ii])

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"Q{i + 1}")


# %% {Simulate_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.namespace["baked_config"]
    debug = False
    if debug:
        from pathlib import Path
        from qm import generate_qua_script
        file_name = Path(__file__).stem
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", 'w') as sourceFile:
            print(generate_qua_script(node.namespace["qua_program"], config), file=sourceFile)
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program, fetch the raw data and store it
    in an xarray dataset called "ds_raw".
    """
    qmm = node.machine.connect()
    config = node.namespace["baked_config"]
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

    # Rename "qubit_pair" -> "qubit" for cryoscope analysis compatibility.
    # Key the "qubit" dim by UNIQUE pair names (not measured-qubit names, which
    # repeat when several pairs share a measured target and would make .sel(qubit=name)
    # return a non-collapsed 3-D slice); keep the measured-qubit name as a side
    # coordinate for display. The shared coupler-branch analysis/plotting iterate by
    # this unique "qubit" coord.
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

    ds_raw = node.results.get("ds_raw")
    if ds_raw is not None and "qubit_pair" in ds_raw.dims:
        # Unique pair-name coordinate + measured-qubit name kept as a side coordinate
        # (see execute_qua_program for the rationale on duplicate measured targets).
        qubit_pair_names = [qp.name for qp in qubit_pairs]
        measured_qubit_names = [q.name for q in measured_qubits]
        ds_raw = ds_raw.rename({"qubit_pair": "qubit"})
        ds_raw = ds_raw.assign_coords(qubit=qubit_pair_names)
        ds_raw = ds_raw.assign_coords(measured_qubit_name=("qubit", measured_qubit_names))
    if ds_raw is not None:
        node.results["ds_raw"] = ds_raw

    node.parameters.n_exponentials = loaded_n_exponentials
    node.parameters.update_state_from_GUI = stored_gui_update_flag
    node.parameters.exponential_fit_time_fractions = loaded_fractions
    node.parameters.use_fir = stored_use_fir
    node.parameters.update_iir = stored_update_iir
    node.parameters.update_fir = stored_update_fir
    node.parameters.freq_to_flux_source = stored_freq_to_flux_source
    node.parameters.debug_plots = stored_debug_plots
    node.parameters.log_time_axis = stored_log_time_axis
    if node.parameters.update_state_from_GUI:
        node.machine = stored_machine
        node.parameters.update_state = True
        node.log("State update from GUI is enabled")


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data, store the fitted data in an xarray dataset "ds_fit" and
    the fitted results in the "fit_results" dictionary.
    """
    if "qubits" not in node.namespace:
        node.namespace["qubits"] = node.namespace["measured_qubits"]

    ds_proc = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_proc"] = ds_proc
    ds_fit, fit_results = fit_raw_data(ds_proc, node)
    node.results["ds_fit"] = ds_fit

    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)

    # fit_results is keyed by the dataset "qubit" coordinate, i.e. unique pair names
    # (the coupler branch of fit_raw_data), so map outcomes straight from it.
    qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
    node.outcomes = {
        qubit_pair_name: (
            "successful" if node.results["fit_results"].get(qubit_pair_name, {}).get("success", False) else "failed"
        )
        for qubit_pair_name in qubit_pair_names
    }

    # --- FIR analysis (optional) ---
    if node.parameters.use_fir:
        fir_results = fit_fir_data(node.results["ds_fit"], node)
        node.namespace["fir_results"] = fir_results
        node.results["fir_results"] = {
            qn: {k: v for k, v in res.items() if not str(k).startswith("fig")}
            for qn, res in fir_results.items()
        }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot flux response and IIR fit (plus debug/FIR figures when enabled)."""
    if "ds_fit" not in node.results:
        return
    qubit_pairs = node.namespace.get("qubit_pairs", get_qubit_pairs(node))
    ds_fit = node.results["ds_fit"]
    fit_results = node.results["fit_results"]
    fir_results = node.namespace.get("fir_results")
    debug_plots = node.parameters.debug_plots

    figures = plot_raw_data_with_fit(
        ds_fit,
        qubit_pairs,
        fit_results,
        debug=debug_plots,
        fir_results=fir_results,
        log_scale=node.parameters.log_time_axis,
    )

    if debug_plots:
        figures.update(plot_raw_data(node.results["ds_raw"], qubit_pairs))
        if fir_results:
            figures.update(plot_fir_figures(ds_fit, qubit_pairs, fir_results, debug=True))

    node.results["figures"] = figures
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Push fitted IIR and/or FIR filters into state when enabled."""
    if not node.parameters.update_state:
        return

    skip_pairs = {qp.name for qp in node.namespace["qubit_pairs"] if node.outcomes.get(qp.name) == "failed"}

    with node.record_state_updates():
        update_coupler_filters(
            node.namespace["qubit_pairs"],
            node.results["fit_results"],
            skip_pairs=skip_pairs,
            update_iir=node.parameters.update_iir,
            update_fir=node.parameters.update_fir,
            fir_results=node.namespace.get("fir_results"),
            log_callable=node.log,
        )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results and state updates."""
    node.save()


# %%

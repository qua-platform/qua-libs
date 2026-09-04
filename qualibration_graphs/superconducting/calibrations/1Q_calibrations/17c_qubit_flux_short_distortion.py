"""Cryoscope calibration for flux line step response — short-time IIR/FIR correction."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.common_utils.flux_distortions import update_filters
from calibration_utils.qubit_flux_short_distortion import (
    Parameters,
    baked_waveform,
    fit_fir_data,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
    resolve_flux_amplitudes,
)
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam


# %% {Node_parameters}
description = """
CRYOSCOPE (17c — short-time flux distortion, optional FIR)

Ramsey-style cryoscope sweeping flux pulse duration at a fixed amplitude,
with frame rotation for phase reconstruction.

Workflow:
For each qubit, resolve a Z-pulse amplitude that places the qubit at
`detuning_in_mhz` below idle via `freq_to_flux_source`, bake 1 ns flux
segments, sweep pulse duration and frame, and reconstruct phase → frequency →
flux step response. Fit a sum of exponentials (IIR); optionally run fixed-length
FIR feedforward analysis (``use_fir``) with a compact summary plot.

Prerequisites
- A valid rotation angle and threshold if using state discrimination
- Calibrated XYZ delay (16a)
- Calibrated x90 pulse
- Each qubit parked at its flux sweetspot. The Z pulse amplitude is derived as a
  magnitude and `f(Φ)` is assumed symmetric about idle.
- A frequency→voltage relation for each qubit, used both to pick the Z amplitude
  and to invert the measurement. `freq_to_flux_source="auto"` (default) takes the
  first available of:
    1. Ramsey vs flux (09a), run ID from `extras['ramsey_vs_flux_calibration_load_id']`
    2. Qubit spectroscopy vs flux (03b), run ID from `extras['qubit_spectroscopy_vs_flux_load_id']`
    3. `freq_vs_flux_01_quad_term` in the state
  Run 09a / 03b with `save_load_id=True` so their run IDs land in the state; no
  run ID is ever typed into this node. Set `freq_to_flux_source` to `"ramsey"`,
  `"spectroscopy"` or `"quad_term"` to force one source.

Outputs and state updates
- Results: processed dataset, fit results, and figures under `node.results`.
- Set `update_state=True` to write filters; use `update_iir` and/or `update_fir`
  to choose which filters are committed.
- Re-load a prior run with `load_data_id`, tune fit settings in the GUI, then set
  `update_state_from_GUI=True` to commit without re-acquiring data.
REMINDER: digital filters add a global delay — recalibrate IQ blobs
(rotation_angle & ge_threshold) and (16a) XYZ_delay.
"""

node = QualibrationNode[Parameters, Quam](
    name="17c_qubit_flux_short_distortion",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    pass


# Instantiate the QUAM class from the state file
stored_machine = Quam.load()

loaded_n_exponentials = node.parameters.n_exponentials
stored_gui_update_flag = node.parameters.update_state_from_GUI
stored_update_iir = node.parameters.update_iir
stored_update_fir = node.parameters.update_fir
stored_freq_to_flux_source = node.parameters.freq_to_flux_source


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    cryoscope_len = node.parameters.cryoscope_len

    # --- Per-qubit flux_amp derivation via the selected freq→flux relation ---
    resolved = resolve_flux_amplitudes(
        qubits,
        detuning_hz=node.parameters.detuning_in_mhz * 1e6,
        freq_to_flux_source=node.parameters.freq_to_flux_source,
        log_callable=node.log,
    )
    amplitudes = resolved.amplitudes
    for i, q in enumerate(qubits):
        node.log(f"  {q.name}: flux_amp={amplitudes[i]:.6f} V ({resolved.sources[i]})")

    cryoscope_time = np.arange(1, cryoscope_len + 1, 1)
    frames = np.linspace(0, 1, node.parameters.num_frames)

    baked_config = node.machine.generate_config()
    baked_signals = {
        q.name: baked_waveform(baked_config, amplitudes[i], q, max_length=16) for i, q in enumerate(qubits)
    }

    node.namespace["baked_config"] = baked_config
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "time": xr.DataArray(cryoscope_time, attrs={"long_name": "Cryoscope pulse duration", "units": "ns"}),
        "frame": xr.DataArray(frames, attrs={"long_name": "Frame rotation index"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        t_left_ns = declare(int)
        t_cycles = declare(int)
        idx = declare(int)
        frame = declare(fixed)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(idx, 1, idx <= cryoscope_len, idx + 1):
                    with for_each_(frame, frames):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
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
                                        for i, qubit in multiplexed_qubits.items():
                                            qubit.xy.play("x90")
                                            qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                            baked_signals[qubit.name][j - 1].run()
                                            qubit.xy.wait((cryoscope_len + 16) >> 2)
                                            qubit.xy.frame_rotation_2pi(frame)
                                            qubit.xy.play("x90")
                        with else_():
                            assign(t_cycles, idx >> 2)
                            assign(t_left_ns, idx - (t_cycles << 2))
                            with switch_(t_left_ns):
                                with case_(0):
                                    align()
                                    for i, qubit in multiplexed_qubits.items():
                                        qubit.xy.play("x90")
                                        qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                        qubit.z.play(
                                            "const",
                                            duration=t_cycles,
                                            amplitude_scale=amplitudes[i] / qubit.z.operations["const"].amplitude,
                                        )
                                        qubit.xy.wait((cryoscope_len + 16) // 4)
                                        qubit.xy.frame_rotation_2pi(frame)
                                        qubit.xy.play("x90")
                                for j in range(1, 4):
                                    with case_(j):
                                        align()
                                        for i, qubit in multiplexed_qubits.items():
                                            qubit.xy.play("x90")
                                            qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                            qubit.z.play(
                                                "const",
                                                duration=t_cycles,
                                                amplitude_scale=amplitudes[i] / qubit.z.operations["const"].amplitude,
                                            )
                                            baked_signals[qubit.name][j - 1].run()
                                            qubit.xy.wait((cryoscope_len + 16) // 4)
                                            qubit.xy.frame_rotation_2pi(frame)
                                            qubit.xy.play("x90")

                        align()
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"Q{i + 1}")


# %% {Simulate}
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
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", "w") as sourceFile:
            print(generate_qua_script(node.namespace["qua_program"], config), file=sourceFile)
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
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
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)
    node.parameters.n_exponentials = loaded_n_exponentials
    node.parameters.update_state_from_GUI = stored_gui_update_flag
    node.parameters.update_iir = stored_update_iir
    node.parameters.update_fir = stored_update_fir
    node.parameters.freq_to_flux_source = stored_freq_to_flux_source
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
    ds_proc = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_proc, node)

    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }

    # --- FIR analysis (optional) ---
    if node.parameters.use_fir:
        fir_results = fit_fir_data(node.results["ds_fit"], node)
        node.namespace["fir_results"] = fir_results
        node.results["fir_results"] = {
            qn: {k: v for k, v in res.items() if not str(k).startswith("fig")} for qn, res in fir_results.items()
        }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot cryoscope freq, flux response, and IIR fit (plus debug/FIR when enabled)."""
    if "ds_fit" not in node.results:
        return
    qubits = node.namespace.get("qubits", get_qubits(node))
    node.results["figures"] = plot_raw_data_with_fit(
        node.results["ds_fit"],
        qubits,
        node.results["fit_results"],
        debug=node.parameters.debug_plots,
        fir_results=node.namespace.get("fir_results"),
    )
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Push fitted IIR and/or FIR filters into state when enabled."""
    if not node.parameters.update_state:
        return

    skip_qubits = {q.name for q in node.namespace["qubits"] if node.outcomes.get(q.name) == "failed"}

    with node.record_state_updates():
        update_filters(
            node.namespace["qubits"],
            node.machine,
            node.results["fit_results"],
            update_iir=node.parameters.update_iir,
            update_fir=node.parameters.update_fir,
            fir_results=node.namespace.get("fir_results"),
            skip_qubits=skip_qubits,
            log_callable=node.log,
        )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results and state updates."""
    node.save()

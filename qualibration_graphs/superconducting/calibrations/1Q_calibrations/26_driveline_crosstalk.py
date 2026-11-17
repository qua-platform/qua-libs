# %% {Imports}
from dataclasses import asdict

import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from calibration_utils.driveline_crosstalk import (
    MODES,
    Parameters,
    amplitude_factors,
    get_pair_context,
    fit_amplitude,
    select_phase,
    coarse_phase_sweep,
    fine_phase_sweep,
    fit_raw_data,
    log_fitted_results,
    plot_amplitude_data_with_fit,
    plot_phase_data,
    plot_check_data,
    plot_crosstalk_matrix,
)

# %% {Node initialisation}
description = """
        DRIVE-LINE CROSSTALK
Measures and compensates the microwave crosstalk between the XY drive lines of the selected qubits. For every directed
pair (drive -> target), the drive line plays a long square pulse at the target frequency and the target state is
measured. The calibration has three stages, each saved as its own dataset:
    1. Amplitude (ds_raw_amplitude): power Rabi of every drive line on every target. The amplitude ratio
       r = a_pi(target -> target) / a_pi(drive -> target) is the compensation magnitude.
    2. Phase (ds_raw_phase): the target line simultaneously plays r times the drive-line pulse with a swept phase,
       first over a full period (coarse) then around the coarse minimum (fine). The compensation phase is the minimum
       of the target excitation.
    3. Check (ds_raw_check): independent no-drive, uncompensated and compensated power Rabi with the selected r and
       phase. A pair passes if the residual excitation is small in absolute value and relative to the uncompensated one.
Pairs whose drive line cannot reach the target frequency within max_drive_if_in_mhz, or that share an output port,
are skipped. The LO frequencies are not modified.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c) and the IQ blobs (node 07).
    - Having calibrated the qubit frequency and the x180 pulse parameters (nodes 03a and 04b).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - For every pair that passed the check, on the drive qubit:
      qubit.extras["drive_crosstalk"][target] = {"amplitude_ratio": r, "phase": phase [rad]}
      i.e. when the drive qubit plays a pulse of amplitude A, the target line must play r * A at the drive frequency,
      with the given phase, to cancel the crosstalk.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="26_driveline_crosstalk",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    # Pairs whose drive line cannot reach the target frequency within max_drive_if_in_mhz are skipped. To measure
    # them, put the drive lines on a common LO for this run only: the change stays in memory and the original LOs are
    # restored in save_results, so it is never written to the state.
    # common_lo_frequency = 5.375e9
    # node.namespace["original_lo"] = {}
    # for name in node.parameters.qubits:
    #     port = node.machine.qubits[name].xy.opx_output
    #     node.namespace["original_lo"][name] = port.upconverter_frequency
    #     port.upconverter_frequency = common_lo_frequency
    #     node.log(f"{name}: drive LO {node.namespace['original_lo'][name] / 1e9:.4f} -> "
    #              f"{common_lo_frequency / 1e9:.4f} GHz for this run")
    pass


# %% {Create_QUA_program}
def play_pair(node, target, drive, drive_scale, compensation_scale, phase, state):
    """Reset, play the drive line at the target frequency (with an optional compensation on the target line) and
    measure the target state."""
    p = node.parameters
    # Reset the qubits to the ground state
    target.reset(p.reset_type, p.simulate, log_callable=node.log)
    if drive is not target:
        drive.reset(p.reset_type, p.simulate, log_callable=node.log)
    align()
    # The drive line plays at the target frequency
    if drive is not target:
        drive.xy.update_frequency(int(round(target.xy.RF_frequency - drive.xy.LO_frequency)))
    align()
    # Phase reference shared by the drive and the compensation lines
    reset_global_phase()
    reset_frame(drive.xy.name)
    reset_if_phase(drive.xy.name)
    if drive is not target:
        reset_frame(target.xy.name)
        reset_if_phase(target.xy.name)
        target.xy.frame_rotation_2pi(phase)
    align()
    # The qubit manipulation sequence
    drive.xy.play(p.operation, amplitude_scale=drive_scale, duration=p.pulse_length_ns // 4)
    if drive is not target:
        target.xy.play(p.operation, amplitude_scale=compensation_scale, duration=p.pulse_length_ns // 4)
    align()
    # Restore the drive frequency and the target frame
    if drive is not target:
        drive.xy.update_frequency(int(round(drive.xy.RF_frequency - drive.xy.LO_frequency)))
    reset_frame(target.xy.name)
    # Measure the state of the target
    target.readout_state(state)
    align()


def compensation_scales(node, amplitude_fit: xr.Dataset) -> xr.DataArray:
    """Amplitude scale of the compensation pulse relative to the normalized drive amplitude; NaN for pairs that can't be
    compensated within the output range."""
    context = node.namespace["context"]
    # Compensation amplitude, in output units, at the end of the normalized sweep
    compensation_amplitude = amplitude_fit.amplitude_ratio * context.pair_amp_scale * context.drive_base_amplitude
    scale = compensation_amplitude / context.target_base_amplitude
    in_range = (scale < 2) & (compensation_amplitude < 1)
    off_diagonal = context.qubit != context.drive_qubit
    return scale.where(context.pair_valid & off_diagonal & in_range & (amplitude_fit.amplitude_ratio > 0))


def get_amplitude_program(node, targets: list):
    """QUA program of the amplitude stage: power Rabi of every drive line on each target. Invalid pairs save -1."""
    qubits, context = node.namespace["qubits"], node.namespace["context"]
    amps = amplitude_factors(node.parameters)
    with program() as qua_program:
        n = declare(int)
        n_st = declare_stream()
        a = declare(fixed)
        state = [declare(int) for _ in range(len(targets))]
        state_st = [declare_stream() for _ in range(len(targets))]

        # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
        for qubit in qubits:
            node.machine.initialize_qpu(target=qubit)
        align()

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)
            for j, target in enumerate(targets):
                for drive in qubits:
                    selection = dict(qubit=target.name, drive_qubit=drive.name)
                    scale = float(context.pair_amp_scale.sel(selection))
                    with for_(*from_array(a, amps)):
                        if bool(context.pair_valid.sel(selection)):
                            play_pair(node, target, drive, a * scale, 0, 0, state[j])
                            save(state[j], state_st[j])
                        else:
                            save(-1, state_st[j])

        with stream_processing():
            n_st.save("n")
            for j in range(len(targets)):
                state_st[j].buffer(len(amps)).buffer(len(qubits)).average().save(f"state{j + 1}")
    return qua_program


def get_phase_program(node, target, drive, compensation_scale: float, phase_turns: np.ndarray):
    """QUA program of the compensation-phase sweep of a single pair."""
    amps = amplitude_factors(node.parameters)
    s = float(node.namespace["context"].pair_amp_scale.sel(qubit=target.name, drive_qubit=drive.name))
    with program() as qua_program:
        n = declare(int)
        n_st = declare_stream()
        a = declare(fixed)
        phase = declare(fixed)
        state = declare(int)
        state_st = declare_stream()
        for qubit in (target, drive):
            node.machine.initialize_qpu(target=qubit)
        align()
        with for_(n, 0, n < node.parameters.num_shots_phase, n + 1):
            save(n, n_st)
            with for_(*from_array(a, amps)):
                with for_(*from_array(phase, phase_turns)):
                    play_pair(node, target, drive, a * s, a * compensation_scale, phase, state)
                    save(state, state_st)
        with stream_processing():
            n_st.save("n")
            state_st.buffer(len(phase_turns)).buffer(len(amps)).buffer(1).average().save("state1")
    return qua_program


def get_check_program(node, target, drive, compensation_scale: float, phase_turns: float):
    """QUA program of the no-drive / uncompensated / compensated check of a single pair."""
    amps = amplitude_factors(node.parameters)
    s = float(node.namespace["context"].pair_amp_scale.sel(qubit=target.name, drive_qubit=drive.name))
    with program() as qua_program:
        n = declare(int)
        n_st = declare_stream()
        a = declare(fixed)
        state = declare(int)
        state_st = declare_stream()
        for qubit in (target, drive):
            node.machine.initialize_qpu(target=qubit)
        align()
        with for_(n, 0, n < node.parameters.num_shots_check, n + 1):
            save(n, n_st)
            for mode in MODES:
                with for_(*from_array(a, amps)):
                    drive_scale = 0 if mode == "no_drive" else a * s
                    play_pair(
                        node,
                        target,
                        drive,
                        drive_scale,
                        a * compensation_scale if mode == "compensated" else 0,
                        phase_turns,
                        state,
                    )
                    save(state, state_st)
        with stream_processing():
            n_st.save("n")
            state_st.buffer(len(amps)).buffer(len(MODES)).buffer(1).average().save("state1")
    return qua_program


@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program of the amplitude stage from the node parameters."""
    # Get the active qubits from the node
    node.namespace["qubits"] = qubits = get_qubits(node)
    if len(qubits) < 2:
        raise ValueError("Drive-line crosstalk requires at least two qubits.")
    # Per-pair amplitudes and the pairs that can be measured with the current LO frequencies
    node.namespace["context"], skipped = get_pair_context(qubits, node.parameters)
    for pair, reason in skipped.items():
        node.log(f"Skipping {pair}: {reason}")
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "drive_qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(
            amplitude_factors(node.parameters), attrs={"long_name": "normalized drive amplitude"}
        ),
    }
    # The QUA program of all targets, used for the simulation. On hardware, the stages run as short jobs (one per
    # target or per pair) to stay within the result-stream timeout of cloud-hosted QOPs.
    node.namespace["qua_program"] = get_amplitude_program(node, list(qubits))


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program of the amplitude stage"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
def execute(node, qua_program, sweep_axes: dict, num_shots: int) -> xr.Dataset:
    """Connect to the QOP, execute a QUA program and fetch the raw data in an xarray dataset."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(qua_program)
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, sweep_axes)
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Unmeasured pairs were saved as -1
    dataset["state"] = dataset.state.where(dataset.state >= 0)
    return dataset


def combine(node, datasets: list, sweep_axes: dict) -> xr.Dataset:
    """Combine the datasets of the short jobs into one dataset over all pairs; pairs that were not measured are NaN.
    Without any job, the dataset is filled with NaN so that the analysis and the plots stay uniform."""
    names = node.namespace["qubits"].get_names()
    if datasets:
        ds = xr.merge(datasets, compat="no_conflicts", join="outer").reindex(qubit=names, drive_qubit=names)
    else:
        coords = {"qubit": names, "drive_qubit": names, **{k: v.values for k, v in sweep_axes.items()}}
        state = np.full([len(v) for v in coords.values()], np.nan)
        ds = xr.Dataset({"state": (tuple(coords), state)}, coords=coords)
    return xr.merge([ds, node.namespace["context"]])


def pair_axes(node, target, drive) -> dict:
    """Sweep axes of a single-pair job."""
    return {
        "qubit": xr.DataArray([target.name]),
        "drive_qubit": xr.DataArray([drive.name]),
        "amp_prefactor": node.namespace["sweep_axes"]["amp_prefactor"],
    }


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the amplitude stage, one job per target, and register its raw dataset."""
    datasets = []
    for target in node.namespace["qubits"]:
        node.log(f"Amplitude stage: target {target.name}")
        sweep_axes = {**node.namespace["sweep_axes"], "qubit": xr.DataArray([target.name])}
        datasets.append(execute(node, get_amplitude_program(node, [target]), sweep_axes, node.parameters.num_shots))
    node.results["ds_raw_amplitude"] = combine(node, datasets, {})


# %% {Execute_phase}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_phase_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the coarse and fine compensation-phase sweeps, one job per pair, using the amplitude ratio of the
    amplitude stage. Pairs without an amplitude ratio are not measured."""
    qubits = node.namespace["qubits"]
    amplitude_fit = fit_amplitude(node.results["ds_raw_amplitude"], node.parameters.min_fit_r2)
    node.namespace["compensation_scales"] = scales = compensation_scales(node, amplitude_fit)
    pairs = [(t, d) for t in qubits for d in qubits if np.isfinite(float(scales.sel(qubit=t.name, drive_qubit=d.name)))]
    stages = []
    phase_turns = coarse_phase_sweep(qubits.get_names(), node.parameters.num_phases)
    for phase_sweep in ["coarse", "fine"]:
        if phase_sweep == "fine":
            center = select_phase(stages[0], amplitude_fit, coarse=True)
            phase_turns = fine_phase_sweep(center, node.parameters.num_phases)
        datasets = []
        for target, drive in pairs:
            node.log(f"Phase stage ({phase_sweep}): {drive.name} -> {target.name}")
            k = float(scales.sel(qubit=target.name, drive_qubit=drive.name))
            turns = phase_turns.sel(qubit=target.name, drive_qubit=drive.name).values
            sweep_axes = {**pair_axes(node, target, drive), "phase_index": xr.DataArray(np.arange(len(turns)))}
            qua_program = get_phase_program(node, target, drive, k, turns)
            datasets.append(execute(node, qua_program, sweep_axes, node.parameters.num_shots_phase))
        empty_axes = {
            "amp_prefactor": node.namespace["sweep_axes"]["amp_prefactor"],
            "phase_index": xr.DataArray(np.arange(node.parameters.num_phases)),
        }
        ds = combine(node, datasets, empty_axes)
        ds["phase_turns"] = phase_turns
        stages.append(ds)
    ds_phase = xr.concat(
        [ds[["state", "phase_turns"]] for ds in stages], dim=xr.DataArray(["coarse", "fine"], dims="phase_sweep")
    )
    ds_phase = xr.merge([ds_phase, node.namespace["context"]])
    ds_phase["amplitude_ratio_used"] = amplitude_fit.amplitude_ratio.where(scales.notnull())
    node.results["ds_raw_phase"] = ds_phase


# %% {Execute_check}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_check_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the independent compensation check, one job per pair, with the selected amplitude ratio and phase."""
    qubits = node.namespace["qubits"]
    ds_phase = node.results["ds_raw_phase"]
    scales = node.namespace["compensation_scales"]
    amplitude_fit = fit_amplitude(node.results["ds_raw_amplitude"], node.parameters.min_fit_r2)
    phase_turns = select_phase(ds_phase.sel(phase_sweep="fine"), amplitude_fit)
    datasets = []
    for target in qubits:
        for drive in qubits:
            k = float(scales.sel(qubit=target.name, drive_qubit=drive.name))
            theta = float(phase_turns.sel(qubit=target.name, drive_qubit=drive.name))
            if not (np.isfinite(k) and np.isfinite(theta)):
                continue
            node.log(f"Check stage: {drive.name} -> {target.name}")
            axes = pair_axes(node, target, drive)
            sweep_axes = {
                "qubit": axes["qubit"],
                "drive_qubit": axes["drive_qubit"],
                "mode": xr.DataArray(MODES),
                "amp_prefactor": axes["amp_prefactor"],
            }
            qua_program = get_check_program(node, target, drive, k, theta)
            datasets.append(execute(node, qua_program, sweep_axes, node.parameters.num_shots_check))
    empty_axes = {"mode": xr.DataArray(MODES), "amp_prefactor": node.namespace["sweep_axes"]["amp_prefactor"]}
    ds_check = combine(node, datasets, empty_axes)
    # Store the values that were actually played, so that loading the data reproduces the verdict
    ds_check["amplitude_ratio_used"] = ds_phase.amplitude_ratio_used.where(phase_turns.notnull())
    ds_check["compensation_phase_used"] = (2 * np.pi * phase_turns).where(ds_check.amplitude_ratio_used.notnull())
    ds_check.attrs["num_shots"] = node.parameters.num_shots_check
    node.results["ds_raw_check"] = ds_check


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the three raw datasets and store the fitted data in ds_fit and the fitted results in fit_results."""
    node.results["ds_fit"], fit_results = fit_raw_data(
        node.results["ds_raw_amplitude"],
        node.results["ds_raw_phase"],
        node.results["ds_raw_check"],
        node,
    )
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    # A drive qubit is successful if it has measured pairs and all of them pass the compensation check
    node.outcomes = {}
    for qubit in node.namespace["qubits"]:
        verdicts = [
            fit["success"]
            for fit in node.results["fit_results"].values()
            if fit["drive_qubit"] == qubit.name and np.isfinite(fit["amplitude_ratio"])
        ]
        node.outcomes[qubit.name] = "successful" if verdicts and all(verdicts) else "failed"


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data of the three stages in figures whose shape is given by qubit.grid_location."""
    qubits = node.namespace["qubits"]
    ds_fit = node.results["ds_fit"]
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": plot_amplitude_data_with_fit(node.results["ds_raw_amplitude"], qubits, ds_fit),
        "phase": plot_phase_data(node.results["ds_raw_phase"], qubits, ds_fit),
        "check": plot_check_data(node.results["ds_raw_check"], qubits, ds_fit),
        "crosstalk_matrix": plot_crosstalk_matrix(ds_fit),
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write the compensation of every pair that passed the check on its drive qubit."""
    # TODO: move to a dedicated drive-crosstalk field once it is available in quam-builder.
    with node.record_state_updates():
        for fit in node.results["fit_results"].values():
            if not fit["success"]:
                continue
            drive = node.machine.qubits[fit["drive_qubit"]]
            if "drive_crosstalk" not in drive.extras:
                drive.extras["drive_crosstalk"] = {}
            drive.extras["drive_crosstalk"][fit["target_qubit"]] = {
                "amplitude_ratio": fit["amplitude_ratio"],
                "phase": fit["phase"],
            }


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    # Restore the drive LOs changed in custom_param, if any, before the machine state is saved
    for name, lo in node.namespace.get("original_lo", {}).items():
        node.machine.qubits[name].xy.opx_output.upconverter_frequency = lo
    node.save()

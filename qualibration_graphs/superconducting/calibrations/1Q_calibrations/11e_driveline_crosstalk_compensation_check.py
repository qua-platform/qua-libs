# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import (
    align,
    declare,
    declare_stream,
    fixed,
    for_,
    program,
    reset_frame,
    reset_global_phase,
    reset_if_phase,
    save,
    stream_processing,
)
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot

from calibration_utils.driveline_crosstalk import CheckParameters as Parameters
from calibration_utils.driveline_crosstalk import (
    amplitude_factors,
    check_compensation,
    log_validation_results,
    plot_check,
    plot_matrix,
    plot_validation_table,
    prepare_machine,
    read_matrix,
    save_matrix,
)
from quam_config import Quam

# %% {Description}
description = """DRIVE-LINE CROSSTALK COMPENSATION CHECK
Read amplitude_matrix.h5 and phase_matrix.h5. Independently measure no_drive,
uncompensated, and compensated power Rabi for all directed pairs. Return the
validation table and additive coefficient K = r*exp(i*theta). Failed candidates
remain labelled failed; a phase minimum alone is not proof of cancellation.
The active machine state is not updated. Full datasets and figures use normal
QUAlibrate storage; the matrix H5 is copied to QEC/crosstalktest.
"""

node = QualibrationNode[Parameters, Quam](
    name="11e_driveline_crosstalk_compensation_check",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# %% {Prepare_machine}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def prepare(node: QualibrationNode[Parameters, Quam]):
    prepare_machine(node)


# %% {Create_QUA_program}
def _build_program(node, magnitude=None, phases=None, target_name=None, drive_name=None):
    """Build a check QUA program for simulation or a short directed-pair job."""
    p, qubits = (node.parameters, node.namespace["qubits"])
    amps = amplitude_factors(p)
    targets = [q for q in qubits if target_name is None or q.name == target_name]
    drives = [q for q in qubits if drive_name is None or q.name == drive_name]
    axes = {"qubit": xr.DataArray([q.name for q in targets]), "drive_qubit": xr.DataArray([q.name for q in drives])}
    axes.update(mode=xr.DataArray(["no_drive", "uncompensated", "compensated"]), amp_prefactor=xr.DataArray(amps))
    for target in targets:
        for drive in drives:
            if drive is target:
                continue
            r = float(magnitude.amplitude_ratio.sel(qubit=target.name, drive_qubit=drive.name))
            source_base = float(drive.xy.operations[p.operation].amplitude)
            target_base = float(target.xy.operations[p.operation].amplitude)
            if not np.isfinite(r) or r <= 0:
                raise ValueError(f"{drive.name}->{target.name}: no valid amplitude fit")
            if p.max_amp_factor * r * source_base / target_base >= 2 or p.max_amp_factor * r * source_base >= 1:
                raise ValueError(f"{drive.name}->{target.name}: compensation exceeds output range")
            if drive.xy.opx_output == target.xy.opx_output:
                raise ValueError("Compensation requires distinct physical output ports")
    for drive in drives:
        if p.max_amp_factor * drive.xy.operations[p.operation].amplitude >= 1:
            raise ValueError(f"{drive.name}: source amplitude exceeds output range")
    with program() as qua_program:
        n, a, phase = (declare(int), declare(fixed), declare(fixed))
        n_st = declare_stream()
        states = [declare(int) for _ in targets]
        streams = [declare_stream() for _ in targets]
        for j, target in enumerate(targets):
            node.machine.initialize_qpu(target=target)
            align()
            with for_(n, 0, n < p.num_shots, n + 1):
                save(n, n_st)
                for drive in drives:
                    diagonal = drive is target
                    modes = range(3)
                    for mode in modes:
                        with for_(*from_array(a, amps)):
                            pair_phases = None
                            if diagonal:
                                save(-1, streams[j])
                            else:
                                theta = (
                                    float(phases.sel(qubit=target.name, drive_qubit=drive.name)) if not diagonal else 0
                                )
                                source_scale = 1
                                _measure_pair(
                                    target, drive, p, a * source_scale, magnitude, theta, mode, states[j], streams[j]
                                )
        with stream_processing():
            n_st.save("n")
            for j, stream in enumerate(streams):
                result = stream.buffer(len(amps))
                result = result.buffer(3)
                result.buffer(len(drives)).average().save(f"state{j + 1}")
    return (qua_program, axes)


def _measure_pair(target, drive, p, a, magnitude, phase, mode, state, stream):
    target.reset(p.reset_type, p.simulate)
    if drive is not target:
        drive.reset(p.reset_type, p.simulate)
    align()
    drive.xy.update_frequency(int(round(target.xy.RF_frequency - drive.xy.LO_frequency)))
    target.xy.update_frequency(int(round(target.xy.RF_frequency - target.xy.LO_frequency)))
    align()
    reset_global_phase()
    reset_frame(drive.xy.name)
    reset_if_phase(drive.xy.name)
    if drive is not target:
        reset_frame(target.xy.name)
        reset_if_phase(target.xy.name)
        target.xy.frame_rotation_2pi(phase)
    align()
    drive.xy.play(p.operation, amplitude_scale=a if mode != 0 else 0, duration=p.pulse_length_ns // 4)
    if drive is not target:
        scale = 0
        if mode == 2:
            r = float(magnitude.amplitude_ratio.sel(qubit=target.name, drive_qubit=drive.name))
            scale = a * (
                r
                * float(drive.xy.operations[p.operation].amplitude)
                / float(target.xy.operations[p.operation].amplitude)
            )
        target.xy.play(p.operation, amplitude_scale=scale, duration=p.pulse_length_ns // 4)
    align()
    drive.xy.update_frequency(int(round(drive.xy.RF_frequency - drive.xy.LO_frequency)))
    reset_frame(target.xy.name)
    target.readout_state(state)
    save(state, stream)
    align()


@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    node.namespace["amplitude_input"] = magnitude = read_matrix(node, "amplitude")
    node.namespace["phase_input"] = phase = read_matrix(node, "phase")
    if phase.attrs.get("amplitude_matrix_sha256") != node.namespace["amplitude_matrix_sha256"]:
        raise ValueError("Phase matrix belongs to a different amplitude run; rerun 11d")
    if not np.allclose(magnitude.amplitude_ratio, phase.amplitude_ratio, rtol=1e-12, atol=0, equal_nan=True):
        raise ValueError("Amplitude matrix changed after phase calibration; rerun 11d")
    off_diagonal = phase.compensation_phase_rad.where(phase.qubit != phase.drive_qubit)
    if int(off_diagonal.count()) != len(phase.qubit) * (len(phase.qubit) - 1):
        raise ValueError("Missing compensation phases; rerun 11d")
    node.namespace["qua_program"], node.namespace["sweep_axes"] = _build_program(
        node, magnitude, phase.compensation_phase_rad.fillna(0) / (2 * np.pi)
    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    samples, figure, report = simulate_and_plot(
        node.machine.connect(), node.machine.generate_config(), node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {"samples": samples, "figure": figure, "wf_report": report}


# %% {Execute}
def _execute_program(node, qua_program, axes):
    dataset = None
    with qm_session(node.machine.connect(), node.machine.generate_config(), timeout=node.parameters.timeout) as qm:
        job = qm.execute(qua_program, options={"timeout": node.parameters.timeout})
        fetcher = XarrayDataFetcher(job, axes)
        for dataset in fetcher:
            progress_counter(fetcher.get("n", 0), node.parameters.num_shots, start_time=fetcher.t_start)
        node.log(job.execution_report())
    if dataset is None or "state" not in dataset:
        raise RuntimeError("No state streams returned")
    context = node.namespace["context"].sel(qubit=axes["qubit"].values, drive_qubit=axes["drive_qubit"].values)
    ds = xr.merge([dataset, context])
    ds.attrs.update(node.namespace["context"].attrs, num_shots=node.parameters.num_shots)
    return ds


def _execute_sweep(node, stage, magnitude, phases):
    """Short cloud jobs, one combined dataset and one normal saved node."""
    datasets = []
    for target in node.namespace["qubits"]:
        for drive in node.namespace["qubits"]:
            if target is drive:
                continue
            node.log(f"{stage}: {drive.name} -> {target.name}")
            qua_program, axes = _build_program(node, magnitude, phases, target.name, drive.name)
            datasets.append(_execute_program(node, qua_program, axes)[["state"]])
    combined = xr.combine_by_coords(datasets, combine_attrs="override")
    return xr.merge([combined, node.namespace["context"]])


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = _execute_sweep(
        node,
        "check",
        node.namespace["amplitude_input"],
        node.namespace["phase_input"].compensation_phase_rad.fillna(0) / (2 * np.pi),
    )
    node.results["ds_raw"]["amplitude_ratio"] = node.namespace["amplitude_input"].amplitude_ratio
    node.results["ds_raw"]["compensation_phase_rad"] = node.namespace["phase_input"].compensation_phase_rad
    node.results["ds_raw"].attrs.update(
        amplitude_matrix_sha256=node.namespace["amplitude_matrix_sha256"],
        phase_matrix_sha256=node.namespace["phase_matrix_sha256"],
    )
    node.results["ds_raw"]["state"] = node.results["ds_raw"].state.where(
        node.results["ds_raw"].qubit != node.results["ds_raw"].drive_qubit
    )


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_id = node.parameters.load_data_id
    node.load_from_id(load_id)
    node.parameters.load_data_id = load_id


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_fit"], node.results["fit_results"] = check_compensation(
        node.results["ds_raw"],
        node.results["ds_raw"][["amplitude_ratio", "target_rf_hz", "drive_base_amplitude", "drive_mV_per_prefactor"]],
        node.results["ds_raw"],
        node.parameters,
    )
    log_validation_results(node.results["fit_results"], node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    node.results["figures"] = {
        "compensation_check": plot_check(node.results["ds_raw"], node.results["ds_fit"]),
        "compensation_matrix": plot_matrix(node.results["ds_fit"], "compensation"),
        "validation_table": plot_validation_table(node.results["fit_results"]),
    }
    plt.show()


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    save_matrix(node, "compensation")

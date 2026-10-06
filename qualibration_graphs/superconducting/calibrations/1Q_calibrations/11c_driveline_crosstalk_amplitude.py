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

from calibration_utils.driveline_crosstalk import AmplitudeParameters as Parameters
from calibration_utils.driveline_crosstalk import (
    amplitude_factors,
    fit_amplitude,
    plot_matrix,
    plot_rabi,
    prepare_machine,
    save_matrix,
)
from quam_config import Quam

# %% {Description}
description = """DRIVE-LINE CROSSTALK AMPLITUDE
Measure every selected drive/target Rabi trace. Return r(source,target) =
a_pi(target->target) / a_pi(source->target), the target-normalized magnitude
needed for compensation. Self-drive uses a smaller amplitude sweep.
The active machine state is not updated. Full datasets and figures use normal
QUAlibrate storage; the matrix H5 is copied to QEC/crosstalktest.
"""

node = QualibrationNode[Parameters, Quam](
    name="11c_driveline_crosstalk_amplitude",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# %% {Prepare_machine}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def prepare(node: QualibrationNode[Parameters, Quam]):
    prepare_machine(node)


# %% {Create_QUA_program}
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
    """Create the amplitude sweep axes and the drive/target power-Rabi QUA program."""
    p, qubits = (node.parameters, node.namespace["qubits"])
    amps = amplitude_factors(p)
    targets = qubits
    drives = qubits
    axes = {"qubit": xr.DataArray([q.name for q in targets]), "drive_qubit": xr.DataArray([q.name for q in drives])}
    axes["amp_prefactor"] = xr.DataArray(amps)
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
                    modes = [None]
                    for mode in modes:
                        with for_(*from_array(a, amps)):
                            pair_phases = None
                            theta = 0
                            source_scale = p.self_drive_amp_scale if diagonal else 1
                            _measure_pair(target, drive, p, a * source_scale, None, theta, 1, states[j], streams[j])
        with stream_processing():
            n_st.save("n")
            for j, stream in enumerate(streams):
                result = stream.buffer(len(amps))
                result.buffer(len(drives)).average().save(f"state{j + 1}")
    node.namespace["qua_program"], node.namespace["sweep_axes"] = (qua_program, axes)


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


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = _execute_program(node, node.namespace["qua_program"], node.namespace["sweep_axes"])


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_id = node.parameters.load_data_id
    node.load_from_id(load_id)
    node.parameters.load_data_id = load_id


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"], node.results["ds_fit"], node.results["fit_results"] = fit_amplitude(
        node.results["ds_raw"], node.parameters
    )


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    node.results["figures"] = {
        "rabi": plot_rabi(node.results["ds_raw"], node.results["ds_fit"]),
        "amplitude_matrix": plot_matrix(node.results["ds_fit"], "amplitude"),
    }
    plt.show()


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    save_matrix(node, "amplitude")

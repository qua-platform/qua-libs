# %% {Imports}
import os

import numpy as np
import xarray as xr
from qm.qua import align, declare, declare_stream, fixed, for_, for_each_, program, reset_frame, save, stream_processing
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from calibration_utils.readout_photon_number import (
    RamseyParameters,
    fit_ramsey,
    plot_ramsey,
    sweep_values,
    validate_drive,
)
from quam_config import Quam

# %% {Description}
description = """
READOUT PHOTON NUMBER WITH RAMSEY
Apply x90, a square resonator drive with swept amplitude and duration, wait
for ring-down, apply a phase-swept x90 and read the qubit. Zero drive provides
a timing-matched reference. Fit finite-pulse dispersive coherence using chi
and kappa from a successful CKP run. Report ground-state steady-state photon
occupation at the same drive frequency and voltage, together with the CKP
comparison. Ramsey amplitudes with lost contrast are marked unresolved.
The two photon estimates share chi/kappa calibration; Ramsey errors reported
here are conditional statistical errors, not an independent absolute calibration.
Set PHOTON_CKP_DATA_ID to the saved 23f CKP node id before hardware execution.
Prerequisites:
    - Calibrated qubit frequency, x90/x180 gates and state discrimination.
    - A calibrated square resonator operation (readout_square by default).
    - Independent resonator and XY cores for overlapping drive and probe.

State update:
    - None; the core assignment changes only the generated run config.
"""

node = QualibrationNode[RamseyParameters, Quam](
    name="23e_readout_photon_ramsey",
    description=description,
    parameters=RamseyParameters(),
    machine=Quam.load(),
)


# %% {Custom_parameters}
# Local debugging overrides are ignored when run through the GUI or a graph.
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[RamseyParameters, Quam]):
    """Override local debugging parameters; GUI and graph runs use the parameter model."""
    node.parameters.qubits = [name.strip() for name in os.getenv("PHOTON_QUBITS", "qA1").split(",")]
    node.parameters.num_shots = int(os.getenv("PHOTON_SHOTS", "100"))
    node.parameters.amp_factors = [float(value) for value in os.getenv("PHOTON_AMPS", "0,0.2,0.4,0.6").split(",")]
    node.parameters.operation = os.getenv("PHOTON_OPERATION", "readout_square")
    node.parameters.drive_detuning_mhz = float(os.getenv("PHOTON_DRIVE_DETUNING_MHZ", "0"))
    node.parameters.reset_type = "thermal"
    node.parameters.multiplexed = False
    selected = os.getenv("PHOTON_CKP_DATA_ID")
    if selected:
        node.parameters.ckp_data_id = int(selected)


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[RamseyParameters, Quam]):
    """Register the sweep axes and generate the QUA pulse sequence."""
    parameters = node.parameters
    type(parameters).model_validate(parameters.model_dump())
    qubits = get_qubits(node)
    qubit_list = list(qubits)
    amps = np.asarray(parameters.amp_factors)
    node.namespace["qubits"] = qubits
    node.results["drive_metadata"] = validate_drive(qubit_list, parameters)
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_factor": xr.DataArray(amps, attrs={"long_name": "square readout amplitude factor"}),
    }
    if parameters.ckp_data_id is None:
        raise ValueError("Supply ckp_data_id (PHOTON_CKP_DATA_ID) from a successful 23f_readout_ckp run")
    calibration = QualibrationNode.load_from_id(parameters.ckp_data_id)
    node.results["ckp_fit_results"] = calibration.results["fit_results"]
    for qubit in qubit_list:
        previous = calibration.results["drive_metadata"][qubit.name]
        current = node.results["drive_metadata"][qubit.name]
        if previous != current or calibration.parameters.amp_factors != parameters.amp_factors:
            raise ValueError(
                "CKP and Ramsey must use identical amplitude factors, pulse voltage and drive/qubit frequencies"
            )
        result = calibration.results["fit_results"][qubit.name]
        if not result["success"]:
            raise ValueError(f"Unresolved CKP fit for {qubit.name}")
        if np.exp(-result["kappa_rad_per_ns"] * parameters.post_drive_idle_ns) > 0.01:
            raise ValueError("Increase post_drive_idle_ns so the second Ramsey gate sees less than 1% residual photons")
    durations = sweep_values(
        parameters.min_duration_ns, parameters.max_duration_ns, parameters.duration_step_ns
    ).astype(int)
    frames = np.arange(parameters.num_frame_rotations) / parameters.num_frame_rotations
    node.namespace["sweep_axes"].update(
        duration=xr.DataArray(durations, attrs={"long_name": "square drive duration", "units": "ns"}),
        frame=xr.DataArray(frames, attrs={"long_name": "analysis frame", "units": "2pi"}),
    )
    node.namespace["config"] = config = node.machine.generate_config()
    for qubit in qubit_list:
        # Match CKP's independent readout timeline without changing saved machine state.
        config["elements"][qubit.resonator.name]["core"] = f"{qubit.name}_photon_readout"
    with program() as node.namespace["qua_program"]:
        n = declare(int)
        amplitude = declare(fixed)
        n_st = declare_stream()
        states = [declare(int) for _ in qubit_list]
        state_streams = [declare_stream() for _ in qubit_list]
        duration = declare(int)
        frame = declare(fixed)
        for qubit in qubit_list:
            node.machine.initialize_qpu(target=qubit)
        align()
        with for_(n, 0, n < parameters.num_shots, n + 1):
            save(n, n_st)
            for index, qubit in enumerate(qubit_list):
                with for_each_(amplitude, amps):
                    with for_(*from_array(duration, durations // 4)):
                        with for_(*from_array(frame, frames)):
                            reset_frame(qubit.xy.name)
                            qubit.reset(parameters.reset_type, parameters.simulate)
                            align(qubit.xy.name, qubit.resonator.name)
                            qubit.xy.play("x90")
                            align(qubit.xy.name, qubit.resonator.name)
                            qubit.resonator.update_frequency(
                                int(qubit.resonator.intermediate_frequency + parameters.drive_detuning_mhz * 1e6)
                            )
                            qubit.resonator.play(parameters.operation, amplitude_scale=amplitude, duration=duration)
                            align(qubit.xy.name, qubit.resonator.name)
                            qubit.xy.wait(parameters.post_drive_idle_ns // 4)
                            qubit.xy.frame_rotation_2pi(frame)
                            qubit.xy.play("x90")
                            align(qubit.xy.name, qubit.resonator.name)
                            qubit.resonator.update_frequency(int(qubit.resonator.intermediate_frequency))
                            qubit.readout_state(states[index])
                            save(states[index], state_streams[index])
                            align()
        with stream_processing():
            n_st.save("n")
            for index, state_stream in enumerate(state_streams):
                state_stream.buffer(len(frames)).buffer(len(durations)).buffer(len(amps)).average().save(
                    f"state{index + 1}"
                )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[RamseyParameters, Quam]):
    """Simulate the pulse sequence and store samples and the waveform report."""
    samples, figure, report = simulate_and_plot(
        node.machine.connect(), node.namespace["config"], node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {"samples": samples, "figure": figure, "wf_report": report}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[RamseyParameters, Quam]):
    """Execute the pulse sequence and store the fetched raw dataset."""
    dataset = None
    with qm_session(node.machine.connect(), node.namespace["config"], timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_program"], options={"timeout": 600})
        fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in fetcher:
            progress_counter(fetcher.get("n", 0), node.parameters.num_shots, start_time=fetcher.t_start)
        node.log(job.execution_report())
    if dataset is None or "state" not in dataset:
        raise RuntimeError("Incomplete photon-characterization result streams")
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[RamseyParameters, Quam]):
    """Load a previously acquired dataset for offline analysis."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[RamseyParameters, Quam]):
    """Fit the Ramsey data and report resolved parameters and outcomes."""
    try:
        ds, results = fit_ramsey(node.results["ds_raw"], node.parameters, node.results["ckp_fit_results"])
    except Exception:
        # Keep the hardware data available for offline repair of an analysis failure.
        node.save()
        raise
    node.results["ds_fit"], node.results["fit_results"] = ds, results
    node.results["summary"] = results
    node.outcomes = {name: ("successful" if result["success"] else "failed") for name, result in results.items()}
    node.log(results)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[RamseyParameters, Quam]):
    """Plot the raw Ramsey data and fitted characterization results."""
    node.results["figures"] = plot_ramsey(node.results["ds_fit"], node.results["fit_results"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[RamseyParameters, Quam]):
    """Save the parameters, acquired data, analysis results and figures."""
    node.save()

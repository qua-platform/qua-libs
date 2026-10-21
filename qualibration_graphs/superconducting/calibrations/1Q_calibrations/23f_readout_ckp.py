# %% {Imports}
import os
from math import gcd

import numpy as np
import xarray as xr
from qm.qua import align, declare, declare_stream, fixed, for_, for_each_, program, reset_frame, save, stream_processing
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from quam.components.pulses import SquarePulse
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from calibration_utils.readout_photon_number import CKPParameters, fit_ckp, plot_ckp, sweep_values, validate_drive
from quam_config import Quam

# %% {Description}
description = """
CHI-KAPPA-POWER (CKP)
Prepare |0> or |1>, ring up a square resonator drive, and probe the shifted
qubit transition while that drive remains on. Sweep drive amplitude,
resonator frequency and qubit probe frequency. Let the resonator ring down
before calibrated state readout. Jointly fit the two Stark ridges to Eq. (7)
of Sank et al., Phys. Rev. Applied 23, 024055 (2025):
https://doi.org/10.1103/PhysRevApplied.23.024055
Return linewidth kappa/(2pi), signed dispersive separation 2chi/(2pi), chi,
and ground-state steady-state photon number at the comparison drive frequency.
Prerequisites:
    - Calibrated qubit frequency, x90/x180 gates and state discrimination.
    - A calibrated square resonator operation (readout_square by default).
    - Independent resonator and XY cores for overlapping drive and probe.

State update:
    - None; the core assignment changes only the generated run config.
"""

node = QualibrationNode[CKPParameters, Quam](
    name="23f_readout_ckp",
    description=description,
    parameters=CKPParameters(),
    machine=Quam.load(),
)


# %% {Custom_parameters}
# Local debugging overrides are ignored when run through the GUI or a graph.
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[CKPParameters, Quam]):
    """Override local debugging parameters; GUI and graph runs use the parameter model."""
    node.parameters.qubits = [name.strip() for name in os.getenv("PHOTON_QUBITS", "qA1").split(",")]
    node.parameters.num_shots = int(os.getenv("PHOTON_SHOTS", "100"))
    node.parameters.amp_factors = [float(value) for value in os.getenv("PHOTON_AMPS", "0,0.2,0.4,0.6").split(",")]
    node.parameters.operation = os.getenv("PHOTON_OPERATION", "readout_square")
    node.parameters.drive_detuning_mhz = float(os.getenv("PHOTON_DRIVE_DETUNING_MHZ", "0"))
    node.parameters.reset_type = "thermal"
    node.parameters.multiplexed = False
    node.parameters.resonator_span_mhz = float(os.getenv("PHOTON_CKP_SPAN_MHZ", "3"))
    node.parameters.qubit_step_mhz = float(os.getenv("PHOTON_CKP_QUBIT_STEP_MHZ", "0.5"))


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[CKPParameters, Quam]):
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
    resonator_offsets = np.rint(
        1e6 * sweep_values(-parameters.resonator_span_mhz, parameters.resonator_span_mhz, parameters.resonator_step_mhz)
    ).astype(int)
    probe_offsets = np.rint(
        1e6
        * sweep_values(parameters.min_qubit_detuning_mhz, parameters.max_qubit_detuning_mhz, parameters.qubit_step_mhz)
    ).astype(int)
    node.namespace["sweep_axes"].update(
        prepared_state=xr.DataArray([0, 1]),
        resonator_detuning=xr.DataArray(resonator_offsets / 1e6, attrs={"units": "MHz"}),
        qubit_detuning=xr.DataArray(probe_offsets / 1e6, attrs={"units": "MHz"}),
    )
    node.namespace["probe_scales"] = {}
    node.results["probe_metadata"] = {}
    for qubit in qubit_list:
        pulse = qubit.xy.operations[parameters.probe_operation]
        if not isinstance(pulse, SquarePulse) or pulse.amplitude == 0:
            raise ValueError("The CKP probe requires a nonzero square pulse")
        reference = qubit.xy.operations[parameters.reference_operation]
        pi_area = float(abs(np.mean(reference.calculate_waveform())) * reference.length)
        scale = parameters.probe_area_factor * pi_area / (parameters.probe_ns * abs(pulse.amplitude))
        if not 0 < scale < 2:
            raise ValueError("Increase probe_ns: required probe amplitude exceeds the QUA scaling range")
        node.namespace["probe_scales"][qubit.name] = scale
        node.results["probe_metadata"][qubit.name] = {
            "reference_operation": parameters.reference_operation,
            "pi_area_v_ns": pi_area,
            "probe_amplitude_scale": scale,
            "probe_duration_ns": parameters.probe_ns,
        }
    node.namespace["shots_per_batch"] = shots_per_batch = gcd(parameters.num_shots, parameters.max_shots_per_batch)
    node.namespace["num_batches"] = parameters.num_shots // shots_per_batch
    node.namespace["config"] = config = node.machine.generate_config()
    for qubit in qubit_list:
        # CKP needs simultaneous resonator drive and XY probe. Shared cores serialize them.
        config["elements"][qubit.resonator.name]["core"] = f"{qubit.name}_photon_readout"

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        amplitude = declare(fixed)
        n_st = declare_stream()
        states = [declare(int) for _ in qubit_list]
        state_streams = [declare_stream() for _ in qubit_list]
        resonator_offset = declare(int)
        probe_offset = declare(int)
        for qubit in qubit_list:
            node.machine.initialize_qpu(target=qubit)
        align()
        with for_(n, 0, n < shots_per_batch, n + 1):
            save(n, n_st)
            for index, qubit in enumerate(qubit_list):
                with for_each_(amplitude, amps):
                    for prepared_state in [0, 1]:
                        with for_(*from_array(resonator_offset, resonator_offsets)):
                            with for_(*from_array(probe_offset, probe_offsets)):
                                reset_frame(qubit.xy.name)
                                qubit.xy.update_frequency(int(qubit.xy.intermediate_frequency))
                                qubit.resonator.update_frequency(int(qubit.resonator.intermediate_frequency))
                                qubit.reset(parameters.reset_type, parameters.simulate)
                                align(qubit.xy.name, qubit.resonator.name)
                                if prepared_state:
                                    qubit.xy.play("x180")
                                align(qubit.xy.name, qubit.resonator.name)
                                qubit.resonator.update_frequency(
                                    int(qubit.resonator.intermediate_frequency) + resonator_offset
                                )
                                qubit.xy.update_frequency(int(qubit.xy.intermediate_frequency) + probe_offset)
                                # No align between ring-up and probe: the two channels overlap.
                                qubit.resonator.play(
                                    parameters.operation,
                                    amplitude_scale=amplitude,
                                    duration=(parameters.ringup_ns + parameters.probe_ns) // 4,
                                )
                                qubit.xy.wait(parameters.ringup_ns // 4)
                                qubit.xy.play(
                                    parameters.probe_operation,
                                    amplitude_scale=node.namespace["probe_scales"][qubit.name],
                                    duration=parameters.probe_ns // 4,
                                )
                                align(qubit.xy.name, qubit.resonator.name)
                                qubit.resonator.wait(parameters.ringdown_wait_ns // 4)
                                align(qubit.xy.name, qubit.resonator.name)
                                qubit.xy.update_frequency(int(qubit.xy.intermediate_frequency))
                                qubit.resonator.update_frequency(int(qubit.resonator.intermediate_frequency))
                                qubit.readout_state(states[index])
                                save(states[index], state_streams[index])
                                align()
        with stream_processing():
            n_st.save("n")
            for index, state_stream in enumerate(state_streams):
                state_stream.buffer(len(probe_offsets)).buffer(len(resonator_offsets)).buffer(2).buffer(
                    len(amps)
                ).average().save(f"state{index + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[CKPParameters, Quam]):
    """Simulate the pulse sequence and store samples and the waveform report."""
    samples, figure, report = simulate_and_plot(
        node.machine.connect(), node.namespace["config"], node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {"samples": samples, "figure": figure, "wf_report": report}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[CKPParameters, Quam]):
    """Execute the pulse sequence and store the fetched raw dataset."""
    datasets = []
    batches = node.namespace["num_batches"]
    with qm_session(node.machine.connect(), node.namespace["config"], timeout=node.parameters.timeout) as qm:
        for batch in range(batches):
            job = qm.execute(node.namespace["qua_program"], options={"timeout": 600})
            fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            dataset = None
            for dataset in fetcher:
                progress_counter(fetcher.get("n", 0), node.namespace["shots_per_batch"], start_time=fetcher.t_start)
            node.log(job.execution_report())
            if dataset is None or "state" not in dataset:
                raise RuntimeError(f"Incomplete CKP result streams in batch {batch + 1}")
            datasets.append(dataset)
            # Preserve completed batches in memory if a later request fails.
            node.results["ds_raw"] = xr.concat(datasets, dim="batch").mean("batch")
            node.results["completed_shots"] = (batch + 1) * node.namespace["shots_per_batch"]
            node.log(f"CKP batch {batch + 1}/{batches} completed")


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[CKPParameters, Quam]):
    """Load a previously acquired dataset for offline analysis."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[CKPParameters, Quam]):
    """Fit the CKP data and report resolved parameters and outcomes."""
    try:
        ds, results = fit_ckp(node.results["ds_raw"], node.parameters)
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
def plot_data(node: QualibrationNode[CKPParameters, Quam]):
    """Plot the raw CKP data and fitted characterization results."""
    node.results["figures"] = plot_ckp(node.results["ds_fit"], node.results["fit_results"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[CKPParameters, Quam]):
    """Save the parameters, acquired data, analysis results and figures."""
    node.save()

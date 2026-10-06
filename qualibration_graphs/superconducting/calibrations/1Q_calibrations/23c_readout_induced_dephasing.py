# %% {Imports}
import os

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
    save,
    stream_processing,
)
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from calibration_utils.readout_induced_dephasing import (
    Parameters,
    amplitude_factors,
    fit_raw_data,
    frame_rotations,
    log_fitted_results,
    plot_results,
    process_raw_dataset,
    summarize_fit_results,
)
from quam_config import Quam

# %% {Description}
description = """READOUT-INDUCED DEPHASING
Measure directed readout crosstalk with cross-Ramsey tomography. For each
aggressor Qi and distinct victim Qj, prepare Qj with x90, read Qi during the
Ramsey interval, sweep the phase of Qj's second x90, and finally measure Qj.

The Ramsey contrast and phase are fitted versus readout-amplitude factor xi:

    C(xi) = C0 exp(-lambda xi^2),    Delta phi(xi) = k xi^2.

At the requested amplitude factor, the node reports the extra dephasing rate,
the corresponding single-readout phase-flip probability, and the coherent Z
rotation for every directed off-diagonal pair.

Prerequisites:
    - Calibrated x90 gates and readout operations for all selected qubits.
    - State-discrimination parameters when use_state_discrimination is enabled.
    - Readout pulse lengths that are positive multiples of 4 ns.

Results:
    - Selected Ramsey line cuts and fitted phase shifts.
    - Directed phase-flip-probability and coherent-Z-rotation matrices.

This diagnostic node does not update machine state. Deterministic coherent
rotation may be compensated later with a virtual Z correction.
"""

node = QualibrationNode[Parameters, Quam](
    name="23c_readout_induced_dephasing",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# %% {Custom_parameters}
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set local debugging parameters; GUI and graph executions ignore these."""
    selected = os.getenv("QEC_RID_QUBITS")
    if selected:
        node.parameters.qubits = [name.strip() for name in selected.split(",")]
        node.parameters.num_shots = int(os.getenv("QEC_RID_SHOTS", "250"))
        node.parameters.min_amp_factor = float(os.getenv("QEC_RID_MIN_AMP", "0.0"))
        node.parameters.max_amp_factor = float(os.getenv("QEC_RID_MAX_AMP", "1.2"))
        node.parameters.amp_factor_step = float(os.getenv("QEC_RID_AMP_STEP", "0.2"))
        node.parameters.num_frame_rotations = int(os.getenv("QEC_RID_FRAMES", "16"))
        node.parameters.report_amp_factor = 1.0
        node.parameters.use_state_discrimination = True
        node.parameters.multiplexed = False
        node.parameters.reset_type = "thermal"


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the directed-pair mask, sweep axes, and cross-Ramsey QUA program."""
    parameters = node.parameters
    type(parameters).model_validate(parameters.model_dump())
    node.namespace["qubits"] = qubits = get_qubits(node)
    qubit_list = list(qubits)
    names = qubits.get_names()
    amp_factors = amplitude_factors(parameters)
    frames = frame_rotations(parameters)

    if len(qubit_list) < 2:
        raise ValueError("Select at least two qubits for a directed cross-Ramsey measurement")
    missing_operations = [qubit.name for qubit in qubit_list if parameters.operation not in qubit.resonator.operations]
    if missing_operations:
        raise ValueError(f"Readout operation {parameters.operation!r} is missing from {missing_operations}")
    durations = {qubit.name: int(qubit.resonator.operations[parameters.operation].length) for qubit in qubit_list}
    invalid_durations = {name: duration for name, duration in durations.items() if duration <= 0 or duration % 4}
    if invalid_durations:
        raise ValueError(f"Readout pulse lengths must be positive multiples of 4 ns: {invalid_durations}")

    node.namespace["measurement_mask"] = measurement_mask = xr.DataArray(
        np.not_equal.outer(names, names),
        dims=("qubit", "aggressor_qubit"),
        coords={"qubit": names, "aggressor_qubit": names},
    )
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(names),
        "aggressor_qubit": xr.DataArray(names),
        "amp_factor": xr.DataArray(amp_factors, attrs={"long_name": "readout amplitude factor"}),
        "frame": xr.DataArray(frames, attrs={"long_name": "analysis frame", "units": "2pi"}),
    }
    node.namespace["config"] = node.machine.generate_config()

    with program() as node.namespace["qua_program"]:
        n, amp_factor, frame = declare(int), declare(fixed), declare(fixed)
        n_st = declare_stream()
        states = [declare(int) for _ in qubit_list]
        state_st = [declare_stream() for _ in qubit_list]
        measured_i = [declare(fixed) for _ in qubit_list]
        measured_q = [declare(fixed) for _ in qubit_list]
        i_st = [declare_stream() for _ in qubit_list]
        q_st = [declare_stream() for _ in qubit_list]
        aggressor_i = [declare(fixed) for _ in qubit_list]
        aggressor_q = [declare(fixed) for _ in qubit_list]

        for qubit in qubit_list:
            node.machine.initialize_qpu(target=qubit)
        align()

        with for_(n, 0, n < parameters.num_shots, n + 1):
            save(n, n_st)
            for victim_index, victim in enumerate(qubit_list):
                for aggressor_index, aggressor in enumerate(qubit_list):
                    with for_(*from_array(amp_factor, amp_factors)):
                        with for_(*from_array(frame, frames)):
                            if victim_index != aggressor_index:
                                reset_frame(victim.xy.name)
                                victim.reset(parameters.reset_type, parameters.simulate)
                                aggressor.reset(parameters.reset_type, parameters.simulate)
                                align()

                                victim.xy.play("x90")
                                align(victim.xy.name, aggressor.resonator.name)
                                aggressor.resonator.measure(
                                    parameters.operation,
                                    qua_vars=(aggressor_i[aggressor_index], aggressor_q[aggressor_index]),
                                    amplitude_scale=amp_factor,
                                )
                                align(victim.xy.name, aggressor.resonator.name)

                                victim.xy.frame_rotation_2pi(frame)
                                victim.xy.play("x90")
                                align()

                                if parameters.use_state_discrimination:
                                    victim.readout_state(states[victim_index])
                                    save(states[victim_index], state_st[victim_index])
                                else:
                                    victim.resonator.measure(
                                        "readout",
                                        qua_vars=(measured_i[victim_index], measured_q[victim_index]),
                                    )
                                    save(measured_i[victim_index], i_st[victim_index])
                                    save(measured_q[victim_index], q_st[victim_index])
                                align()
                            else:
                                if parameters.use_state_discrimination:
                                    save(-1, state_st[victim_index])
                                else:
                                    save(0.0, i_st[victim_index])
                                    save(0.0, q_st[victim_index])

        with stream_processing():
            n_st.save("n")
            for index in range(len(qubit_list)):
                if parameters.use_state_discrimination:
                    state_st[index].buffer(len(frames)).buffer(len(amp_factors)).buffer(len(qubit_list)).average().save(
                        f"state{index + 1}"
                    )
                else:
                    i_st[index].buffer(len(frames)).buffer(len(amp_factors)).buffer(len(qubit_list)).average().save(
                        f"I{index + 1}"
                    )
                    q_st[index].buffer(len(frames)).buffer(len(amp_factors)).buffer(len(qubit_list)).average().save(
                        f"Q{index + 1}"
                    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Simulate the generated program and store samples and waveform report."""
    samples, figure, report = simulate_and_plot(
        node.machine.connect(),
        node.namespace["config"],
        node.namespace["qua_program"],
        node.parameters,
    )
    node.results["simulation"] = {"figure": figure, "wf_report": report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the cross-Ramsey sweep and fetch averaged results."""
    qmm = node.machine.connect()
    with qm_session(qmm, node.namespace["config"], timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"], options={"timeout": 300})
        fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        dataset = None
        for dataset in fetcher:
            progress_counter(fetcher.get("n", 0), node.parameters.num_shots, start_time=fetcher.t_start)
        node.log(job.execution_report())
    if dataset is None:
        raise RuntimeError("No readout-induced dephasing data were returned")
    for data_var in ("state", "I", "Q"):
        if data_var in dataset:
            dataset[data_var] = dataset[data_var].where(node.namespace["measurement_mask"])
    dataset["measurement_mask"] = node.namespace["measurement_mask"]
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously saved node run for offline analysis."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit Ramsey contrast and phase for every directed off-diagonal pair."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], node.results["fit_results"] = fit_raw_data(node.results["ds_raw"], node)
    log_fitted_results(node.results["fit_results"], node.log)
    node.results["summary"] = summarize_fit_results(node.results["fit_results"])
    summary = node.results["summary"]
    node.log(
        f"Successful pairs: {summary['num_successful_pairs']}; "
        f"mean P_phi={100 * summary['mean_phase_flip_probability']:.4g}%; "
        f"mean |Delta_phi|={summary['mean_abs_coherent_rotation_deg']:.4g} deg; "
        f"worst pair={summary['worst_pair']}"
    )
    node.outcomes = {
        pair: ("successful" if result["success"] else "failed") for pair, result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot Ramsey line cuts, phase shifts, and directed crosstalk matrices."""
    node.results["figures"] = plot_results(node.results["ds_fit"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist node parameters, datasets, fit results, and figures."""
    node.save()

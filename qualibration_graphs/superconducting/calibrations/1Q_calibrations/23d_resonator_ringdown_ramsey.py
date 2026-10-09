"""Measure resonator ring-down from the qubit Ramsey phase after readout."""

# %% {Imports}
import os

import xarray as xr
from qm.qua import align, declare, declare_stream, fixed, for_, program, reset_frame, save, stream_processing
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from calibration_utils.resonator_ringdown_ramsey import (
    Parameters,
    delay_values,
    fit_raw_data,
    frame_rotations,
    log_fitted_results,
    plot_results,
)
from quam_config import Quam

# %% {Description}
description = """
RESONATOR RING-DOWN WITH A RAMSEY PROBE

Reset the qubit, apply its configured readout pulse, wait a variable delay,
then run a fixed-duration phase-swept Ramsey experiment on the same qubit.
Residual resonator photons produce an AC-Stark phase and may reduce Ramsey
contrast.  Fit the Ramsey phase to

    phi(t) = A exp(-t/tau_ringdown) + phi_inf

and report tau_ringdown for every selected qubit.  The node does not update
machine state.
"""

node = QualibrationNode[Parameters, Quam](
    name="23d_resonator_ringdown_ramsey",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# %% {Custom_parameters}
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    selected = os.getenv("QEC_RINGDOWN_QUBITS", "qA1,qA4")
    node.parameters.qubits = [name.strip() for name in selected.split(",")]
    node.parameters.num_shots = int(os.getenv("QEC_RINGDOWN_SHOTS", "250"))
    node.parameters.operation = os.getenv("QEC_RINGDOWN_OPERATION", "readout")
    node.parameters.min_ringdown_delay_ns = int(os.getenv("QEC_RINGDOWN_MIN_NS", "16"))
    node.parameters.max_ringdown_delay_ns = int(os.getenv("QEC_RINGDOWN_MAX_NS", "3016"))
    node.parameters.ringdown_delay_step_ns = int(os.getenv("QEC_RINGDOWN_STEP_NS", "80"))
    node.parameters.ramsey_idle_ns = int(os.getenv("QEC_RINGDOWN_RAMSEY_NS", "200"))
    node.parameters.num_frame_rotations = int(os.getenv("QEC_RINGDOWN_FRAMES", "12"))
    node.parameters.use_state_discrimination = True
    node.parameters.multiplexed = False
    node.parameters.reset_type = "thermal"


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    parameters = node.parameters
    qubits = get_qubits(node)
    qubit_list = list(qubits)
    delays_ns = delay_values(parameters)
    delay_cycles = delays_ns // 4
    frames = frame_rotations(parameters)
    missing = [q.name for q in qubit_list if parameters.operation not in q.resonator.operations]
    if missing:
        raise ValueError(f"Missing resonator operation {parameters.operation!r}: {missing}")

    node.namespace.update(qubits=qubits, delays_ns=delays_ns, frames=frames)
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "ringdown_delay": xr.DataArray(delays_ns, attrs={"long_name": "delay after readout", "units": "ns"}),
        "frame": xr.DataArray(frames, attrs={"long_name": "analysis frame", "units": "2pi"}),
    }
    node.namespace["config"] = node.machine.generate_config()

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        delay = declare(int)
        frame = declare(fixed)
        n_st = declare_stream()
        states = [declare(int) for _ in qubit_list]
        state_streams = [declare_stream() for _ in qubit_list]
        readout_i = [declare(fixed) for _ in qubit_list]
        readout_q = [declare(fixed) for _ in qubit_list]

        for qubit in qubit_list:
            node.machine.initialize_qpu(target=qubit)
        align()

        with for_(n, 0, n < parameters.num_shots, n + 1):
            save(n, n_st)
            for index, qubit in enumerate(qubit_list):
                with for_(*from_array(delay, delay_cycles)):
                    with for_(*from_array(frame, frames)):
                        reset_frame(qubit.xy.name)
                        qubit.reset(parameters.reset_type, parameters.simulate)
                        align(qubit.xy.name, qubit.resonator.name)

                        # Ring up the resonator with the real calibrated readout.
                        qubit.resonator.measure(
                            parameters.operation,
                            qua_vars=(readout_i[index], readout_q[index]),
                        )
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.xy.wait(delay)

                        # Fixed Ramsey probe of residual-photon AC-Stark phase.
                        qubit.xy.play("x90")
                        qubit.xy.wait(parameters.ramsey_idle_ns // 4)
                        qubit.xy.frame_rotation_2pi(frame)
                        qubit.xy.play("x90")
                        align()
                        qubit.readout_state(states[index])
                        save(states[index], state_streams[index])
                        align()

        with stream_processing():
            n_st.save("n")
            for index, state_stream in enumerate(state_streams):
                state_stream.buffer(len(frames)).buffer(len(delays_ns)).average().save(f"state{index + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    samples, figure, report = simulate_and_plot(
        node.machine.connect(), node.namespace["config"], node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {"samples": samples, "figure": figure, "wf_report": report}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    dataset = None
    with qm_session(node.machine.connect(), node.namespace["config"], timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_program"], options={"timeout": 300})
        fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in fetcher:
            progress_counter(fetcher.get("n", 0), node.parameters.num_shots, start_time=fetcher.t_start)
        node.log(job.execution_report())
    if dataset is None or "state" not in dataset:
        returned = [] if dataset is None else sorted(dataset.data_vars)
        raise RuntimeError(f"Incomplete ring-down result streams: {returned}")
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit Ramsey phase decay and report per-qubit ring-down times."""
    node.results["ds_raw"], node.results["fit_results"] = fit_raw_data(node.results["ds_raw"])
    node.results["summary"] = node.results["fit_results"]
    node.outcomes = {
        name: ("successful" if result["success"] else "failed") for name, result in node.results["fit_results"].items()
    }
    log_fitted_results(node.results["fit_results"], node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the fitted phase decay and contrast recovery."""
    node.results["figures"] = plot_results(node.results["ds_raw"], node.results["fit_results"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

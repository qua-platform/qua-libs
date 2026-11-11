# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from calibration_utils.lru_phase_compensation import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from quam_config import Quam
from quam_config.lru_components import LRU_OPERATION, LruMacro, validate_lru_qubits

# %% {Node initialisation}
description = """
        f-LRU PHASE COMPENSATION
The microwave f-LRU pump (node 26a) shifts the qubit frequency while it plays (AC Stark shift), so a qubit in a
superposition picks up a phase during each LRU. This node measures that phase and stores the virtual Z that cancels
it, as node 34a does for the local phases of a CZ.

The sequence is a Ramsey around the LRU: x90 - LRU - virtual Z(theta) - x90, read as g/e. The LRU is played through
qubit.macros["lru"], so it already carries the stored correction, and theta is swept over one turn. P(e) peaks where
theta cancels what remains; the fitted peak gives the residual phase, and the new correction is the stored one minus
that residual. With num_lru_repetitions > 1, several LRUs are played between the x90 pulses to amplify a small
residual.

A qubit with no LRU macro gets one for this run (no correction). It is kept in the state only when the fit succeeds,
with the measured correction.

Prerequisites:
    - A microwave LRU stored on each qubit by 26a_lru_calibration (qubit.lru with the "lru" pulse).
    - Calibrated x90 pulse, qubit frequency (06a_ramsey) and g/e readout.

State update:
    - qubit.macros["lru"].phase_shift: virtual Z applied to qubit.xy after the LRU pump, in units of 2π.
"""

node = QualibrationNode[Parameters, Quam](
    name="26c_lru_phase_compensation",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # node.parameters.qubits = ["q1", "q2"]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    validate_lru_qubits(qubits)
    operation = node.parameters.operation
    n_avg = node.parameters.num_shots
    repetitions = node.parameters.num_lru_repetitions
    frames = np.arange(node.parameters.num_frames) / node.parameters.num_frames

    # Qubits without an LRU macro get one for this run; update_state decides whether it stays
    node.namespace["added_macros"] = []
    for qubit in qubits:
        if LRU_OPERATION not in qubit.macros:
            qubit.macros[LRU_OPERATION] = LruMacro(operation=operation)
            node.namespace["added_macros"].append(qubit.name)
            node.log(f"{qubit.name}: no LRU macro stored; measuring with no phase correction.")
        elif qubit.macros[LRU_OPERATION].operation != operation:
            raise ValueError(
                f"{qubit.name}: macros['{LRU_OPERATION}'] plays '{qubit.macros[LRU_OPERATION].operation}', "
                f"not the requested '{operation}'."
            )
    node.namespace["previous_phase_shifts"] = {q.name: float(q.macros[LRU_OPERATION].phase_shift) for q in qubits}

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "frame": xr.DataArray(frames, attrs={"long_name": "virtual Z after the LRU", "units": "2π"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()
        frame = declare(fixed)
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(frame, frames)):
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        reset_frame(qubit.xy.name)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x90")
                        for _ in range(repetitions):
                            # Pump, then the stored virtual Z
                            qubit.macros[LRU_OPERATION].apply()
                        qubit.xy.frame_rotation_2pi(frame)
                        qubit.xy.play("x90")
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.readout_state(state[i])
                        save(state[i], state_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                state_st[i].buffer(len(frames)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data into the xarray dataset "ds_raw"."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = qubits = get_qubits(node)
    # The correction the loaded run was measured with
    node.namespace["previous_phase_shifts"] = {
        q.name: float(q.macros[LRU_OPERATION].phase_shift) if LRU_OPERATION in q.macros else 0.0 for q in qubits
    }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit P(e) against the swept frame and compute the new correction for each qubit."""
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        name: ("successful" if result["success"] else "failed") for name, result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot P(e) against the swept frame with the fit."""
    fig = plot_raw_data_with_fit(node.results["ds_fit"], node.namespace["qubits"])
    plt.show()
    node.results["figures"] = {"phase": fig}


# %% {Keep_or_remove_LRU_macros}
@node.run_action()
def keep_or_remove_lru_macros(node: QualibrationNode[Parameters, Quam]):
    """Keep a macro added for this run only when its qubit succeeded, already holding the measured correction.

    A new macro is not a simple value that record_state_updates could offer as a proposal, and node.save()
    writes the machine as it is in memory, so this choice is made here (as node 26a does for new LRU elements).
    """
    for name in node.namespace.get("added_macros", []):
        qubit = node.machine.qubits[name]
        if node.outcomes.get(name) == "successful":
            qubit.macros[LRU_OPERATION].phase_shift = node.results["fit_results"][name]["phase_shift"]
            node.log(f"{name}: keeping the new LRU macro in the QUAM state.")
        else:
            del qubit.macros[LRU_OPERATION]


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Propose the new correction for every successful qubit whose LRU macro existed before this run."""
    added = node.namespace.get("added_macros", [])
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed" or q.name in added:
                continue
            q.macros[LRU_OPERATION].phase_shift = node.results["fit_results"][q.name]["phase_shift"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

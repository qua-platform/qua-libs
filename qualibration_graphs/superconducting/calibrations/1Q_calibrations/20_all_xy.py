# pylint: disable=R0801

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
from quam_config import Quam
from calibration_utils.all_xy import (
    ALL_XY_SEQUENCES,
    N_ALL_XY,
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        ALL-XY SEQUENCE
The All-XY sequence consists of 21 pairs of single-qubit gates (I, X, Y, X-Y, Y-X, etc.)
designed to probe gate errors and phase coherence. Each sequence returns the qubit to
the ground or excited state in the ideal case; deviations indicate miscalibration.

References:
    - https://doi.org/10.1103/PRXQuantum.2.040202
    - M. Reed, PhD thesis (Yale, 2013):
      https://rsl.yale.edu/sites/default/files/2024-08/2013-RSL-Thesis-Matthew-Reed.pdf

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).
    - Having calibrated x90 / y90 / y180 as well as x180.
    - (optional) Having calibrated state discrimination (node 07_iq_blobs) if use_state_discrimination is True.
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - None (no machine-state update). Outcomes succeed if RMS vs the ideal
      All-XY staircase (normalized to [0, 1]) is below rms_threshold.
"""


node = QualibrationNode[Parameters, Quam](
    name="20_all_xy",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    pass


def play_all_xy_gate(qubit, gate: str) -> None:
    """Play a named All-XY gate; identity is a wait matching x90 duration."""
    if gate == "I":
        qubit.xy.wait(qubit.xy.operations["x90"].length // 4)
    else:
        qubit.xy.play(gate)


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for the 21 All-XY sequences."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    sequence_indices = np.arange(N_ALL_XY)
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "sequence_index": xr.DataArray(
            sequence_indices,
            attrs={"long_name": "All-XY sequence index", "units": ""},
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        seq_idx = declare(int)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(seq_idx, sequence_indices)):
                    for qubit in multiplexed_qubits.values():
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()

                    with switch_(seq_idx):
                        for seq_index, (g1, g2) in enumerate(ALL_XY_SEQUENCES):
                            with case_(seq_index):
                                for qubit in multiplexed_qubits.values():
                                    play_all_xy_gate(qubit, g1)
                                    play_all_xy_gate(qubit, g2)

                    align()
                    for i, qubit in multiplexed_qubits.items():
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(N_ALL_XY).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(N_ALL_XY).average().save(f"I{i + 1}")
                    Q_st[i].buffer(N_ALL_XY).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch the raw data into ds_raw."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
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


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Score each qubit against the ideal All-XY staircase and set outcomes."""
    node.results["ds_proc"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_proc"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot processed All-XY data against ideal reference levels."""
    fig_raw = plot_raw_data_with_fit(
        node.results["ds_proc"],
        node.namespace["qubits"],
    )
    plt.show()
    node.results["figures"] = {
        "all_xy": fig_raw,
    }


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()

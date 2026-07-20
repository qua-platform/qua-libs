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

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - (Skeleton: no state update.)
"""


node = QualibrationNode[Parameters, Quam](
    name="22b_all_xy",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # node.parameters.qubits = ["q1", "q2"]


# %%
# All-XY sequences: 21 pairs of single-qubit gates (Reed's thesis / Phys. Rev. A 82).
# "I" = identity (wait). Names must match qubit.xy.operations.
ALL_XY_SEQUENCES = [
    ("I", "I"),
    ("x180", "x180"),
    ("y180", "y180"),
    ("x180", "y180"),
    ("y180", "x180"),
    ("x90", "I"),
    ("y90", "I"),
    ("x90", "y90"),
    ("y90", "x90"),
    ("x90", "y180"),
    ("y90", "x180"),
    ("x180", "y90"),
    ("y180", "x90"),
    ("x90", "x180"),
    ("x180", "x90"),
    ("y90", "y180"),
    ("y180", "y90"),
    ("x180", "I"),
    ("y180", "I"),
    ("x90", "x90"),
    ("y90", "y90"),
]
N_ALL_XY = len(ALL_XY_SEQUENCES)


# %% {Create_QUA_program}
# pylint: disable=too-many-branches,too-many-statements
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
        seq_idx = declare(int)
        n = declare(int)
        n_st = declare_stream()

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

                    # Play the All-XY gate pair for this sequence index (switch/case: index -> gates)
                    with switch_(seq_idx):
                        with case_(0):  # I, I
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.wait(qubit.xy.operations["x90"].length // 4)
                                qubit.xy.wait(qubit.xy.operations["x90"].length // 4)
                        with case_(1):  # x180, x180
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x180")
                                qubit.xy.play("x180")
                        with case_(2):  # y180, y180
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("y180")
                                qubit.xy.play("y180")
                        with case_(3):  # x180, y180
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x180")
                                qubit.xy.play("y180")
                        with case_(4):  # y180, x180
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("y180")
                                qubit.xy.play("x180")
                        with case_(5):  # x90, I
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x90")
                                qubit.xy.wait(qubit.xy.operations["x90"].length // 4)
                        with case_(6):  # y90, I
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("y90")
                                qubit.xy.wait(qubit.xy.operations["x90"].length // 4)
                        with case_(7):  # x90, y90
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x90")
                                qubit.xy.play("y90")
                        with case_(8):  # y90, x90
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("y90")
                                qubit.xy.play("x90")
                        with case_(9):  # x90, y180
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x90")
                                qubit.xy.play("y180")
                        with case_(10):  # y90, x180
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("y90")
                                qubit.xy.play("x180")
                        with case_(11):  # x180, y90
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x180")
                                qubit.xy.play("y90")
                        with case_(12):  # y180, x90
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("y180")
                                qubit.xy.play("x90")
                        with case_(13):  # x90, x180
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x90")
                                qubit.xy.play("x180")
                        with case_(14):  # x180, x90
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x180")
                                qubit.xy.play("x90")
                        with case_(15):  # y90, y180
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("y90")
                                qubit.xy.play("y180")
                        with case_(16):  # y180, y90
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("y180")
                                qubit.xy.play("y90")
                        with case_(17):  # x180, I
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x180")
                                qubit.xy.wait(qubit.xy.operations["x90"].length // 4)
                        with case_(18):  # y180, I
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("y180")
                                qubit.xy.wait(qubit.xy.operations["x90"].length // 4)
                        with case_(19):  # x90, x90
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x90")
                                qubit.xy.play("x90")
                        with case_(20):  # y90, y90
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("y90")
                                qubit.xy.play("y90")

                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.readout_state(state[i])
                        save(state[i], state_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                state_st[i].buffer(N_ALL_XY).average().save(f"state{i + 1}")


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


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""
    fig_raw = plot_raw_data_with_fit(
        node.results["ds_raw"],
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

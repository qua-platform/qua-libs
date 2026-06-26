# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome

from quam_config import Quam

from calibration_utils.common_utils.experiment import (
    get_qubit_pairs,
    progress_counter_with_log,
    suppress_fetcher_axis_log_spam,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.singlet_triplet_oscillations import (
    Parameters,
    analyse_singlet_triplet_oscillations,
    log_fitted_results,
    plot_singlet_triplet_oscillations,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        SINGLET-TRIPLET OSCILLATIONS
This node measures singlet-triplet oscillations by preparing a qubit-pair state,
waiting for a variable duration, and then measuring the dot-pair state.

Sequence:
    qubit_pair.quantum_dot_pair.initialize() -> wait(duration) -> measure()

The wait duration is swept in 1D. The analysis extracts the dominant oscillation
frequency and period, and overlays a sinusoidal fit on the measured trace.
"""

node = QualibrationNode[Parameters, Quam](
    name="07e_singlet_triplet_oscillations",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local parameter edits for debugging."""
    pass


node.machine = Quam.load()


def _resolve_qubit_pair(node: QualibrationNode[Parameters, Quam]):
    """Resolve the single qubit pair requested by node parameters."""
    if node.parameters.qubit_pair not in (None, ""):
        return node.machine.qubit_pairs[node.parameters.qubit_pair]

    available_pairs = list(get_qubit_pairs(node))
    if not available_pairs:
        raise ValueError("No qubit pairs available to run this node.")
    return available_pairs[0]


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create sweep axis and generate singlet-triplet oscillation program."""
    qubit_pair = _resolve_qubit_pair(node)
    dot_pair = qubit_pair.quantum_dot_pair

    node.namespace["qubit_pair"] = qubit_pair
    node.namespace["dot_pair"] = dot_pair

    wait_min = int(node.parameters.min_wait_duration_in_ns)
    wait_max = int(node.parameters.max_wait_duration_in_ns)
    wait_step = int(node.parameters.wait_duration_step_in_ns)
    if wait_min % 4 != 0 or wait_max % 4 != 0 or wait_step % 4 != 0:
        raise ValueError(
            f"Wait settings must be divisible by 4. "
            f"Got min={wait_min}, max={wait_max}, step={wait_step}"
        )

    wait_ns_array = np.arange(wait_min, wait_max, wait_step, dtype=int)
    wait_cc_array = (wait_ns_array // 4).astype(int)

    node.namespace["sweep_axes"] = {
        "wait_duration": xr.DataArray(
            wait_ns_array,
            attrs={"long_name": "wait duration", "units": "ns"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_output_stream()
        heralded_and_return_n_loops = getattr(node.parameters, "return_n_loops", False)
        n_loops_st = declare_output_stream() if heralded_and_return_n_loops else None
        wait_cc = declare(int)

        state_int = declare(int)
        state_st = declare_output_stream()
        i_st = declare_output_stream()
        q_st = declare_output_stream()

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            with for_(*from_array(wait_cc, wait_cc_array)):
                n_init = dot_pair.initialize(
                    qubit_role="control",
                )
                if heralded_and_return_n_loops:
                    save(n_init, n_loops_st)
                wait(wait_cc)
                (i, q, state) = dot_pair.measure(return_iq=True)

                assign(state_int, Cast.to_int(state))
                save(state_int, state_st)
                save(i, i_st)
                save(q, q_st)

                align()
                dot_pair.voltage_sequence.ramp_to_zero()

        with stream_processing():
            n_st.save("n")
            state_st.buffer(len(wait_cc_array)).average().save(f"state_{qubit_pair.name}")
            i_st.buffer(len(wait_cc_array)).average().save(f"I_{qubit_pair.name}")
            q_st.buffer(len(wait_cc_array)).average().save(f"Q_{qubit_pair.name}")
            if heralded_and_return_n_loops:
                n_loops_st.buffer(len(wait_cc_array)).average().save(
                    f"n_loops_{qubit_pair.name}"
                )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch raw data."""
    suppress_fetcher_axis_log_spam()
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node,
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
    node.namespace["qubit_pair"] = _resolve_qubit_pair(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Extract oscillation frequency/period and fit a sinusoidal model."""
    qubit_pair = node.namespace["qubit_pair"]

    ds_fit, fit_result = analyse_singlet_triplet_oscillations(
        node.results["ds_raw"],
        qubit_pair.name,
        analysis_signal=node.parameters.analysis_signal,
    )
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {qubit_pair.name: fit_result}
    log_fitted_results(qubit_pair.name, fit_result, log_callable=node.log)

    node.outcomes = {
        qubit_pair.name: (
            Outcome.SUCCESSFUL if fit_result.get("success", False) else Outcome.FAILED
        )
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot oscillation traces and FFT summary."""
    qubit_pair = node.namespace["qubit_pair"]
    fig = plot_singlet_triplet_oscillations(
        node.results["ds_raw"],
        qubit_pair.name,
        ds_fit=node.results.get("ds_fit"),
    )
    node.results["figure"] = fig
    node.results["figures"] = {"singlet_triplet_oscillations": fig}
    annotate_node_figures(node)
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Placeholder for optional macro updates based on fitted oscillation period."""
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

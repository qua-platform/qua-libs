# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome
from quam_config import Quam

from calibration_utils.init_ramp_rate import (
    Parameters,
    analyse_ramp_rate,
    log_fitted_results,
    plot_avg_state_vs_ramp_duration,
    plot_iq_vs_ramp_duration,
    plot_q_density_vs_ramp_duration,
    plot_i_density_vs_ramp_duration,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        INITIALISATION RAMP RATE CALIBRATION
This sequence calibrates the ramp duration of the initialisation macro by sweeping the ramp rate
and measuring how mixed the resultant state is.

For each ramp duration the sequence empties the dots, initialises with the given ramp duration,
then performs a state measurement using the balanced measurement macro.  The boolean state
assignment (0 or 1) is averaged over many shots to produce the mean state occupation for each
ramp duration.

The analysis identifies the ramp duration that yields the minimum (or maximum, controlled by the
``find_minimum`` parameter) average state assignment, corresponding to the purest initialisation.

Prerequisites:
    - Having initialised the Quam.
    - Having calibrated the PSB measurement point (06a-06c).
    - Having the balanced measurement macro configured with a valid threshold.

State update:
    - The initialisation macro ``ramp_duration`` on each qubit pair.
"""

node = QualibrationNode[Parameters, Quam](
    name="07_init_ramp_rate_calibration",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes."""
    pass


node.machine = Quam.load()


def _resolve_qubit_pairs(node: QualibrationNode[Parameters, Quam]):
    """Resolve qubit pairs from parameters or default to all machine pairs."""
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    qubit_pairs = _resolve_qubit_pairs(node)
    dot_pair_objects = [qp.quantum_dot_pair for qp in qubit_pairs]

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["dot_pairs"] = dot_pair_objects

    ramp_min = int(node.parameters.ramp_duration_min)
    ramp_max = int(node.parameters.ramp_duration_max)
    ramp_step = int(node.parameters.ramp_duration_step)
    if ramp_min % 4 != 0 or ramp_max % 4 != 0 or ramp_step % 4 != 0:
        raise ValueError(
            f"Ramp settings must be divisible by 4. "
            f"Got min={ramp_min}, max={ramp_max}, step={ramp_step}"
        )

    if node.parameters.ramp_log_scale: 
        n_ramp_pts = int((ramp_max - ramp_min)//ramp_step)
        ramp_duration_array = np.logspace(ramp_min, ramp_max, n_ramp_pts, dtype=int, endpoint = True)
    else: 
        ramp_duration_array = np.arange(ramp_min, ramp_max, ramp_step, dtype=int)

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([qp.name for qp in qubit_pairs]),
        "shot": xr.DataArray(np.arange(node.parameters.num_shots)),
        "ramp_duration": xr.DataArray(
            ramp_duration_array,
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_output_stream()
        heralded_and_return_n_loops = getattr(node.parameters, "return_n_loops", False)
        n_loops_st = (
            {qp.name: declare_output_stream() for qp in qubit_pairs}
            if heralded_and_return_n_loops
            else {}
        )
        ramp_dur = declare(int)

        state_int = {qp.name: declare(int) for qp in qubit_pairs}
        state_st = {qp.name: declare_output_stream() for qp in qubit_pairs}

        i_st = {qp.name: declare_output_stream() for qp in qubit_pairs}
        q_st = {qp.name: declare_output_stream() for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                dot_pair = qubit_pair.quantum_dot_pair

                with for_(*from_array(ramp_dur, ramp_duration_array)):

                    n_init = dot_pair.initialize(
                        ramp_duration=ramp_dur,
                    )
                    if heralded_and_return_n_loops:
                        save(n_init, n_loops_st[qubit_pair.name])
                    (i, q, state) = dot_pair.measure(return_iq=True)

                    assign(
                        state_int[qubit_pair.name],
                        Cast.to_int(state),
                    )

                    align()
                    dot_pair.voltage_sequence.ramp_to_zero()

                    save(state_int[qubit_pair.name], state_st[qubit_pair.name])
                    save(i, i_st[qubit_pair.name])
                    save(q, q_st[qubit_pair.name])

        with stream_processing():
            n_st.save("n")
            for qp in qubit_pairs:
                state_st[qp.name].buffer(len(ramp_duration_array)).buffer(
                    node.parameters.num_shots
                ).save(f"state_{qp.name}")
                i_st[qp.name].buffer(len(ramp_duration_array)).buffer(
                    node.parameters.num_shots
                ).save(f"I_{qp.name}")
                q_st[qp.name].buffer(len(ramp_duration_array)).buffer(
                    node.parameters.num_shots
                ).save(f"Q_{qp.name}")
                if heralded_and_return_n_loops:
                    n_loops_st[qp.name].buffer(len(ramp_duration_array)).average().save(
                        f"n_loops_{qp.name}"
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
    """Connect to the QOP, execute the QUA program and fetch the raw data."""
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
                node=node
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
    node.namespace["qubit_pairs"] = _resolve_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Find the optimal ramp duration from averaged state assignment data."""
    qubit_pairs = node.namespace["qubit_pairs"]
    qp_names = [qp.name for qp in qubit_pairs]

    ds_raw, fit_results = analyse_ramp_rate(
        node.results["ds_raw"],
        qp_names,
        find_minimum=node.parameters.find_minimum,
    )
    node.results["ds_raw"] = ds_raw
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qp_name: (Outcome.SUCCESSFUL if r["success"] else Outcome.FAILED)
        for qp_name, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot average state assignment and IQ signal vs ramp duration."""
    qubit_pairs = node.namespace["qubit_pairs"]
    qp_names = [qp.name for qp in qubit_pairs]

    fig_state = plot_avg_state_vs_ramp_duration(
        node.results["ds_raw"],
        qp_names,
        fit_results=node.results.get("fit_results"),
    )
    fig_iq = plot_iq_vs_ramp_duration(node.results["ds_raw"], qp_names)
    fig_q_density = plot_q_density_vs_ramp_duration(node.results["ds_raw"], qp_names)
    fig_i_density = plot_i_density_vs_ramp_duration(node.results["ds_raw"], qp_names)

    node.results["figure"] = fig_state
    node.results["figures"] = {
        "avg_state_vs_ramp_duration": fig_state,
        "iq_vs_ramp_duration": fig_iq,
        "q_density_vs_ramp_duration": fig_q_density,
        "i_density_vs_ramp_duration": fig_i_density,
    }
    annotate_node_figures(node)
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the initialisation macro ramp_duration on each qubit pair."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            fit_result = node.results["fit_results"].get(qp.name, {})
            if not fit_result.get("success", False):
                continue

            dot_pair = qp.quantum_dot_pair
            optimal_ramp = fit_result["optimal_ramp_duration"]

            init_macro = dot_pair.macros.get("initialize")
            if init_macro is not None and hasattr(init_macro, "update"):
                init_macro.update(ramp_duration=optimal_ramp)
            else:
                node.log(
                    f"  {qp.name}: no updatable initialise macro found on "
                    f"{dot_pair.name}"
                )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

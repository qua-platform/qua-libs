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

from calibration_utils.init_2d import (
    Parameters,
    plot_2d_summary,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        INITIALISATION 2D CALIBRATION (RAMP DURATION × WAIT DURATION)
This sequence extends the ramp-rate calibration by adding a second sweep axis: the wait
duration between the initialisation ramp and the state measurement.

For each (ramp_duration, wait_duration) point the sequence empties the dots, initialises
with the given ramp duration, waits for the specified duration, then performs a state
measurement using the balanced measurement macro.  The boolean state assignment (0 or 1) is
averaged over many shots to produce a 2D map of mean state occupation.

The analysis identifies the (ramp_duration, wait_duration) pair that yields the minimum
(or maximum, controlled by the ``find_minimum`` parameter) average state assignment.

Prerequisites:
    - Having initialised the Quam.
    - Having calibrated the PSB measurement point (06a-06c).
    - Having the balanced measurement macro configured with a valid threshold.

State update:
    - The initialisation macro ``ramp_duration`` on each qubit pair.
"""

node = QualibrationNode[Parameters, Quam](
    name="07a_init_2d_calibration",
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
    """Create the sweep axes and generate the QUA program with a 2D sweep over ramp duration and wait duration."""
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

    wait_min = int(node.parameters.wait_duration_min)
    wait_max = int(node.parameters.wait_duration_max)
    wait_step = int(node.parameters.wait_duration_step)
    if wait_min % 4 != 0 or wait_max % 4 != 0 or wait_step % 4 != 0:
        raise ValueError(
            f"Wait settings must be divisible by 4. "
            f"Got min={wait_min}, max={wait_max}, step={wait_step}"
        )
    if wait_min < 16:
        raise ValueError(
            f"Minimum wait duration must be >= 16 ns (4 clock cycles). Got {wait_min}"
        )

    ramp_duration_array = np.arange(ramp_min, ramp_max, ramp_step, dtype=int)
    wait_ns_array = np.arange(wait_min, wait_max, wait_step, dtype=int)
    wait_cc_array = (wait_ns_array // 4).astype(int)

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([qp.name for qp in qubit_pairs]),
        "ramp_duration": xr.DataArray(
            ramp_duration_array,
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
        "wait_duration": xr.DataArray(
            wait_ns_array,
            attrs={"long_name": "wait duration", "units": "ns"},
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
        wait_dur = declare(int)

        state_int = {qp.name: declare(int) for qp in qubit_pairs}
        state_st = {qp.name: declare_output_stream() for qp in qubit_pairs}

        i_st = {qp.name: declare_output_stream() for qp in qubit_pairs}
        q_st = {qp.name: declare_output_stream() for qp in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                dot_pair = qubit_pair.quantum_dot_pair

                with for_(*from_array(ramp_dur, ramp_duration_array)):
                    with for_(*from_array(wait_dur, wait_cc_array)):

                        n_init = dot_pair.initialize(
                            ramp_duration=ramp_dur,
                        )
                        if heralded_and_return_n_loops:
                            save(n_init, n_loops_st[qubit_pair.name])
                        wait(wait_dur)
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
                # inner buffer = wait (innermost loop), outer buffer = ramp
                state_st[qp.name].buffer(len(wait_cc_array)).buffer(
                    len(ramp_duration_array)
                ).average().save(f"state_{qp.name}")
                i_st[qp.name].buffer(len(wait_cc_array)).buffer(
                    len(ramp_duration_array)
                ).average().save(f"I_{qp.name}")
                q_st[qp.name].buffer(len(wait_cc_array)).buffer(
                    len(ramp_duration_array)
                ).average().save(f"Q_{qp.name}")
                if heralded_and_return_n_loops:
                    n_loops_st[qp.name].buffer(len(wait_cc_array)).buffer(
                        len(ramp_duration_array)
                    ).average().save(f"n_loops_{qp.name}")


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
    """Find the optimal (ramp_duration, wait_duration) from the 2D state map."""
    qubit_pairs = node.namespace["qubit_pairs"]

    fit_results = {}
    for qp in qubit_pairs:
        avg_state = node.results["ds_raw"][f"state_{qp.name}"].values
        ramp_durations = node.results["ds_raw"]["ramp_duration"].values
        wait_durations = node.results["ds_raw"]["wait_duration"].values

        if node.parameters.find_minimum:
            opt_flat = int(np.argmin(avg_state))
        else:
            opt_flat = int(np.argmax(avg_state))

        opt_ramp_idx, opt_wait_idx = np.unravel_index(opt_flat, avg_state.shape)

        fit_results[qp.name] = {
            "success": True,
            "optimal_ramp_duration": int(ramp_durations[opt_ramp_idx]),
            "optimal_wait_duration": int(wait_durations[opt_wait_idx]),
            "optimal_avg_state": float(avg_state[opt_ramp_idx, opt_wait_idx]),
            "find_minimum": node.parameters.find_minimum,
        }

    node.results["fit_results"] = fit_results

    extremum = "minimum" if node.parameters.find_minimum else "maximum"
    for qp_name, r in fit_results.items():
        node.log(
            f"  {qp_name}: optimal ramp={r['optimal_ramp_duration']} ns, "
            f"wait={r['optimal_wait_duration']} ns "
            f"({extremum} avg state = {r['optimal_avg_state']:.4f})"
        )

    node.outcomes = {
        qp_name: (Outcome.SUCCESSFUL if r["success"] else Outcome.FAILED)
        for qp_name, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot 2D heatmaps of state and IQ signal vs ramp duration and wait duration."""
    qubit_pairs = node.namespace["qubit_pairs"]
    qp_names = [qp.name for qp in qubit_pairs]

    fig_summary = plot_2d_summary(
        node.results["ds_raw"],
        qp_names,
        fit_results=node.results.get("fit_results"),
    )

    node.results["figure"] = fig_summary
    node.results["figures"] = {
        "summary_2d": fig_summary,
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

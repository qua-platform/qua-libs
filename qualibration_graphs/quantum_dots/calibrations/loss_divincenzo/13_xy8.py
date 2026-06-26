# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.loops import from_array

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.xy8_parity_diff import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.common_utils.experiment import (
    get_qubits,
    enable_dual_drive_mw,
    progress_counter_with_log,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.parity_streams import (
    declare_parity_streams,
    save_parity_measurement,
    buffer_parity_streams,
    process_parity_streams,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        XY8 DYNAMICAL DECOUPLING T2 MEASUREMENT - using standard QUA (pulse > 16ns and 4ns granularity)
The goal of this script is to measure the qubit coherence time under XY8 dynamical decoupling.
The XY8 sequence applies 8 refocusing pi pulses with alternating X and Y rotation axes,
using CPMG timing (half-intervals at bookends, full intervals between pulses).

Unlike the Hahn echo (single pi pulse), XY8 filters higher-frequency noise components
and cancels pulse imperfections to first order through the XYXY·YXYX alternation pattern.
The resulting T2_XY8 is typically longer than T2_echo (Hahn echo) and reflects the
coherence limit set by noise at the DD filter frequency 1/(2τ).

The QUA program is divided into three sections:
    1) step between the initialization point and the operation point using sticky elements.
    2) apply the XY8 pulse sequence with CPMG timing:
       pi/2 - τ - X - 2τ - Y - 2τ - X - 2τ - Y - 2τ - Y - 2τ - X - 2τ - Y - 2τ - X - τ - pi/2
    3) measure the state of the qubit using RF reflectometry via parity readout.

Total idle time per sweep point: 16τ (2 half-intervals + 7 full intervals).

The measurement sweeps the half inter-pulse spacing τ (joint-outcome streams).
The fitting uses profiled differential evolution: a 1-D global search over T2_xy8 with
the linear parameters (offset, amplitude) solved analytically at each step via least-squares.

Prerequisites:
    - Having run the Hahn echo node (12) and its prerequisites.
    - Having calibrated pi and pi/2 pulse parameters from Rabi measurements.
    - Having calibrated y180 pulse (same amplitude as x180, axis_angle=pi/2).

Before proceeding to the next node:
    - Compare T2_XY8 to T2_echo to assess the benefit of dynamical decoupling.
    - Consider whether the coherence is pulse-error limited or decoherence limited.

State update:
    - None (diagnostic measurement)
"""


node = QualibrationNode[Parameters, Quam](
    name="13_xy8", description=description, parameters=Parameters()
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]
    # node.parameters.num_shots = 10
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for the XY8 sequence.

    Sweeps half inter-pulse spacing τ. Pulse sequence per sweep point:

        empty → measure(p1) → initialise → x90 → [τ-X-2τ-Y-2τ-X-2τ-Y-2τ-Y-2τ-X-2τ-Y-2τ-X-τ] → x90 → measure(p2)
    """
    node.namespace["qubits"] = qubits = get_qubits(node)

    n_avg = node.parameters.num_shots
    tau_values = np.arange(
        node.parameters.tau_min,
        node.parameters.tau_max,
        node.parameters.tau_step,
    )
    tau_clock_cycles = tau_values // 4

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(
            tau_values, attrs={"long_name": "half inter-pulse spacing", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw(node)

        t = declare(int)
        n = declare(int)

        p2, p1, parity_streams = declare_parity_streams(node, qubits)

        n_st = declare_output_stream()
        heralded_and_return_n_loops = getattr(node.parameters, "return_n_loops", False)
        n_loops_st = (
            {qubit.name: declare_output_stream() for qubit in qubits}
            if heralded_and_return_n_loops
            else {}
        )

        for qubit in qubits:
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(t, tau_clock_cycles)):
                    reset_frame(qubit.xy.name)

                    if node.parameters.parity_pre_measurement:
                        qubit.empty()
                        a1 = qubit.measure()

                    n_init = qubit.initialize()
                    if heralded_and_return_n_loops:
                        save(n_init, n_loops_st[qubit.name])

                    align()

                    with strict_timing_():
                        # Opening pi/2
                        qubit.x90()

                        # First half-interval (τ)
                        wait(t, qubit.xy.name)

                        # Pulse 1: X
                        qubit.x180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 2: Y
                        qubit.y180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 3: X
                        qubit.x180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 4: Y
                        qubit.y180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 5: Y
                        qubit.y180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 6: X
                        qubit.x180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 7: Y
                        qubit.y180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 8: X
                        qubit.x180()

                        # Last half-interval (τ)
                        wait(t, qubit.xy.name)

                        # Closing pi/2
                        qubit.x90()

                    align()

                    a2 = qubit.measure()

                    qubit.voltage_sequence.ramp_to_zero()

                    align()

                    assign(p2, Cast.to_int(a2))

                    if node.parameters.parity_pre_measurement:
                        assign(p1, Cast.to_int(a1))

                    save_parity_measurement(node, qubit.name, p1, p2, parity_streams)

        with stream_processing():
            n_st.save("n")

            n_tau = len(tau_values)
            for qubit in qubits:
                buffer_parity_streams(node, qubit.name, parity_streams, n_tau)
                if heralded_and_return_n_loops:
                    n_loops_st[qubit.name].buffer(n_tau).average().save(
                        f"n_loops_{qubit.name}"
                    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
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
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
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
    node.namespace["qubits"] = get_qubits(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome streams."""
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [q.name for q in node.namespace["qubits"]],
        parity_pre_measurement=node.parameters.parity_pre_measurement,
        sweep_dims=("tau",),
    )

# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit an exponential decay to the XY8 data for each qubit.

    Stores ``fit_results`` dict and a fitted xarray Dataset ``ds_fit``.
    """
    ds_fit, fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qname: ("successful" if r["success"] else "failed")
        for qname, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted XY8 data."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubits"],
        node.results.get("fit_results", {}),
        analysis_signal=node.parameters.analysis_signal,
    )
    node.results["figure"] = fig
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No state update — XY8 is a diagnostic measurement.

    Results are available in node.results["fit_results"] for inspection.
    """
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

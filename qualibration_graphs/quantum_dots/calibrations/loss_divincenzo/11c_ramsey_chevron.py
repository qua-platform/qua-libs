# %% {Imports}
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

from qm.qua import *

from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.ramsey import RamseyChevronParameters as Parameters
from calibration_utils.ramsey_chevron_parity_diff import (
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.common_utils.experiment import get_qubits, enable_dual_drive_mw
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.parity_streams import (
    declare_parity_streams,
    save_parity_measurement,
    buffer_parity_streams,
    process_parity_streams,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.sweep import get_idle_times_in_clock_cycles

# %% {Node initialisation}
description = """
        RAMSEY CHEVRON PARITY DIFFERENCE
This sequence performs a Ramsey measurement with parity difference to characterize the qubit detuning and idle time.
The measurement involves sweeping the detuning frequency of the qubit, and performing a sequence of
two π/2 rotations with a swept idle time in between to create a 2D measurement. PSB is used to measure the
parity of the resulting state.

The sequence uses voltage sequences to navigate through a triangle in voltage space (empty -
initialization - measurement) using OPX channels on the fast lines of the bias-tees. At each pulse duration,
the parity is measured before (P1) and after (P2) the qubit manipulation; joint-outcome streams are
saved and reduced to conditional readout expectations in post-processing.

The analysis signal (default: conditional second parity given first parity) reveals Ramsey oscillations as a function of pulse duration and as a function of
pulse detuning, which can be used to extract the qubit coupling strength, coherence time, and optimal pulse parameters.

Prerequisites:
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the voltage points (empty - initialization - measurement).
    - Qubit pulse calibration (X90 pulse amplitude and frequency).

State update:
    - The qubit Larmor frequency.
    - The qubit  T2* (Ramsey) time.
"""


node = QualibrationNode[Parameters, Quam](
    name="11c_ramsey_chevron",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.num_shots = 10
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    u = unit(coerce_to_integer=True)

    node.namespace["qubits"] = qubits = get_qubits(node)

    n_avg = node.parameters.num_shots
    # Idle time sweep (in clock cycles of 4ns)
    tau_values = get_idle_times_in_clock_cycles(node.parameters)
    # Detuning sweep
    detuning_values = np.arange(
        -node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_step_in_mhz * u.MHz,
    )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            detuning_values,
            attrs={"long_name": "frequency detuning", "units": "Hz"},
        ),
        "tau": xr.DataArray(
            tau_values * 4,
            attrs={"long_name": "idle time", "units": "ns"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw(node)

        t = declare(int)
        df = declare(int)
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
            intermediate_frequency = qubit.xy.intermediate_frequency

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(df, detuning_values)):
                    with for_each_(t, tau_values):

                        qubit.xy.update_frequency(intermediate_frequency + df)
                        reset_frame(qubit.xy.name)

                        align()

                        if node.parameters.parity_pre_measurement:
                            qubit.empty()
                            a1 = qubit.measure()

                        n_init = qubit.initialize()
                        if heralded_and_return_n_loops:
                            save(n_init, n_loops_st[qubit.name])

                        align()

                        with strict_timing_():
                            qubit.x90()
                            wait(t, qubit.xy.name)
                            qubit.x90()

                        align()

                        a2 = qubit.measure()

                        qubit.voltage_sequence.ramp_to_zero()

                        align()

                        assign(p2, Cast.to_int(a2))

                        if node.parameters.parity_pre_measurement:
                            assign(p1, Cast.to_int(a1))

                        save_parity_measurement(node, qubit.name, p1, p2, parity_streams)

            qubit.xy.update_frequency(intermediate_frequency)

        with stream_processing():
            n_st.save("n")

            n_detuning = len(detuning_values)
            n_tau = len(tau_values)
            for qubit in qubits:
                buffer_parity_streams(node, qubit.name, parity_streams, n_detuning, n_tau)
                if heralded_and_return_n_loops:
                    n_loops_st[qubit.name].buffer(n_detuning).buffer(n_tau).average().save(
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
    node.namespace["qubits"] = get_qubits(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome streams."""
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [q.name for q in node.namespace["qubits"]],
        parity_pre_measurement=node.parameters.parity_pre_measurement,
        sweep_dims=("detuning", "tau"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data to extract frequency offset and T2*."""
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
    """Plot the raw and fitted data."""
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
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if not node.results["fit_results"][qubit.name]["success"]:
                continue

            fit_result = node.results["fit_results"][qubit.name]
            qubit.larmor_frequency = qubit.larmor_frequency + fit_result["freq_offset"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

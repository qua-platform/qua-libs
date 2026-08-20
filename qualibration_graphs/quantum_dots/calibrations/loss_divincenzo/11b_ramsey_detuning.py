# %% {Imports}
import logging
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.ramsey_detuning_parity_diff import (
    Parameters as RamseyDetuningParameters,
    analyse_raw_data,
    log_fitted_results,
    plot_all,
)
from qualibration_libs.parameters import get_qubits
from calibration_utils.measurement_utils.measurement_streams import (
    declare_streams,
    save_measurement,
    buffer_streams,
    process_streams,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
RAMSEY DETUNING PARITY DIFFERENCE (TWO-τ)

Sweeps the drive-frequency detuning at two fixed idle times (τ_short
and τ_long) and measures pre/post parity via joint-outcome streams;
analysis uses the selected conditional expectation (default: P(second=1|first=0)).

The two traces act as a Vernier: wide fringes (short τ) localise the
resonance coarsely, narrow fringes (long τ) sharpen the estimate.  Each
trace is fitted independently with a profiled differential-evolution
search over the oscillation frequency (linear parameters solved by
least-squares).  The resonance detuning δ₀ is the amplitude-weighted
mean of the per-trace estimates.  The amplitude ratio between traces
gives the exponential decay rate γ and dephasing time T₂*.

Prerequisites:
    - Calibrated resonators and voltage points (empty - init - measure).
    - Calibrated X90 pulse amplitude and frequency.

State update:
    - qubit.xy.intermediate_frequency
"""


node = QualibrationNode[RamseyDetuningParameters, Quam](
    name="11b_ramsey_detuning",
    description=description,
    parameters=RamseyDetuningParameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Create the sweep axes and generate the QUA program.

    Sweeps drive-frequency detuning at two fixed idle times (τ_short
    and τ_long), performing a π/2 – idle – π/2 Ramsey sequence with
    parity-difference readout at each.  Produces a 2-D dataset
    (tau × detuning).
    """
    # ── Experiment parameters (Python side) ──────────────────────────────
    u = unit(coerce_to_integer=True)

    node.namespace["qubits"] = qubits = get_qubits(node)

    n_avg = node.parameters.num_shots
    # Two idle times in clock cycles (4 ns each)
    idle_times_cc = np.array(
        [
            node.parameters.idle_time_ns // 4,
            node.parameters.idle_time_long_ns // 4,
        ]
    )
    idle_times_ns = np.array(
        [
            node.parameters.idle_time_ns,
            node.parameters.idle_time_long_ns,
        ],
        dtype=float,
    )
    detuning_values = np.arange(
        -node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_step_in_mhz * u.MHz,
    )

    # Register the sweep axes to be added to the dataset when fetching data.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(
            idle_times_ns, attrs={"long_name": "idle time", "units": "ns"}
        ),
        "detuning": xr.DataArray(
            detuning_values, attrs={"long_name": "frequency detuning", "units": "Hz"}
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        t = declare(int)
        df = declare(int)
        n = declare(int)

        # Streams store the joint pre/post parity outcomes for later conversion into
        # conditional expectations such as E_p1_given_p0_0.
        p_post, p_pre, streams = declare_streams(node, qubits)

        n_st = declare_output_stream()

        for qubit in qubits:
            intermediate_frequency = qubit.xy.intermediate_frequency

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(df, detuning_values)):
                    with for_(*from_array(t, idle_times_cc)):
                        # Reset the drive frequency and frame before each Ramsey shot.
                        qubit.xy.update_frequency(intermediate_frequency + df)
                        reset_frame(qubit.xy.name)

                        align()

                        # Optional pre-measurement parity readout for conditional analysis.
                        if node.parameters.parity_measurement:
                            qubit.empty()
                            a1 = qubit.measure()

                        # Re-initialize at the operating point before applying the Ramsey pulses.
                        qubit.initialize()

                        align()

                        with strict_timing_():
                            # Apply the Ramsey π/2 – idle – π/2 sequence for this detuning/τ point.
                            qubit.x90()
                            wait(t, qubit.xy.name)
                            qubit.x90()

                        align()

                        a2 = qubit.measure()

                        # Return the slow voltages to zero before the next shot.
                        qubit.voltage_sequence.ramp_to_zero()

                        align()

                        assign(p_post, Cast.to_int(a2))

                        if node.parameters.parity_measurement:
                            assign(p_pre, Cast.to_int(a1))

                        save_measurement(node, qubit.name, p_pre, p_post, streams)

            qubit.xy.update_frequency(intermediate_frequency)

        with stream_processing():
            n_st.save("n")

            n_tau = len(idle_times_cc)
            n_detuning = len(detuning_values)
            for qubit in qubits:
                # Buffer one 2-D trace per qubit (tau × detuning) before host-side processing.
                buffer_streams(node, qubit.name, streams, n_tau, n_detuning)


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Connect to the QOP, execute the QUA program, and fetch raw joint-outcome data into ``ds_raw``."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Compute conditional expectations from joint-outcome streams."""
    node.results["ds_raw"] = process_streams(
        node.results["ds_raw"],
        [q.name for q in node.namespace["qubits"]],
        parity_measurement=node.parameters.parity_measurement,
        sweep_dims=("tau", "detuning"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Fit resonance detuning via joint two-τ differential evolution."""
    node.results["ds_fit"], fit_results, node.namespace["_fit_results_full"] = analyse_raw_data(
        node.results["ds_raw"], node
    )
    node.results["fit_results"] = fit_results
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Plot conditional readout signal vs detuning for both τ traces with joint fit."""
    fit_with_diag = node.namespace.get("_fit_results_full", node.results.get("fit_results", {}))
    node.results["figures"] = plot_all(
        node.results["ds_raw"],
        node.namespace["qubits"],
        ds_fit=node.results.get("ds_fit"),
        fit_results=fit_with_diag,
        analysis_signal=node.parameters.analysis_signal,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Correct the qubit intermediate frequency by the fitted resonance offset."""

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if node.outcomes.get(qubit.name) != "successful":
                continue

            fit_result = node.results["fit_results"][qubit.name]
            try:
                qubit.larmor_frequency = qubit.larmor_frequency + fit_result["freq_offset"]

            except ValueError as exc:
                logger.warning("%s: skipping state update — %s", qubit.name, exc)
                node.outcomes[qubit.name] = "failed"


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Persist the node results and any recorded state updates."""
    node.save()

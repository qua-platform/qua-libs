# %% {Imports}
import logging
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

logger = logging.getLogger(__name__)

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import (
    progress_counter_with_log,
)
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam
from calibration_utils.common_utils.experiment import get_qubits, enable_dual_drive_mw
from calibration_utils.common_utils.parity_streams import (
    declare_parity_streams,
    save_parity_measurement,
    buffer_parity_streams,
    process_parity_streams,
)
from qualibration_libs.core import tracked_updates
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.qubit_spectroscopy_parity_diff import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        QUBIT SPECTROSCOPY PARITY DIFFERENCE
This sequence involves parking the qubit at the manipulation bias point, and playing pulses of varying frequency
to drive the qubit. When the pulse frequency is the Larmor frequency, the qubit is driven, and the parity is measured
via PSB.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the PSB readout scheme.


State update:
    - The qubit frequency (and optionally the corresponding LO/IF plan) for the specified qubit operation.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="08b_qubit_spectroscopy",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1"]
    # node.parameters.num_shots = 1
    # node.parameters.simulate = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    operation_len = node.parameters.operation_len_in_ns

    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    operation_amp_factor = node.parameters.operation_amplitude_factor

    for qubit in qubits:
        qubit.x.update(
            amplitude_scale=operation_amp_factor,
            duration=operation_len,
        )

    u = unit(coerce_to_integer=True)
    n_avg = node.parameters.num_shots  # The number of averages

    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    pulse_scale = node.parameters.rotation_scale
    node.namespace["tracked_qubits"] = []
    for qubit in qubits:
        with tracked_updates(qubit.xy, auto_revert=False, dont_assign_to_none=True) as q_xy:
            pulse_name = f"{node.machine.pulse_family}_x180"
            operations_dict = q_xy.operations
            try:
                pulse_obj = operations_dict[pulse_name]
            except KeyError:
                pulse_obj = None

            if pulse_obj is not None:
                pulse_obj.amplitude = pulse_scale * pulse_obj.amplitude

            node.namespace["tracked_qubits"].append(q_xy)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            dfs, attrs={"long_name": "drive frequency", "units": "Hz"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw(node)

        df = declare(int)
        n = declare(int)

        p2, p1, parity_streams = declare_parity_streams(node, qubits)

        i_st = {qubit.name: declare_output_stream() for qubit in qubits}
        q_st = {qubit.name: declare_output_stream() for qubit in qubits}

        heralded_and_return_n_loops = getattr(node.parameters, "return_n_loops", False)
        if heralded_and_return_n_loops:
            n_loops_st = {qubit.name: declare_output_stream() for qubit in qubits}

        n_st = declare_output_stream()

        for qubit in qubits:
            intermediate_frequency = qubit.xy.intermediate_frequency

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    qubit.xy.update_frequency(intermediate_frequency)

                    align()

                    if node.parameters.parity_pre_measurement:
                        qubit.empty()
                        a1 = qubit.measure()
                    
                    n_init = qubit.initialize()

                    align()
                    qubit.xy.update_frequency(intermediate_frequency + df)

                    align()
                    qubit.x180()
                    align()

                    (i, q, a2) = qubit.measure(return_iq=True)

                    qubit.voltage_sequence.ramp_to_zero()

                    align()

                    if node.parameters.parity_pre_measurement:
                        assign(p1, Cast.to_int(a1))

                    assign(p2, Cast.to_int(a2))

                    save_parity_measurement(node, qubit.name, p1, p2, parity_streams)
                    save(i, i_st[qubit.name])
                    save(q, q_st[qubit.name])
                    if heralded_and_return_n_loops:
                        save(n_init, n_loops_st[qubit.name])

            qubit.xy.update_frequency(intermediate_frequency)

        with stream_processing():
            n_st.save("n")
            n_dfs = len(dfs)
            for qubit in qubits:
                buffer_parity_streams(node, qubit.name, parity_streams, n_dfs)
                i_st[qubit.name].buffer(n_dfs).average().save(f"I_{qubit.name}")
                q_st[qubit.name].buffer(n_dfs).average().save(f"Q_{qubit.name}")
                if heralded_and_return_n_loops:
                    n_loops_st[qubit.name].buffer(n_dfs).average().save(f"n_loops_{qubit.name}")

# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
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
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
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
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome parity streams."""
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [q.name for q in node.namespace["qubits"]],
        parity_pre_measurement=node.parameters.parity_pre_measurement,
        sweep_dims=("detuning",),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    qubits = node.namespace["qubits"]
    ds_raw = node.results["ds_raw"]

    fig_raw_fit = plot_raw_data_with_fit(
        ds_raw,
        qubits,
        node.results["ds_fit"],
        analysis_signal=node.parameters.analysis_signal,
    )

    # IQ vs frequency — one subplot per qubit, I and Q as separate lines
    qubit_names = [q.name for q in qubits]
    n = len(qubit_names)
    fig_iq, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for idx, qname in enumerate(qubit_names):
        ax = axes[0, idx]
        i_key = f"I_{qname.rstrip('0123456789')}"
        q_key = f"Q_{qname.rstrip('0123456789')}"
        i_vals = ds_raw[i_key].sel(qubit=qname).values
        q_vals = ds_raw[q_key].sel(qubit=qname).values
        detuning_mhz = ds_raw["detuning"].values / 1e6
        ax_q = ax.twinx()
        line_i, = ax.plot(detuning_mhz, i_vals, color="C0", label="I", zorder=3)
        line_q, = ax_q.plot(detuning_mhz, q_vals, color="C1", label="Q", zorder=2)
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("I", color="C0")
        ax_q.set_ylabel("Q", color="C1")
        ax.tick_params(axis="y", labelcolor="C0")
        ax_q.tick_params(axis="y", labelcolor="C1")
        ax.set_title(f"IQ vs frequency — {qname}")
        ax.legend(handles=[line_i, line_q])
    fig_iq.suptitle("Raw IQ signal")
    fig_iq.tight_layout()

    node.results["figures"] = {
        "qubit_spectroscopy": fig_raw_fit,
        "iq_scatter": fig_iq,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit spectroscopy parity-diff analysis was successful.

    When two peaks are found, the closest to centre updates the qubit under
    study and the second peak updates the preferred-readout qubit.
    """
    for tracked_qubit in node.namespace.get("tracked_qubits", []):
        tracked_qubit.revert_changes()
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            fit = node.results["fit_results"][q.name]
            opt_frequency = fit["frequency"]
            try:
                q.larmor_frequency = opt_frequency
                q.x.update(frequency=opt_frequency)
            except ValueError as exc:
                logger.warning("%s: skipping state update — %s", q.name, exc)
                node.outcomes[q.name] = "failed"
                continue

            readout_freq = fit.get("readout_qubit_frequency")
            if readout_freq is not None:
                try:
                    readout_dot_id = q.preferred_readout_quantum_dot
                    readout_qubit = next(
                        rq for rq in node.machine.qubits.values()
                        if rq.quantum_dot.id == readout_dot_id
                    )
                    readout_qubit.larmor_frequency = readout_freq
                    readout_qubit.x.update(frequency=readout_freq)
                    logger.info(
                        "%s: updated readout qubit %s frequency to %.3f GHz",
                        q.name, readout_qubit.name, readout_freq * 1e-9,
                    )
                except Exception as exc:
                    logger.warning(
                        "%s: could not update readout qubit — %s", q.name, exc
                    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

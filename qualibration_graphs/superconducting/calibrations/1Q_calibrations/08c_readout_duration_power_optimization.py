"""Joint readout integration-duration x amplitude optimization."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils import (
    accumulated_demod_batches,
    declare_path_arrays,
    preflight_accumulated_demod,
)
from calibration_utils.readout_duration_power_optimization import (
    DEFAULT_INTEGRATION_WEIGHTS,
    Parameters,
    declare_recombination_variables,
    fit_raw_data,
    get_amplitude_prefactors,
    get_durations_in_ns,
    get_samples_per_chunk,
    has_custom_integration_weights,
    log_fitted_results,
    plot_amplitude_cut,
    plot_duration_cut,
    plot_fidelity_map,
    process_raw_dataset,
    readout_config_override,
    save_accumulated_quadratures,
    set_integration_weights,
)
from calibration_utils.iq_blobs.plotting import plot_iq_blobs, plot_confusion_matrices, plot_histograms
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        READOUT DURATION x POWER OPTIMIZATION (2D)
The sequence measures the resonator after thermalization (qubit in |g>) and after a x180 pulse
(qubit in |e>) while sweeping the readout amplitude, and acquires every measurement with
accumulated demodulation so that each shot also yields every integration duration up to the
maximum. Assignment fidelity is mapped over the resulting (amplitude, duration) grid and the
operating point is chosen from it.

This is node 08b with a second axis. The duration axis is free: accumulated demodulation returns
the running integral of the readout signal after every chunk, so the whole duration sweep comes
out of the same shots that the amplitude sweep already paid for. Acquisition time is therefore
the same as node 08b's.

Purpose:
    - Choose the readout amplitude and the readout integration duration together, rather than
      fixing one while optimizing the other.
    - Show the fidelity saturation knee along the duration axis, which is where readout time
      can be bought back from every downstream experiment at no fidelity cost.

Operating point selection (per qubit):
    1. At every grid point a two-component spherical Gaussian mixture is fitted to the |g> and
       |e> IQ samples, giving the assignment fidelity, the non-outlier fraction, and the ratio
       between the two fitted blob variances.
    2. A point is eligible only if its non-outlier fraction is at least `outliers_threshold`
       AND its blob variance ratio is at most `max_variance_ratio`. The variance gate catches
       what the outlier gate alone misses: at high power the excited blob spreads into an arc
       while the ground blob stays tight, which keeps the non-outlier fraction healthy while
       making the fidelity number meaningless.
    3. The operating point is the global fidelity maximum over the eligible points. A qubit
       with no eligible point fails and nothing is written for it; the log names the gate.
    4. With `update_readout_length` off the readout keeps its current length, so the search is
       pinned to that duration and reduces to an amplitude sweep there. Everything written to
       the state then describes the integration duration the readout will actually run at. A
       qubit whose current readout length is not on the swept duration axis fails.

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a and 04b).
    - Proper thermalization time set (qubit.thermalization_time).

Notes:
    - The node OVERRIDES every selected qubit's readout pulse length to `max_duration_in_ns`
      while it generates its config, so the sweep can reach past the length currently in the
      state, and so qubits with different readout lengths share one duration axis. The
      override never reaches the state: the originals are restored as soon as the config has
      been generated, and the state update below writes the chosen duration explicitly.
    - Custom integration weights span the previous pulse length, so they no longer tile the
      readout pulse once its length changes. They are always reset to the default constant
      weights for the sweep itself, and reset in the state only when the length is updated.
    - Only thermal reset is supported: active reset judges the qubit through the very readout
      pulse this node rescales and lengthens, against a threshold calibrated for the old one.
    - `max_duration_in_ns / num_durations` must be a multiple of 4 ns, the chunk granularity
      of accumulated demodulation. With `update_readout_length` off, a qubit's current readout
      length must land on that same grid, since the search is pinned to it.
    - Accumulated demodulation costs 4 PPU processing blocks per measured qubit against 16 per
      MW-FEM (20 per OPX+), so a multiplexed run is split into batches of 4 qubits per MW-FEM
      (5 per OPX+); set `multiplexed=False` to measure one qubit at a time.
    - Qubits outside the batch currently being measured are silent, so the feedline carries
      fewer readout tones than a production multiplexed readout does. If readout crosstalk
      matters on your chip, the optimum found here can sit slightly off the one a full
      multiplexed readout would give.

State update:
    - The readout pulse length: qubit.resonator.operations[operation].length (if
      `update_readout_length`)
    - The readout amplitude: qubit.resonator.operations[operation].amplitude
    - The integration weights: reset to the defaults (only if `update_readout_length`)
    - The integration weight angle: qubit.resonator.operations[operation].integration_weights_angle
    - The ge discrimination threshold: qubit.resonator.operations[operation].threshold
    - The Repeat Until Success threshold: qubit.resonator.operations[operation].rus_exit_threshold
    - The confusion matrix: qubit.resonator.confusion_matrix
"""


node = QualibrationNode[Parameters, Quam](
    name="08c_readout_duration_power_optimization",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Override parameters for local debugging runs."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    if node.parameters.reset_type != "thermal":
        raise ValueError(
            f"Only 'thermal' reset is supported, got {node.parameters.reset_type!r}. Active reset would "
            f"judge the qubit through the readout pulse this node rescales and lengthens."
        )
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    operation = node.parameters.operation
    n_runs = node.parameters.num_shots
    amps = get_amplitude_prefactors(node.parameters)
    durations = get_durations_in_ns(node.parameters)
    num_durations = node.parameters.num_durations
    samples_per_chunk = get_samples_per_chunk(node.parameters)

    # Register the sweep axes to be added to the dataset when fetching data. The order must
    # match the stream buffering below: duration is the inner (fast) axis.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "readout amplitude prefactor", "units": ""}),
        "duration": xr.DataArray(durations, attrs={"long_name": "integration duration", "units": "ns"}),
    }

    # The pulse is lengthened to the sweep maximum ONLY while the program and the config are
    # built, so that every qubit shares one duration axis and the sweep can reach past the
    # lengths held in the state. `readout_config_override` restores the originals on the way
    # out, including on failure, so this can never reach `node.save()`.
    with readout_config_override(qubits, operation, node.parameters.max_duration_in_ns, node.log):
        # Everything that would make the acquisition quietly wrong fails here rather than later.
        preflight_accumulated_demod(qubits, operation, samples_per_chunk, node.parameters.reset_type)
        measurement_batches = accumulated_demod_batches(qubits, node.parameters.multiplexed)
        node.log(
            f"Accumulated demodulation at {samples_per_chunk * 4} ns chunks, "
            f"{num_durations} durations up to {node.parameters.max_duration_in_ns} ns, "
            f"in {len(measurement_batches)} measurement batch(es)."
        )

        with program() as node.namespace["qua_program"]:
            n = declare(int)
            k = declare(int)
            a = declare(fixed)
            n_st = declare_stream()
            I_g_st = [declare_stream() for _ in range(num_qubits)]
            Q_g_st = [declare_stream() for _ in range(num_qubits)]
            I_e_st = [declare_stream() for _ in range(num_qubits)]
            Q_e_st = [declare_stream() for _ in range(num_qubits)]
            # Declared once for every qubit, up front, so the batch loop below only plays and
            # measures -- the same shape as `Ig[i]` in node 08b. qm-qua hoists every `declare`
            # to program scope wherever it is written, so this is a readability choice rather
            # than a requirement, but it keeps the declarations independent of the batching.
            paths_g = [declare_path_arrays(num_durations) for _ in range(num_qubits)]
            paths_e = [declare_path_arrays(num_durations) for _ in range(num_qubits)]
            scratch = declare_recombination_variables()

            for measurement_batch in measurement_batches:
                # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
                for qubit in measurement_batch.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_runs, n + 1):
                    save(n, n_st)
                    with for_(*from_array(a, amps)):
                        # Ground state: qubit initialization
                        for i, qubit in measurement_batch.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        # Ground state: qubit readout
                        for i, qubit in measurement_batch.items():
                            qubit.resonator.measure_accumulated(
                                operation,
                                amplitude_scale=a,
                                segment_length=samples_per_chunk,
                                qua_vars=paths_g[i],
                            )
                            qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        for i in measurement_batch:
                            save_accumulated_quadratures(paths_g[i], num_durations, k, scratch, I_g_st[i], Q_g_st[i])
                        align()

                        # Excited state: qubit initialization
                        for i, qubit in measurement_batch.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        # Excited state: qubit readout
                        for i, qubit in measurement_batch.items():
                            # Play the x180 gate to put the qubit in the excited state
                            qubit.xy.play("x180")
                            # Align the elements to measure after playing the qubit pulses.
                            qubit.align()
                            qubit.resonator.measure_accumulated(
                                operation,
                                amplitude_scale=a,
                                segment_length=samples_per_chunk,
                                qua_vars=paths_e[i],
                            )
                            qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        for i in measurement_batch:
                            save_accumulated_quadratures(paths_e[i], num_durations, k, scratch, I_e_st[i], Q_e_st[i])
                        align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    I_g_st[i].buffer(num_durations).buffer(len(amps)).buffer(n_runs).save(f"Ig{i + 1}")
                    Q_g_st[i].buffer(num_durations).buffer(len(amps)).buffer(n_runs).save(f"Qg{i + 1}")
                    I_e_st[i].buffer(num_durations).buffer(len(amps)).buffer(n_runs).save(f"Ie{i + 1}")
                    Q_e_st[i].buffer(num_durations).buffer(len(amps)).buffer(n_runs).save(f"Qe{i + 1}")

        # The config must be generated while the override is still in place, and reused as-is
        # by the execute and simulate actions below.
        node.namespace["config"] = node.machine.generate_config()


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    config = node.namespace["config"]
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and store raw data in `ds_raw`."""
    qmm = node.machine.connect()
    # The config carries the lengthened readout pulse, so it is the one built above rather
    # than a freshly generated one.
    config = node.namespace["config"]
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
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    # load_from_id rebuilds node.parameters from the SAVED run, which would revert any
    # re-fit / analysis knob the user changed to re-analyse loaded data. Snapshot the
    # user's current values for those knobs (+ load_data_id) and restore them after load.
    _refit_keep = {
        k: getattr(node.parameters, k)
        for k in ("load_data_id", "outliers_threshold", "max_variance_ratio", "update_readout_length")
    }
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    for _k, _v in _refit_keep.items():
        setattr(node.parameters, _k, _v)
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse raw data and store fits in `ds_fit` and `fit_results`."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], node.results["ds_iq_blobs"], fit_results = fit_raw_data(node.results["ds_raw"], node)
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
    """Plot the fidelity map, its cuts through the operating point, and the blobs there."""
    qubits = node.namespace["qubits"]
    fig_map = plot_fidelity_map(
        node.results["ds_raw"],
        qubits,
        node.results["ds_fit"],
        node.parameters.outliers_threshold,
        node.parameters.max_variance_ratio,
    )
    fig_amp_cut = plot_amplitude_cut(node.results["ds_raw"], qubits, node.results["ds_fit"])
    fig_dur_cut = plot_duration_cut(node.results["ds_raw"], qubits, node.results["ds_fit"])
    fig_iq = plot_iq_blobs(node.results["ds_raw"], qubits, node.results["ds_iq_blobs"])
    fig_confusion = plot_confusion_matrices(node.results["ds_raw"], qubits, node.results["ds_iq_blobs"])
    fig_histogram_log = plot_histograms(node.results["ds_raw"], qubits, node.results["ds_iq_blobs"], log_scale=True)
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "fidelity_map": fig_map,
        "amplitude_cut": fig_amp_cut,
        "duration_cut": fig_dur_cut,
        "iq_blobs": fig_iq,
        "confusion_matrix": fig_confusion,
        "histograms_log": fig_histogram_log,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    operation_name = node.parameters.operation
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_results = node.results["fit_results"][q.name]
            operation = q.resonator.operations[operation_name]

            # The length goes first: every quantity below is derived AT that length, so
            # writing it afterwards would leave the thresholds describing the old pulse. When
            # the length is not updated the analysis has already pinned the search to the
            # current length, so `optimal_duration` equals `operation.length` either way and
            # the demod-unit conversions below are consistent with the pulse that will run.
            if node.parameters.update_readout_length:
                operation.length = int(fit_results["optimal_duration"])
                # Custom weights span the previous length and no longer tile the pulse.
                # Reset them to the defaults and say so in the log. This only happens when the
                # length actually changes: on an amplitude-only run the weights still fit the
                # pulse, so throwing away a previous weights calibration would be gratuitous.
                if has_custom_integration_weights(operation):
                    node.log(
                        f"{q.name}: resetting custom integration weights to the defaults, they no longer "
                        f"span the {operation.length} ns readout pulse."
                    )
                    set_integration_weights(operation, DEFAULT_INTEGRATION_WEIGHTS)

            operation.integration_weights_angle -= float(fit_results["iw_angle"])
            operation.threshold = float(fit_results["ge_threshold"]) * operation.length / 2**12
            operation.rus_exit_threshold = float(fit_results["rus_threshold"]) * operation.length / 2**12
            operation.amplitude = float(fit_results["optimal_amplitude"])
            q.resonator.confusion_matrix = fit_results["confusion_matrix"]

            # The amplitude is written as measured, never clamped: a genuinely high optimum
            # should be reported rather than silently truncated to a value the data never
            # supported.
            if operation.amplitude > node.parameters.max_readout_amplitude:
                node.log(
                    f"WARNING: {q.name}: the chosen readout amplitude "
                    f"{operation.amplitude * 1e3:.1f} mV exceeds max_readout_amplitude "
                    f"({node.parameters.max_readout_amplitude * 1e3:.1f} mV). Note that the sweep is "
                    f"relative to the CURRENT amplitude, so repeated runs compound."
                )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist node results and state updates."""
    node.save()

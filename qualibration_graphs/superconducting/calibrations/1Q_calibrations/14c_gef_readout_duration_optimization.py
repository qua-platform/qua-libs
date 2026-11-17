"""G-E-F readout duration optimization (node 14c)."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils import (
    accumulated_demod_batches,
    declare_path_arrays,
    ensure_gef_readout_pulse,
    preflight_accumulated_demod,
    reset_for_gef,
    set_gef_readout_frequency,
)
from calibration_utils.readout_duration_power_optimization import (
    DEFAULT_INTEGRATION_WEIGHTS,
    declare_recombination_variables,
    has_custom_integration_weights,
    readout_config_override,
    save_accumulated_quadratures,
    set_integration_weights,
)
from calibration_utils.readout_gef_duration_optimization import (
    Parameters,
    fit_raw_data,
    get_durations_in_ns,
    get_samples_per_chunk,
    log_fitted_results,
    plot_fidelity_vs_duration,
    process_raw_dataset,
)
from calibration_utils.iq_blobs_ef import plot_confusion_matrices, plot_iq_blobs
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        G-E-F READOUT DURATION OPTIMIZATION
The three-state version of node 08c. The resonator is measured at the G-E-F readout frequency after
preparing the qubit in |g>, in |e> (x180) and in |f> (x180 + EF_x180). Every measurement is acquired
with accumulated demodulation, so each shot also yields every integration duration up to the
maximum: the duration axis comes for free out of the same shots, with no extra pulse per duration.

At every duration the g/e/f blobs are classified as node 15 does it (Gaussian blob centers, then
the |dI| + |dQ| nearest-center rule of readout_state_gef), and the three-state assignment fidelity
is the mean of the confusion-matrix diagonal. The chosen duration is the shortest one whose
fidelity is within fidelity_tolerance of the best on the axis.

Prerequisites:
    - Having calibrated the qubit x180 and EF_x180 pulses (nodes 04b and 13).
    - Having calibrated the G-E-F readout frequency and amplitude (nodes 14a and 14b).
    - Proper thermalization time set (qubit.thermalization_time).

Notes:
    - The node OVERRIDES every selected qubit's readout_GEF length to max_duration_in_ns while it
      generates its config, and resets custom integration weights to the defaults for the sweep.
      The originals are restored as soon as the config has been generated.
    - Only thermal reset is supported: active reset would judge the qubit through the lengthened
      pulse with stale centers.
    - max_duration_in_ns / num_durations must be a multiple of 4 ns, the chunk granularity of
      accumulated demodulation. Multiplexed runs are split in batches of 4 qubits per MW-FEM
      (5 per OPX+), the processing-block budget of accumulated demodulation.

State update:
    - The G-E-F readout length: qubit.resonator.operations[operation].length
    - Custom integration weights of that operation are reset to the defaults, since they no longer
      span the new length.
    - The g/e/f centers and confusion matrix are owned by node 15: they depend on the readout length,
      so run node 15 after this node.
"""


node = QualibrationNode[Parameters, Quam](
    name="14c_gef_readout_duration_optimization",
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
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    for qubit in qubits:
        if "EF_x180" not in qubit.xy.operations:
            raise ValueError(f"{qubit.name}: calibrate EF_x180 (node 13) before running this node.")
    ensure_gef_readout_pulse(qubits, log_callable=node.log)

    operation = node.parameters.operation
    n_runs = node.parameters.num_shots
    durations = get_durations_in_ns(node.parameters)
    num_durations = node.parameters.num_durations
    samples_per_chunk = get_samples_per_chunk(node.parameters)

    # Register the sweep axes to be added to the dataset when fetching data. The order must
    # match the stream buffering below: duration is the inner (fast) axis.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}),
        "duration": xr.DataArray(durations, attrs={"long_name": "integration duration", "units": "ns"}),
    }

    # The pulse is lengthened to the sweep maximum only while the program and the config are built.
    with readout_config_override(qubits, operation, node.parameters.max_duration_in_ns, node.log):
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
            n_st = declare_stream()
            I_g_st = [declare_stream() for _ in range(num_qubits)]
            Q_g_st = [declare_stream() for _ in range(num_qubits)]
            I_e_st = [declare_stream() for _ in range(num_qubits)]
            Q_e_st = [declare_stream() for _ in range(num_qubits)]
            I_f_st = [declare_stream() for _ in range(num_qubits)]
            Q_f_st = [declare_stream() for _ in range(num_qubits)]
            paths_g = [declare_path_arrays(num_durations) for _ in range(num_qubits)]
            paths_e = [declare_path_arrays(num_durations) for _ in range(num_qubits)]
            paths_f = [declare_path_arrays(num_durations) for _ in range(num_qubits)]
            scratch = declare_recombination_variables()

            for measurement_batch in measurement_batches:
                # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
                for qubit in measurement_batch.values():
                    node.machine.initialize_qpu(target=qubit)
                align()
                for qubit in measurement_batch.values():
                    set_gef_readout_frequency(qubit)

                with for_(n, 0, n < n_runs, n + 1):
                    save(n, n_st)
                    # Ground state: qubit initialization
                    for qubit in measurement_batch.values():
                        reset_for_gef(qubit, node.parameters.reset_type, node.parameters.simulate, u)
                    align()
                    # Ground state: qubit readout
                    for i, qubit in measurement_batch.items():
                        qubit.resonator.measure_accumulated(
                            operation, segment_length=samples_per_chunk, qua_vars=paths_g[i]
                        )
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    for i in measurement_batch:
                        save_accumulated_quadratures(paths_g[i], num_durations, k, scratch, I_g_st[i], Q_g_st[i])
                    align()

                    # Excited state: qubit initialization
                    for qubit in measurement_batch.values():
                        reset_for_gef(qubit, node.parameters.reset_type, node.parameters.simulate, u)
                    align()
                    # Excited state: qubit readout
                    for i, qubit in measurement_batch.items():
                        # Play the x180 gate to put the qubit in the excited state
                        qubit.xy.play("x180")
                        # Align the elements to measure after playing the qubit pulses.
                        qubit.align()
                        qubit.resonator.measure_accumulated(
                            operation, segment_length=samples_per_chunk, qua_vars=paths_e[i]
                        )
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    for i in measurement_batch:
                        save_accumulated_quadratures(paths_e[i], num_durations, k, scratch, I_e_st[i], Q_e_st[i])
                    align()

                    # Second excited state: qubit initialization
                    for qubit in measurement_batch.values():
                        reset_for_gef(qubit, node.parameters.reset_type, node.parameters.simulate, u)
                    align()
                    # Second excited state: qubit readout
                    for i, qubit in measurement_batch.items():
                        # Play the x180 and EF_x180 gates to put the qubit in the second excited state
                        qubit.xy.play("x180")
                        update_frequency(
                            qubit.xy.name,
                            qubit.xy.intermediate_frequency - abs(qubit.anharmonicity),
                            keep_phase=True,
                        )
                        qubit.xy.play("EF_x180")
                        update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency, keep_phase=True)
                        # Align the elements to measure after playing the qubit pulses.
                        qubit.align()
                        qubit.resonator.measure_accumulated(
                            operation, segment_length=samples_per_chunk, qua_vars=paths_f[i]
                        )
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    for i in measurement_batch:
                        save_accumulated_quadratures(paths_f[i], num_durations, k, scratch, I_f_st[i], Q_f_st[i])
                    align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    I_g_st[i].buffer(num_durations).buffer(n_runs).save(f"Ig{i + 1}")
                    Q_g_st[i].buffer(num_durations).buffer(n_runs).save(f"Qg{i + 1}")
                    I_e_st[i].buffer(num_durations).buffer(n_runs).save(f"Ie{i + 1}")
                    Q_e_st[i].buffer(num_durations).buffer(n_runs).save(f"Qe{i + 1}")
                    I_f_st[i].buffer(num_durations).buffer(n_runs).save(f"If{i + 1}")
                    Q_f_st[i].buffer(num_durations).buffer(n_runs).save(f"Qf{i + 1}")

        # The config must be generated while the override is still in place, and reused as-is
        # by the execute and simulate actions below.
        node.namespace["config"] = node.machine.generate_config()


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(
        qmm, node.namespace["config"], node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and store raw data, in Volts per duration, in `ds_raw`."""
    qmm = node.machine.connect()
    # The config carries the lengthened readout pulse, so it is the one built above.
    with qm_session(qmm, node.namespace["config"], timeout=node.parameters.timeout) as qm:
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
    # Register the raw dataset, converted to Volts with one divisor per duration slice
    node.results["ds_raw"] = process_raw_dataset(dataset, node)


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset, keeping the current selection knobs for the re-analysis."""
    keep = {k: getattr(node.parameters, k) for k in ("load_data_id", "fidelity_tolerance", "min_assignment_fidelity")}
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    for key, value in keep.items():
        setattr(node.parameters, key, value)
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Classify the g/e/f shots at every duration and pick each qubit's readout duration."""
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
    """Plot the fidelity against duration, and the IQ blobs and confusion matrix at the chosen duration."""
    qubits = node.namespace["qubits"]
    figures = {
        "fidelity_vs_duration": plot_fidelity_vs_duration(
            node.results["ds_raw"], qubits, node.results["ds_fit"], node.parameters.fidelity_tolerance
        )
    }
    if node.results["ds_iq_blobs"] is not None:
        figures["iq_blobs"] = plot_iq_blobs(node.results["ds_iq_blobs"], qubits, node.results["ds_iq_blobs"])
        figures["confusion_matrix"] = plot_confusion_matrices(
            node.results["ds_iq_blobs"], qubits, node.results["ds_iq_blobs"]
        )
    plt.show()
    # Store the generated figures
    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write the chosen G-E-F readout length if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            operation = q.resonator.operations[node.parameters.operation]
            operation.length = int(node.results["fit_results"][q.name]["optimal_duration"])
            # Custom weights span the previous length and no longer tile the pulse.
            if has_custom_integration_weights(operation):
                node.log(
                    f"{q.name}: resetting custom integration weights to the defaults, they no longer "
                    f"span the {operation.length} ns readout pulse."
                )
                set_integration_weights(operation, DEFAULT_INTEGRATION_WEIGHTS)
            node.log(f"{q.name}: run node 15 to rebuild the g/e/f centers at the new readout length.")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist node results and state updates."""
    node.save()

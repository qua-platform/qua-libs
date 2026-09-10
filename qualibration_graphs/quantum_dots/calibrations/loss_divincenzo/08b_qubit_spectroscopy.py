# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam
from calibration_utils.qubit_spectroscopy import (
    Parameters,
    fit_raw_data,
    generate_simulated_dataset,
    log_fitted_results,
    plot_all,
    process_raw_dataset,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.experiment import get_qubits
from qualibration_libs.core import tracked_updates

# %% {Node initialization}
description = """
QUBIT SPECTROSCOPY

This node sweeps the qubit drive frequency around the current RF estimate and
measures the resulting response via PSB. When the drive frequency crosses
the Larmor frequency, the signal as measured via PSB develops a resonant feature
that is fitted to extract the updated qubit frequency.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the PSB readout scheme.
    - Having a reasonable initial RF frequency estimate for the selected qubits.

Datasets:
    - ``ds_raw``: untouched thresholded ``state`` and raw I/Q streams fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed spectroscopy traces plus fit outputs and derived coordinates. Used by ``plot_data``.
    - ``fit_results``: compact per-qubit calibration dict (``asdict(FitParameters)``). Used by logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][qubit]``):
    - ``success``: whether the spectroscopy fit passed the node criteria.
    - ``frequency`` [Hz]: fitted qubit Larmor frequency.
    - ``relative_freq`` [Hz]: fitted detuning relative to the current RF frequency.
    - ``fwhm`` [Hz]: linewidth of the primary fitted peak.
    - ``num_peaks``: number of Lorentzian peaks selected by model comparison.
    - ``readout_qubit_frequency`` [Hz]: optional fitted frequency for the preferred-readout qubit.

Figures (``node.results["figures"]``):
    - ``"qubit_spectroscopy"``: state-probability trace with fitted curve overlays.
    - ``"iq_scatter"``: averaged raw I and Q traces versus drive detuning.

State update:
    - Update the qubit Larmor frequency from the fitted primary peak.
    - When two peaks are resolved, also update the preferred-readout qubit frequency.
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
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 1D frequency-detuning sweep and the QUA pulse sequence."""

    u = unit(coerce_to_integer=True)

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # repetitions averaged at each detuning point

    # Build the detuning sweep
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    # Temporary drive settings applied before calling ``qubit.x180()``.
    operation_len = node.parameters.operation_len_in_ns

    # Pulse amplitude prefactor must stay within [-2, 2) for QUA fixed-point arithmetic.
    operation_amp_factor = node.parameters.operation_amplitude_factor

    # Stage temporary pulse settings that are later reverted before permanent state updates.
    node.namespace["tracked_qubits"] = []
    for qubit in qubits:
        with tracked_updates(qubit, auto_revert=False, dont_assign_to_none=True) as q:
            q.x.update(
                amplitude_scale=operation_amp_factor,
                duration=operation_len,
            )
            node.namespace["tracked_qubits"].append(q)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "drive frequency", "units": "Hz"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Real-time variables:
        # state[i] : thresholded post-manipulation measurement (0/1) for qubit i
        # n        : shot counter
        # df       : integer frequency detuning value
        n = declare(int)
        df = declare(int)

        # Streams:
        # state_st[i] : per-qubit thresholded state stream
        # i_st, q_st : averaged raw I/Q readout traces
        # n_st : shot counter for progress reporting
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_output_stream() for _ in range(num_qubits)]
        i_st = [declare_output_stream() for _ in range(num_qubits)]
        q_st = [declare_output_stream() for _ in range(num_qubits)]
        n_st = declare_output_stream()

        # Python loop over the qubits specified in the node parameters
        for i, qubit in enumerate(qubits):
            # Remember calibrated IF so we can restore it after the detuning sweep
            intermediate_frequency = qubit.xy.intermediate_frequency

            # ── OUTER LOOP: average over shots ───────────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ── INNER LOOP: sweep frequency detuning ────────────────
                with for_(*from_array(df, dfs)):

                    # Set the qubit drive frequency to the stored IF, for initialization
                    qubit.xy.update_frequency(intermediate_frequency)

                    # Perform the initialize macro
                    qubit.initialize()
                    align()

                    # Retune the XY drive to (calibrated IF + df)
                    qubit.xy.update_frequency(intermediate_frequency + df)

                    align()
                    # Play the x180 gate
                    qubit.x180()
                    align()

                    # Thresholded PSB readout → averaged state probability
                    # Also capture the unthresholded I and Q components.
                    (i_comp, q_comp, s) = qubit.measure(return_iq=True)
                    assign(state[i], Cast.to_int(s))
                    save(state[i], state_st[i])
                    save(i_comp, i_st[i])
                    save(q_comp, q_st[i])

                    align()
                    # Return gate voltages to zero before the next point.
                    qubit.voltage_sequence.ramp_to_zero()

            # Restore the qubit's calibrated drive frequency after the sweep
            qubit.xy.update_frequency(intermediate_frequency)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits): 
                # Each save() is one frequency point.
                # .buffer(len(dfs)) : group points along the frequency axis
                # .average()      : average over all shots (n_avg repetitions)
                # Result: 1D state/I/Q-probability trace vs detuning per qubit.
                state_st[i].buffer(len(dfs)).average().save(f"state{i + 1}")
                i_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        # "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.use_simulated_data
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
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated spectroscopy data so the full analysis pipeline can run without hardware."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit the spectroscopy response and store both the fitted dataset and fit summary."""
    ds_processed = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
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
    """Plot processed data and fit overlays; store figures in ``node.results["figures"]``."""
    node.results["figures"] = plot_all(
        node.results["ds_fit"],
        node.namespace["qubits"],
        node.results["fit_results"],
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.use_simulated_data)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit spectroscopy analysis was successful.

    When two peaks are found, the closest to centre updates the qubit under
    study and the second peak updates the preferred-readout qubit.
    """
    # Revert the qubit's pulse changes first
    for q in node.namespace.get("tracked_qubits", []):
        q.revert_changes()

    # Update the state
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            fit = node.results["fit_results"][q.name]
            opt_frequency = fit["frequency"]
            try:
                q.larmor_frequency = opt_frequency
                q.x.update(frequency=opt_frequency)

                q.x.update(
                    amplitude_scale=node.parameters.operation_amplitude_factor,
                    duration=node.parameters.operation_len_in_ns,
                )
            except ValueError as exc:
                node.log(f"{q.name}: skipping state update - {exc}")
                node.outcomes[q.name] = "failed"
                continue

            readout_freq = fit.get("readout_qubit_frequency")
            if readout_freq is not None:
                try:
                    readout_dot_id = q.preferred_readout_quantum_dot
                    readout_qubit = next(
                        rq for rq in node.machine.qubits.values() if rq.quantum_dot.id == readout_dot_id
                    )
                    readout_qubit.larmor_frequency = readout_freq
                    readout_qubit.x.update(frequency=readout_freq)
                    node.log(
                        f"{q.name}: updated readout qubit "
                        f"{readout_qubit.name} frequency to {readout_freq * 1e-9:.3f} GHz"
                    )

                except Exception as exc:
                    node.log(f"{q.name}: could not update readout qubit - {exc}")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the node results and any recorded state updates."""
    node.save()

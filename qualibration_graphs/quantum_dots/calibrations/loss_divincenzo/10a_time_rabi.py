# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam

from calibration_utils.time_rabi import (
    Parameters,
    fit_raw_data,
    generate_simulated_dataset,
    log_fitted_results,
    plot_all,
    process_raw_dataset,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot

# %% {Node initialisation}
description = """
        TIME RABI
After initialization to the target spin state, this sequence applies an XY drive pulse whose duration is
swept and measures the spin state with thresholded PSB readout. Averaged state probabilities versus pulse duration
are fitted to extract the π-pulse duration.

Prerequisites:
    - Having calibrated the resonators coupled to the sensor dots.
    - Having calibrated the voltage points (empty, initialization, measurement).
    - Having a rough qubit XY drive calibration (frequency and amplitude).

Datasets:
    - ``ds_raw``: untouched ``state`` stream fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed sweeps plus analysis outputs (fitted traces). Used by ``plot_data``.
    - ``fit_results``: compact per-qubit calibration dict (``FitParameters`` serialized with ``asdict``). Used by
      logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][qubit]``):
    - ``success``: whether the fit passed sanity checks and the state update is applied.
    - ``optimal_duration`` [ns]: π-pulse duration extracted from the Rabi oscillation.
    - ``rabi_frequency`` [rad / ns]: fitted Rabi frequency in the time domain.
    - ``decay_rate`` [1 / ns]: fitted decay rate of the Rabi envelope (γ ≈ 1/T₂*).

Figures (``node.results["figures"]``):
    - ``"rabi"``: state vs pulse duration with damped-sinusoid fit overlay.
    - ``"fft"``: FFT magnitude spectrum with peak fit per qubit.

State update:
    - The pulse duration of the selected operation (``node.parameters.operation``).
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="10a_time_rabi",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    # node.parameters.use_simulated_data = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 1D pulse-duration sweep and the QUA pulse sequence."""
    # ── Experiment parameters (Python side) ──────────────────────────────

    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # repetitions averaged at each duration point
    operation = node.parameters.operation  # qubit gate whose duration is swept (x180 or x90)

    # Duration axis [ns]: sweep around the expected π-pulse length (quantised to 4 ns on the OPX)
    pulse_durations = np.arange(
        node.parameters.min_wait_time_in_ns,
        node.parameters.max_wait_time_in_ns,
        node.parameters.time_step_in_ns,
    )

    # Metadata for data fetching: labels the saved state arrays when results come back from the OPX
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "pulse_duration": xr.DataArray(pulse_durations, attrs={"long_name": "qubit pulse duration", "units": "ns"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Real-time variables:
        # t         : current manipulation pulse duration [ns]
        # n         : shot counter
        # state[i]  : thresholded post-manipulation measurement (0/1) for qubit i
        t = declare(int)
        n = declare(int)
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        n_st = declare_output_stream()

        for i, qubit in enumerate(qubits):

            # ── OUTER LOOP: average over shots ───────────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ── INNER LOOP: sweep pulse duration in  clock cycles ─────────────────────
                with for_(*from_array(t, pulse_durations // 4)):

                    # Perform the initialize macro
                    qubit.initialize()
                    align()

                    # Play the selected gate at the current duration (time-Rabi)
                    qubit.macros[operation].apply(duration=t)
                    align()

                    # Thresholded PSB readout → averaged state probability
                    assign(state[i], Cast.to_int(qubit.measure()))
                    save(state[i], state_st[i])

                    # Return gate voltages to zero before the next shot to avoid accumulation of fixed point errors
                    align()
                    qubit.voltage_sequence.ramp_to_zero()

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Each save() is one duration point.
                # .buffer(len(pulse_durations)) : group points along the duration axis
                # .average()                    : average over all shots (n_avg repetitions)
                # Result: 1D state vs pulse_duration per qubit
                state_st[i].buffer(len(pulse_durations)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
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
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated time-Rabi data so the full analysis pipeline can run without hardware."""
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
    """Process state streams, fit time-Rabi data, and store results."""
    ds_processed = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {qname: ("successful" if r["success"] else "failed") for qname, r in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot processed data and fit overlays; store figures in ``node.results["figures"]``."""
    node.results["figures"] = plot_all(
        node.results["ds_fit"],
        node.namespace["qubits"],
        node.results.get("fit_results", {}),
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if node.outcomes[qubit.name] == "failed":
                continue

            fit_result = node.results["fit_results"][qubit.name]
            qubit.macros[node.parameters.operation].update(duration=fit_result["optimal_duration"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

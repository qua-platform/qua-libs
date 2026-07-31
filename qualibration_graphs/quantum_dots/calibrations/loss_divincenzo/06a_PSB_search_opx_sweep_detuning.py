# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from quam_builder.architecture.quantum_dots.operations.names import VoltagePointName
from qualibrate.core import QualibrationNode
from qualibration_libs.parameters.experiment import get_qubit_pairs
from quam_config import QubitQuam as Quam
from calibration_utils.psb_search_sweep_detuning import (
    Parameters,
    generate_simulated_dataset,
    process_raw_dataset,
    fit_raw_data_pca_gaussian,
    log_fitted_results,
    plot_all,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Node initialisation}
description = """
PAULI SPIN BLOCKADE SEARCH — Sweep detuning (OPX)

This node searches for the Pauli Spin Blockade (PSB) region by sweeping the
inter-dot detuning and measuring the sensor response during a PSB readout window.
Each sweep point plays a voltage sequence (prepare → ramp → measure) using OPX 
fast-line channels, and acquires per-shot I/Q data.

Prerequisites
-------------
- QUAM configured and loaded (``quam_config/populate_quam_state_*.py``).
- Sensor-dot readout calibrated (resonator calibration nodes completed).
- Detuning axis and voltage points (empty / initialize / measure) defined on the dot pair.

Datasets
--------
- ``ds_raw``: shot-level ``I`` and ``Q`` vs ``detuning`` (dims: ``qubit_pair``, ``n_runs``, ``detuning``).
- ``ds_fit``: readout metrics and optimum detuning per pair (from ``iq_sweep`` analysis).
- ``fit_results``: per-pair scalar results (serialized dataclass) for logging and state updates.

Results
-------
For each qubit pair, the node identifies an optimal detuning (by fidelity or visibility)
and extracts the readout axis and threshold used for the PSB readout discrimination.

Figures
-------
- Fidelity and visibility vs detuning
- Sweep summary (fidelity + visibility on twin axes)
- Shot histograms vs detuning (projected readout axis)
- Rotated IQ density at the optimal detuning with the chosen threshold

State update
------------
Updates the dot pair ``MEASURE`` voltage point detuning and stores the readout threshold
for the selected optimal detuning (only for successful pairs).
"""


node = QualibrationNode[Parameters, Quam](
    name="06a_PSB_search_opx_sweep_detuning",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.num_shots = 5000
    node.parameters.use_simulated_data = True
    node.parameters.plot_kde = False
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the sweep axes and the QUA pulse sequence."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Select which qubit-pairs participate in this calibration
    qubit_pairs = get_qubit_pairs(node)
    node.namespace["qubit_pairs"] = qubit_pairs
    num_qubit_pairs = len(qubit_pairs)

    # Number of shots per detuning point
    n_avg = node.parameters.num_shots

    # Build the detuning sweep
    detuning_min = node.parameters.detuning_min
    detuning_max = node.parameters.detuning_max
    detuning_points = node.parameters.detuning_points
    detuning_array = np.linspace(detuning_min, detuning_max, detuning_points)

    # The swept axes. Buffer order is (detuning) then (n_runs).
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "detuning": xr.DataArray(detuning_array, attrs={"long_name": "voltage", "units": "V"}),
        "n_runs": xr.DataArray(np.arange(n_avg), attrs={"long_name": "shot"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Allocate real-time variables on the OPX:
        #   I[i], Q[i]   : demodulated quadratures for qubit_pair i
        #   I_st[i], Q_st[i] : buffers collecting I/Q before transfer to PC
        #   n            : shot counter
        #   n_st         : stream reporting shot index to PC (progress bar)
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)

        # Real-time variable holding the detuning value
        detuning = declare(fixed)

        # ── OUTER LOOP: repeat the full sweep n_avg times ──
        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st) # tell the PC which shot we are on

            # Perform them all sequentially for now. Can add footprint batching later
            for i, qubit_pair in enumerate(qubit_pairs):

                # Extract the underlying quantum dot pair
                dot_pair = qubit_pair.quantum_dot_pair
                op_name = "readout" + f"_{dot_pair.name}"

                # Use the first sensor dot in the list of sensors associated with the quantum dot pair
                sensor = dot_pair.sensor_dots[0]

                # ── INNER LOOP: sweep sensor plunger gate voltage ──────────
                with for_(*from_array(detuning, detuning_array)):

                    # ── STEP 1 - INITIALIZE: Perform the initialization sequence ──────────

                    # Requires the chosen macro on dot_pair.macros (empty or initialize)
                    if node.parameters.qubit_pair_to_initialize is not None: 
                        initialization_qubit_pair = node.machine.qubit_pairs[node.parameters.qubit_pair_to_initialize]
                        dp = initialization_qubit_pair.quantum_dot_pair
                        dp.macros[node.parameters.initialization_macro].apply()
                    else: 
                        dot_pair.macros[node.parameters.initialization_macro].apply()
                    
                    if node.parameters.qubit_to_pulse is not None: 
                        q = node.machine.qubits[node.parameters.qubit_to_pulse]
                        q.x180()


                    # ── STEP 2 - RAMP: Ramp to the correct detuning point ──────────

                    # First ramp to the fixed detuning point
                    dot_pair.ramp_to_voltages(
                        {dot_pair.name: detuning, dot_pair.barrier_gate.name : node.parameters.barrier_gate_voltage},
                        ramp_duration=node.parameters.ramp_duration,
                        duration=node.parameters.buffer_duration,
                    )
                    

                    # ── STEP 3 - MEASURE: Perform the measurement at the PSB point ──────────

                    rr = sensor.readout_resonator
                    readout_length = rr.operations[op_name].length

                    # Make sure to track the duration of the readout pulse. This is to ensure that the compensation pulse calculation is correct
                    dot_pair.voltage_sequence.track_sticky_duration(readout_length)

                    align(rr.id, dot_pair.physical_channel.id) # Make sure to align the measure command to be AFTER the ramp + wait

                    # Play the "readout_{quantum_dot_pair.name}" pulse and integrate I/Q into I[i], Q[i]
                    rr.measure(op_name, qua_vars=(I[i], Q[i]))

                    # Append this voltage point's I/Q to the stream buffer
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    align()

                    # Apply the compensation pulse via the voltage sequence. This both steps to 0 before, and goes back to 0 after
                    dot_pair.voltage_sequence.apply_compensation_pulse(go_to_zero = True, return_to_zero = True)
                    
                    align()

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n") # expose shot counter as "n" in the fetched dataset
            for i, qubit_pair in enumerate(qubit_pairs):
                # Each save() above is one voltage point. 
                # .buffer(len(detuning_array)) : group points along the detuning axis
                # .buffer(n_avg) : group points along the repetitions axis
                # Result : 2D trace I(detuning, n_avg), Q(detuning, n_avg) per qubit pair
                I_st[i].buffer(len(detuning_array)).buffer(n_avg).save(
                    f"I_{qubit_pair.name}"
                )
                Q_st[i].buffer(len(detuning_array)).buffer(n_avg).save(
                    f"Q_{qubit_pair.name}"
                )


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
        "samples": samples,
    }


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Build synthetic shot-by-shot I/Q (Barthel forward model) for offline analysis."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated PSB detuning dataset generated successfully.")


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
        job.wait_until("Done")     
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Reshape the per-pair streams into a qubit_pair-indexed ds with I/Q variables.
    pair_names = [pair.name for pair in node.namespace["qubit_pairs"]]
    I_arr = xr.concat([dataset[f"I_{p}"] for p in pair_names], dim="qubit_pair")
    Q_arr = xr.concat([dataset[f"Q_{p}"] for p in pair_names], dim="qubit_pair")
    I_arr = I_arr.assign_coords(qubit_pair=pair_names)
    Q_arr = Q_arr.assign_coords(qubit_pair=pair_names)
    node.results["ds_raw"] = xr.Dataset({"I": I_arr, "Q": Q_arr})


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw dataset into a plotting/analysis-ready dataset (keeps ds_raw immutable)."""
    node.results["ds_processed"] = process_raw_dataset(node.results["ds_raw"], node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""

    node.results["ds_fit"], fit_results = fit_raw_data_pca_gaussian(
        node.results["ds_processed"], node
    )
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
    """Plot all node figures via the shared plotting API."""
    # s and alpha are relevant kwargs for plotting a scatter plot. 
    # Hard coded here as 4 and 0.15, since they should not be exposed as node parameters. 
    node.results["figures"] = plot_all(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        node.results["ds_fit"],
        sweep_name=node.parameters.sweep_name,
        fit_results=node.results["fit_results"],
        plot_kde = node.parameters.plot_kde,
        s = 4, 
        alpha = 0.15
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor data analysis was successful."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            fit_result = node.results["fit_results"][qp.name]
            if not fit_result["success"]:
                continue

            dot_pair = qp.quantum_dot_pair
            dot_pair.add_point(
                VoltagePointName.MEASURE,
                voltages={dot_pair.name: float(fit_result["optimal_sweep_value"]), dot_pair.barrier_gate.name: node.parameters.barrier_gate_voltage},
                duration=node.parameters.buffer_duration,
                replace_existing_point=True,
            )

            # Current PSB readout assumes the first sensor dot defines the pair readout.
            sensor_dot = dot_pair.sensor_dots[0]

            # SensorDot.measure("readout") already returns IQ demodulated using
            # the operation's existing integration_weights_angle from prior
            # readout calibration nodes (e.g. 05c), so 06a only updates the
            # readout threshold for the selected detuning point.
            pair_ids = {
                getattr(dot_pair, "id", None),
                getattr(dot_pair, "name", None),
            } - {None, ""}
            for pair_id in pair_ids:
                sensor_dot._add_readout_params(pair_id, threshold=float(fit_result["I_threshold"]))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

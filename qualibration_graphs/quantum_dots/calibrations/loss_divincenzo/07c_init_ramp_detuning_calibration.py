# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from qualibration_libs.parameters.experiment import get_qubit_pairs
from quam_config import QubitQuam as Quam

from calibration_utils.init_ramp_detuning import (
    Parameters,
    analyse_init_ramp_detuning,
    log_fitted_results,
    plot_all,
    process_raw_dataset,
    generate_simulated_dataset,
)
from calibration_utils.init_ramp_rate.helper_utils import validate_and_build_ramp_sweep

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialization}
description = """
        INITIALIZATION 2D CALIBRATION (RAMP DURATION × DETUNING VOLTAGE)
This sequence extends the ramp-duration calibration by adding a second sweep axis: the detuning
voltage of the initialize voltage point.

For each (ramp_duration, detuning) point the sequence sets the initialize voltage point to
the given detuning, initializes with the given ramp duration, then performs a state
measurement using the balanced measurement macro.  The boolean state assignment (0 or 1) is
averaged over many shots to produce a 2D map of mean state occupation.

The analysis identifies the (ramp_duration, detuning) pair that yields the minimum
(or maximum, controlled by the ``find_minimum`` parameter) average state assignment.

Prerequisites:
    - Having initialized the Quam.
    - Having calibrated the PSB measurement point (06a-06c).
    - Having the balanced measurement macro configured with a valid threshold.

Datasets:
    - ``ds_raw``: 2D arrays averaged on the OPX (never modified after acquisition).
      Contains averaged state/I/Q traces indexed by ``(ramp_duration, detuning)``.
    - ``ds_fit``: analysis-ready dataset with normalized per-qubit-pair variables and summary outputs.
      Used by ``plot_data``.
    - ``fit_results``: compact per-qubit-pair dict with the optimal coordinates.

Results (``node.results["fit_results"][<qubit_pair>]``):
    - ``success``: whether the analysis succeeded and the state update is applied.
    - ``optimal_ramp_duration`` [ns]: selected ramp duration from the 2D state map.
    - ``optimal_detuning`` [V]: selected detuning from the 2D state map.
    - ``optimal_avg_state``: averaged state value at the selected operating point.
    - ``find_minimum``: whether the optimum was chosen by minimization or maximization.
    - ``failure_reason``: populated when the analysis fails.

Figures (``node.results["figures"]``):
    - ``"summary_2d"``: 6-panel summary (state, I, Q heatmaps and their FFTs along detuning axis).

State update:
    - The initialize macro ``ramp_duration`` and ``point`` on each qubit pair.
"""

node = QualibrationNode[Parameters, Quam](
    name="07c_init_ramp_detuning_calibration",
    description=description,
    parameters=Parameters(),
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
    """Build the 2D sweep axes and the QUA pulse sequence (ramp duration × detuning)."""

    # ── Experiment parameters (Python side) ──────────────────────────────
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Sweep axis 1: ramp duration (ns)
    ramp_duration_array = validate_and_build_ramp_sweep(node)

    # Sweep axis 2: detuning voltage applied at the initialize point
    detuning_min = float(node.parameters.detuning_min)
    detuning_max = float(node.parameters.detuning_max)
    detuning_step = float(node.parameters.detuning_step)

    detuning_array = np.arange(detuning_min, detuning_max, detuning_step)

    # Metadata for data fetching: labels the saved arrays when results come back from the OPX
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([qp.name for qp in qubit_pairs]),
        "ramp_duration": xr.DataArray(
            ramp_duration_array,
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
        "detuning": xr.DataArray(
            detuning_array,
            attrs={"long_name": "detuning", "units": "V"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        # Allocate real-time variables on the OPX:
        # ramp_dur      : current ramp duration for the initialize macro
        # det           : current detuning value to ramp to during the initialize macro
        # n             : shot counter
        # state[j]      : thresholded post-initialization measurement (0/1) for qubit pair j
        # n_st          : stream reporting shot index to PC (progress bar)
        # i_st[j], q_st[j] : buffers collecting I/Q before transfer to PC
        n = declare(int)
        n_st = declare_output_stream()
        state = [declare(int) for _ in qubit_pairs]
        state_st = [declare_output_stream() for _ in qubit_pairs]
        i_st = [declare_output_stream() for _ in qubit_pairs]
        q_st = [declare_output_stream() for _ in qubit_pairs]

        ramp_dur = declare(int)
        det = declare(fixed)

        # ── For each qubit-pair, sweep ramp duration and measure ─────
        for j, qubit_pair in enumerate(qubit_pairs):
            dot_pair = qubit_pair.quantum_dot_pair

            # ── OUTER LOOP: repeat the full 2D sweep num_shots times ─────────
            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)  # tell the PC which shot we are on

                # ── INNER LOOPS: sweep ramp duration and detuning ────────
                with for_(*from_array(ramp_dur, ramp_duration_array)):
                    with for_(*from_array(det, detuning_array)):

                        # Initialize with the requested ramp duration at a particular detuning point
                        dot_pair.initialize(ramp_duration=ramp_dur, point={dot_pair.name: det})
                        align()

                        # Thresholded PSB readout → averaged state probability
                        (i, q, s) = dot_pair.measure(return_iq=True)
                        assign(state[j], Cast.to_int(s))
                        save(state[j], state_st[j])
                        save(i, i_st[j])
                        save(q, q_st[j])

                        # Return gate voltages to zero before the next shot to avoid accumulation of fixed point errors
                        align()
                        dot_pair.voltage_sequence.ramp_to_zero()

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")  # expose shot counter as "n" in the fetched dataset
            for j in range(num_qubit_pairs):
                # Each save() above is one (ramp_duration, detuning) point.
                # .buffer(len(detuning_array))       : group points along detuning axis (innermost loop)
                # .buffer(len(ramp_duration_array))  : group points along ramp axis (outer loop)
                # .average()                         : average over all shots on the OPX
                state_st[j].buffer(len(detuning_array)).buffer(len(ramp_duration_array)).average().save(f"state{j + 1}")
                i_st[j].buffer(len(detuning_array)).buffer(len(ramp_duration_array)).average().save(f"I{j + 1}")
                q_st[j].buffer(len(detuning_array)).buffer(len(ramp_duration_array)).average().save(f"Q{j + 1}")


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
    """Connect to the QOP, execute the QUA program and fetch the raw data."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused by the fetcher
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar while streaming data back
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated data so the full analysis pipeline can run without hardware."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Find the optimal (ramp_duration, detuning) from the 2D state map."""
    qubit_pairs = node.namespace["qubit_pairs"]

    qp_names = [qp.name for qp in qubit_pairs]
    ds_fit, fit_results = analyse_init_ramp_detuning(
        process_raw_dataset(node.results["ds_raw"].copy(deep=True), node),
        qp_names,
        find_minimum=node.parameters.find_minimum,
    )
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)

    node.outcomes = {
        qp_name: ("successful" if r["success"] else "failed") for qp_name, r in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot 2D heatmaps of state and IQ signal vs ramp duration and detuning."""
    qubit_pairs = node.namespace["qubit_pairs"]
    qp_names = [qp.name for qp in qubit_pairs]

    node.results["figures"] = plot_all(
        node.results.get("ds_fit", node.results["ds_raw"]),
        qp_names,
        fit_results=node.results.get("fit_results"),
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.use_simulated_data)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the initialize macro ramp_duration and initialize-point detuning on each qubit pair."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes.get(qp.name) != "successful":
                continue

            dot_pair = qp.quantum_dot_pair
            optimal_ramp = node.results["fit_results"][qp.name]["optimal_ramp_duration"]
            optimal_detuning = node.results["fit_results"][qp.name]["optimal_detuning"]

            point_name = dot_pair._create_point_name("initialize")
            point = dot_pair.voltage_sequence.gate_set.get_macros()[point_name]
            point.voltages[dot_pair.name] = float(optimal_detuning)

            init_macro = dot_pair.macros.get("initialize")
            if init_macro is not None and hasattr(init_macro, "update"):
                init_macro.update(ramp_duration=optimal_ramp)
            else:
                node.log(f"  {qp.name}: no updatable initialize macro found on " f"{dot_pair.name}")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

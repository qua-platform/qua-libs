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

from calibration_utils.init_2d import (
    Parameters,
    analyse_init_2d,
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
        INITIALIZATION 2D CALIBRATION (RAMP DURATION × WAIT DURATION)
This sequence extends the ramp-duration calibration by adding a second sweep axis: the wait
duration between the initialization ramp and the state measurement.

For each (ramp_duration, wait_duration) point the sequence initializes
with the given ramp duration, waits for the specified duration, then performs a state
measurement using the measurement macro.  The boolean state assignment (0 or 1) is
averaged over many shots to produce a 2D map of mean state occupation.

The analysis identifies the (ramp_duration, wait_duration) pair that yields the minimum
(or maximum, controlled by the ``find_minimum`` parameter) average state assignment.

Prerequisites:
    - Having initialized the Quam.
    - Having calibrated the PSB measurement point (06a-06c).
    - Having the measurement macro configured with a valid threshold.

Datasets:
    - ``ds_raw``: 2D arrays averaged on the OPX (never modified after acquisition).
      Contains averaged state/I/Q traces indexed by ``(ramp_duration, wait_duration)``.
    - ``ds_fit``: analysis-ready dataset with normalized per-qubit-pair variables and summary outputs.
      Used by ``plot_data``.
    - ``fit_results``: compact per-qubit-pair dict with the optimal coordinates.

Results (``node.results["fit_results"][<qubit_pair>]``):
    - ``success``: whether the analysis succeeded and the state update is applied.
    - ``optimal_ramp_duration`` [ns]: selected ramp duration from the 2D state map.
    - ``optimal_wait_duration`` [ns]: selected wait duration from the 2D state map.
    - ``optimal_avg_state``: averaged state value at the selected operating point.
    - ``find_minimum``: whether the optimum was chosen by minimization or maximization.
    - ``failure_reason``: populated when the analysis fails.

Figures (``node.results["figures"]``):
    - ``"summary_2d"``: 6-panel summary (state + I/Q heatmaps and their FFTs along wait axis).

State update:
    - The initialize macro ``ramp_duration`` on each qubit pair.
"""

node = QualibrationNode[Parameters, Quam](
    name="07b_init_2d_calibration",
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
    """Build the 2D sweep axes and the QUA pulse sequence (ramp duration × wait duration)."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Which qubit pairs to perform this experiment with
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Sweep axis 1: ramp duration (ns)
    ramp_duration_array = validate_and_build_ramp_sweep(node)

    # Sweep axis 2: wait duration between init and measurement (ns)
    wait_min = int(node.parameters.wait_duration_min)
    wait_max = int(node.parameters.wait_duration_max)
    wait_step = int(node.parameters.wait_duration_step)

    # An OPX clock cycle is 4ns. Therefore, all wait durations must be divisible by 4
    if wait_min % 4 != 0 or wait_max % 4 != 0 or wait_step % 4 != 0:
        raise ValueError(
            f"Wait settings must be divisible by 4. " f"Got min={wait_min}, max={wait_max}, step={wait_step}"
        )
    if wait_min < 16:
        raise ValueError(f"Minimum wait duration must be >= 16 ns (4 clock cycles). Got {wait_min}")

    wait_ns_array = np.arange(wait_min, wait_max, wait_step, dtype=int)

    # Convert wait times from ns to clock-cycles for the QUA `wait()` instruction
    wait_cc_array = (wait_ns_array // 4).astype(int)

    # Metadata for data fetching: labels the saved arrays when results come back from the OPX
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([qp.name for qp in qubit_pairs]),
        "ramp_duration": xr.DataArray(
            ramp_duration_array,
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
        "wait_duration": xr.DataArray(
            wait_ns_array,
            attrs={"long_name": "wait duration", "units": "ns"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Allocate real-time variables on the OPX:
        # ramp_dur      : current ramp duration for the initialize macro
        # wait_dur      : current wait duration after the initialize macro
        # n             : shot counter
        # state[j]      : thresholded post-initialization measurement (0/1) for qubit pair j
        # n_st          : stream reporting shot index to PC (progress bar)
        # I_st[j], Q_st[j] : buffers collecting I/Q before transfer to PC
        state = [declare(int) for _ in qubit_pairs]
        state_st = [declare_output_stream() for _ in qubit_pairs]
        _, I_st, _, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)

        ramp_dur = declare(int)
        wait_dur = declare(int)

        # ── For each qubit-pair, sweep ramp duration and measure ─────
        for j, qubit_pair in enumerate(qubit_pairs):
            dot_pair = qubit_pair.quantum_dot_pair

            # ── OUTER LOOP: repeat the full 2D sweep num_shots times ─────────
            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)  # tell the PC which shot we are on

                # ── INNER LOOPS: sweep ramp duration and wait duration ───
                with for_(*from_array(ramp_dur, ramp_duration_array)):
                    with for_(*from_array(wait_dur, wait_cc_array)):

                        # Initialize with the requested ramp duration
                        dot_pair.initialize(ramp_duration=ramp_dur)
                        align()

                        # Wait between init and measurement
                        wait(wait_dur)
                        align()

                        # Thresholded PSB readout → averaged state probability
                        (i, q, s) = dot_pair.measure(return_iq=True)
                        assign(state[j], Cast.to_int(s))
                        save(state[j], state_st[j])
                        save(i, I_st[j])
                        save(q, Q_st[j])

                        # Return gate voltages to zero before the next shot to avoid accumulation of fixed point errors
                        align()
                        dot_pair.voltage_sequence.ramp_to_zero()

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")  # expose shot counter as "n" in the fetched dataset
            for j in range(num_qubit_pairs):
                # Each save() above is one (ramp_duration, wait_duration) point.
                # .buffer(len(wait_cc_array))        : group points along wait axis (innermost loop)
                # .buffer(len(ramp_duration_array))  : group points along ramp axis (outer loop)
                # .average()                         : average over all shots on the OPX
                state_st[j].buffer(len(wait_cc_array)).buffer(len(ramp_duration_array)).average().save(f"state{j + 1}")
                I_st[j].buffer(len(wait_cc_array)).buffer(len(ramp_duration_array)).average().save(f"I{j + 1}")
                Q_st[j].buffer(len(wait_cc_array)).buffer(len(ramp_duration_array)).average().save(f"Q{j + 1}")


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
    """Find the optimal (ramp_duration, wait_duration) from the 2D state map."""
    qubit_pairs = node.namespace["qubit_pairs"]

    qp_names = [qp.name for qp in qubit_pairs]
    ds_in = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
    ds_fit, fit_results = analyse_init_2d(
        ds_in,
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
    """Plot 2D heatmaps of state and IQ signal vs ramp duration and wait duration."""
    qubit_pairs = node.namespace["qubit_pairs"]

    node.results["figures"] = plot_all(
        node.results.get("ds_fit", node.results["ds_raw"]),
        qubit_pairs,
        fit_results=node.results.get("fit_results"),
        plot_fft=node.parameters.plot_fft,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.use_simulated_data)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the initialize macro ramp_duration on each qubit pair."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes.get(qp.name) != "successful":
                continue

            dot_pair = qp.quantum_dot_pair
            optimal_ramp = node.results["fit_results"][qp.name]["optimal_ramp_duration"]

            init_macro = dot_pair.macros.get("initialize")
            if init_macro is not None and hasattr(init_macro, "update"):
                init_macro.update(ramp_duration=optimal_ramp)
            else:
                node.log(f"  {qp.name}: no updatable initialize macro found on " f"{dot_pair.name}")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the node results and any recorded state updates."""
    node.save()

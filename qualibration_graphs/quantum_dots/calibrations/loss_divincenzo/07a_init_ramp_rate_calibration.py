# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibration_libs.parameters.experiment import get_qubit_pairs
from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam
from calibration_utils.init_ramp_rate import (
    Parameters,
    analyse_ramp_rate,
    log_fitted_results,
    plot_all,
    generate_simulated_dataset,
)

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Node initialisation}
description = """
        INITIALISATION RAMP RATE CALIBRATION
This sequence calibrates the ramp duration of the initialisation macro by sweeping the ramp rate
and measuring the consistency of initialising into either state. 

For each ramp duration the sequence optionally empties the dots, initializes with the given ramp duration,
then performs a state measurement using the balanced measurement macro.  The boolean state
assignment (0 or 1) is averaged over many shots to produce the mean state occupation for each
ramp duration.

The analysis identifies the ramp duration that yields the minimum (or maximum, controlled by the
``find_minimum`` parameter) average state assignment, corresponding to the purest initialisation.

Prerequisites:
    - Having initialized the Quam.
    - Having calibrated the PSB measurement point (06a-06c).
    - Having the balanced measurement macro configured with a valid threshold.

Datasets:
    - ``ds_raw``: untouched data fetched from the OPX (or generated synthetically when
      ``use_simulated_data=True``). Contains per-qubit-pair variables:
      ``state_<pair>``, ``I_<pair>``, ``Q_<pair>`` indexed by ``ramp_duration`` (averaged on the OPX).
    - ``fit_results``: compact per-qubit-pair summary dict produced by ``analyse_ramp_rate``.

Figures (``node.results["figures"]``):
    - ``"avg_state_vs_ramp_duration"``: average state vs ramp duration with optimum marker.
    - ``"iq_vs_ramp_duration"``: average I and Q vs ramp duration.

State update:
    - The initialisation macro ``ramp_duration`` on each qubit pair.
"""

node = QualibrationNode[Parameters, Quam](
    name="07a_init_ramp_rate_calibration",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes."""
    # You can get type hinting in your IDE by typing node.parameters.
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

    # Number of shots per ramp-duration point
    n_avg = node.parameters.num_shots

    # Build the ramp-duration sweep (in ns)
    ramp_min = int(node.parameters.ramp_duration_min)
    ramp_max = int(node.parameters.ramp_duration_max)
    ramp_step = int(node.parameters.ramp_duration_step)

    # An OPX clock cycle is 4ns. Therefore, all ramp durations must be divisible by 4
    if ramp_min % 4 != 0 or ramp_max % 4 != 0 or ramp_step % 4 != 0:
        raise ValueError(
            f"Ramp settings must be divisible by 4. " f"Got min={ramp_min}, max={ramp_max}, step={ramp_step}"
        )

    # If log is preferred, extract the desired resolution and generate a log scale. Else use a normal arange
    if node.parameters.ramp_log_scale:
        n_ramp_pts = int((ramp_max - ramp_min) // ramp_step)
        ramp_duration_array = np.logspace(
            np.log10(ramp_min),
            np.log10(ramp_max),
            n_ramp_pts,
            dtype=int,
            endpoint=True,
        )
    else:
        ramp_duration_array = np.arange(ramp_min, ramp_max, ramp_step, dtype=int)

    # Metadata for data fetching: labels the saved arrays when results come back from the OPX
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([qp.name for qp in qubit_pairs]),
        "ramp_duration": xr.DataArray(
            ramp_duration_array,
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        # Allocate real-time variables on the OPX:
        #   - n            : shot counter
        #   - n_st         : stream reporting shot index to PC (progress bar)
        #   - I_st/Q_st    : per-qubit-pair streams for demodulated I/Q
        #   - state_int/st : per-qubit-pair state assignment (0/1) and stream
        n = declare(int)
        n_st = declare_output_stream()
        I_st = {qp.name: declare_output_stream() for qp in qubit_pairs}
        Q_st = {qp.name: declare_output_stream() for qp in qubit_pairs}

        state_int = {qp.name: declare(int) for qp in qubit_pairs}
        state_st = {qp.name: declare_output_stream() for qp in qubit_pairs}

        # Real-time variable holding the ramp duration (ns)
        ramp_dur = declare(int)

        # ── OUTER LOOP: repeat the full sweep n_avg times ────────────────
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)  # tell the PC which shot we are on

            # ── For each qubit-pair, sweep ramp duration and measure ─────
            for qubit_pair in qubit_pairs:
                dot_pair = qubit_pair.quantum_dot_pair

                # ── INNER LOOP: sweep ramp duration ──────────────────────
                with for_(*from_array(ramp_dur, ramp_duration_array)):
                    # Initialize with the requested ramp duration
                    dot_pair.initialize(
                        ramp_duration=ramp_dur,
                        target_state=node.parameters.target_state,
                        max_loops=node.parameters.max_loops,
                    )
                    (i, q, state) = dot_pair.measure(return_iq=True)

                    # Store the boolean state as an int stream (0/1)
                    assign(
                        state_int[qubit_pair.name],
                        Cast.to_int(state),
                    )

                    align()

                    # Ramp back to zero, since all the outputs are sticky
                    dot_pair.voltage_sequence.ramp_to_zero()

                    # Append this point's data to the stream buffers
                    save(state_int[qubit_pair.name], state_st[qubit_pair.name])
                    save(i, I_st[qubit_pair.name])
                    save(q, Q_st[qubit_pair.name])

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")  # expose shot counter as "n" in the fetched dataset
            for qp in qubit_pairs:
                # Each save() above is one ramp-duration point.
                # .buffer(len(ramp_duration_array)) : group points along the ramp_duration axis
                state_st[qp.name].buffer(len(ramp_duration_array)).average().save(f"state_{qp.name}")
                I_st[qp.name].buffer(len(ramp_duration_array)).average().save(f"I_{qp.name}")
                Q_st[qp.name].buffer(len(ramp_duration_array)).average().save(f"Q_{qp.name}")


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
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused by the fetcher
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar while streaming data back
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start, node=node
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
    # Re-resolve qubit-pairs based on the loaded node parameters
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Find the optimal ramp duration from averaged state assignment data."""
    qubit_pairs = node.namespace["qubit_pairs"]
    qp_names = [qp.name for qp in qubit_pairs]

    ds_in = node.results["ds_raw"].copy(deep=True)
    ds_fit, fit_results = analyse_ramp_rate(
        ds_in,
        qp_names,
        find_minimum=node.parameters.find_minimum,
    )
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    # Log the relevant information extracted from the analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qp_name: ("successful" if r["success"] else "failed") for qp_name, r in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot average state assignment and IQ signal vs ramp duration."""
    qubit_pairs = node.namespace["qubit_pairs"]
    qp_names = [qp.name for qp in qubit_pairs]

    node.results["figures"] = plot_all(
        node.results.get("ds_fit", node.results["ds_raw"]),
        qp_names,
        fit_results=node.results.get("fit_results"),
    )
    node.results["figure"] = node.results["figures"]["avg_state_vs_ramp_duration"]

    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the initialisation macro ramp_duration on each qubit pair."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes.get(qp.name) != "successful":
                continue

            dot_pair = qp.quantum_dot_pair
            optimal_ramp = node.results["fit_results"][qp.name]["optimal_ramp_duration"]

            # Update the initialize macro if it supports state updates
            init_macro = dot_pair.macros.get("initialize")
            if init_macro is not None and hasattr(init_macro, "update"):
                init_macro.update(ramp_duration=optimal_ramp)
            else:
                node.log(f"  {qp.name}: no updatable initialize macro found on " f"{dot_pair.name}")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

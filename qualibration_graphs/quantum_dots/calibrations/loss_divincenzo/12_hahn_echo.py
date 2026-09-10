# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam

from calibration_utils.hahn_echo import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_all,
    generate_simulated_dataset,
)

from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot

# %% {Node initialization}
description = """
        HAHN ECHO (SPIN ECHO) T2 MEASUREMENT
This node measures the spin-spin relaxation time T2 using the Hahn echo (spin echo) technique.
Unlike Ramsey (T2*), the Hahn echo refocuses static dephasing and yields the intrinsic T2 coherence
time, which is always >= T2*.

The sequence is x90 - tau - y180 - tau - x90. The swept parameter tau is the duration
of each of the two idle gaps (after the first x90 and after the y180 refocusing pulse).
Total free evolution is 2*tau. The echo amplitude decays as exp(-2*tau/T2_echo).

Prerequisites:
    - Ramsey node (qubit frequency and T2*) and its prerequisites.
    - Calibrated x90 and y180 pulses from Rabi measurements.

Datasets:
    - ``ds_raw``: untouched ``state`` stream fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed sweeps plus analysis outputs (fitted traces and summary scalars).
      Used by ``plot_data``.
    - ``fit_results``: compact per-qubit calibration dict (``FitParameters`` serialized with
      ``asdict``). Used by logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][<qubit>]``):
    - ``success``: whether the exponential fit converged to a physical result.
    - ``T2_echo`` [ns]: Hahn echo coherence time.
    - ``amplitude``: echo contrast.
    - ``offset``: baseline level.
    - ``decay_rate`` [1/ns]: effective rate 2 / T2_echo.

Figures (``node.results["figures"]``):
    - ``"decay"``: horizontal subplots of state vs idle delay tau
      (each pi/2-pi segment; 2 tau total evolution) with exponential fit overlay.

State update:
    - ``qubit.T2echo`` from fitted ``T2_echo`` (successful qubits only).
"""


node = QualibrationNode[Parameters, Quam](name="12_hahn_echo", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local parameter overrides for debugging (ignored in the GUI / graph)."""
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 1D Hahn-echo idle-delay sweep and the QUA pulse sequence.

    Sweeps idle delay τ (each x90-y180 segment; total evolution 2τ).
    """
    # ── Experiment parameters (Python side) ──────────────────────────────
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # repetitions averaged at each idle-delay point

    # Tau axis: An array of wait times
    tau_values = np.arange(
        node.parameters.tau_min,
        node.parameters.tau_max,
        node.parameters.tau_step,
    )
    tau_clock_cycles = tau_values // 4  # ns → QUA clock cycles (1 cycle = 4 ns)

    # Metadata for data fetching: labels the saved state arrays when results come back from the OPX
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(
            tau_values,
            attrs={
                "long_name": "Hahn echo idle delay τ (each x90-y180 segment)",
                "units": "ns",
            },
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        # Real-time variables:
        # t         : idle delay in clock cycles
        # n         : shot counter
        # state[i]  : thresholded post-manipulation measurement (0/1) for qubit i
        t = declare(int)  # swept τ in QUA clock cycles (1 cycle = 4 ns)
        n = declare(int)  # shot counter
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        n_st = declare_output_stream()

        for i, qubit in enumerate(qubits):
            # ── OUTER LOOP: average n_avg shots per tau point ────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ── INNER LOOP: sweep per-arm idle time tau ────────────────────
                with for_(*from_array(t, tau_clock_cycles)):

                    # Reset the qubit's XY frame
                    reset_frame(qubit.xy.name)
                    align()

                    # Perform the initialize macro
                    qubit.initialize()
                    align()

                    # Perform the Hahn-echo sequence.
                    with strict_timing_():
                        qubit.x90()
                        wait(t, qubit.xy.name)
                        qubit.y180()
                        wait(t, qubit.xy.name)
                        qubit.x90()
                    align()

                    # Thresholded PSB readout → averaged state probability
                    s = qubit.measure()
                    assign(state[i], Cast.to_int(s))
                    save(state[i], state_st[i])

                    # Return gate voltages to zero before the next shot to avoid accumulation of fixed point errors
                    align()
                    qubit.voltage_sequence.ramp_to_zero()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Each save() is one wait time point.
                # .buffer(len(tau_values)) : group points along the tau axis
                # .average()               : average over all shots (n_avg repetitions)
                # Result: 1D state vs tau per qubit
                # Buffer tau sweep; average over shots -> 1D trace per qubit
                state_st[i].buffer(len(tau_values)).average().save(f"state{i + 1}")


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
    """Connect to the QOP, execute the QUA program, and fetch raw state data into ``ds_raw``."""
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
    """Generate synthetic Hahn echo data so the analysis pipeline runs without hardware."""
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
    """Fit an exponential decay to the Hahn echo data for each qubit.

    Processes raw state data, fits each qubit, and stores ``ds_fit`` (with fitted
    curves and summary scalars) and ``fit_results``.
    """
    ds_processed = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {qname: ("successful" if r["success"] else "failed") for qname, r in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot decay traces and fit overlays; store figures in ``node.results["figures"]``."""
    node.results["figures"] = plot_all(
        node.results["ds_fit"],
        node.results["fit_results"],
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write fitted T2_echo to ``qubit.T2echo`` for successful qubits."""

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_result = node.results["fit_results"][q.name]
            q.T2echo = fit_result["T2_echo"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist node results and parameters."""
    node.save()

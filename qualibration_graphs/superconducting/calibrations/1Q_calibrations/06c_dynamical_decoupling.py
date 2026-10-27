# %% {Imports}
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.dynamical_decoupling import (
    Parameters,
    get_dd_sequence,
    get_sweep_schedule,
    assign_schedule_coords,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_decay_curves,
    plot_t2_vs_pulses,
    plot_error_per_round,
    plot_noise_spectrum,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        DYNAMICAL DECOUPLING (CPMG, XY4, XY8, XY16)
Measures T2 of an idle qubit under different dynamical decoupling sequences, as a function of the number of pi pulses
applied during a fixed idle window (e.g. an ancilla readout in quantum error correction). The window is repeated to
record a decay for each number of pulses.

To see whether DD helps, the dephasing error per round is plotted against the number of pulses and compared with no
DD (T2*). The optimal number of pulses is the fewest that reach the lowest error. Optionally, a first-order dephasing
noise spectrum is computed from the measured T2.

Prerequisites:
    - Calibrated qubit frequency (06a_ramsey) and x90, x180, y180 pulses (04b_power_rabi, 10b_drag_calibration).
    - Optional: T1 (05_T1) and T2* (06a_ramsey) for the reference lines and the noise spectrum.
    - Flux point set if relevant (qubit.z.flux_point).

State update (one set of keys per sequence, shown here for CPMG):
    - qubit.extras["dd_CPMG_pulses_per_window"], ["dd_CPMG_tau_opt"], ["dd_CPMG_window"]
    - qubit.extras["dd_CPMG_T2"], ["dd_CPMG_alpha"], ["dd_CPMG_error_per_round"]
"""


node = QualibrationNode[Parameters, Quam](
    name="06c_dynamical_decoupling", description=description, parameters=Parameters(), machine=Quam.load()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    # node.parameters.sequence.sequence = "XY4"
    # node.parameters.sequence.window_ns = 1500
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.sweep.num_shots
    node.namespace["schedule"] = schedule = get_sweep_schedule(node.parameters, qubits)
    sequence = get_dd_sequence(schedule["sequence"])
    n_values = schedule["pulses_per_window"]
    window_counts = schedule["window_counts"]
    num_points = len(window_counts)
    node.log(
        f"{sequence.name}: window = {schedule['window_ns']} ns, pulses per window = {n_values.tolist()}, "
        f"{num_points} points from 0 to {window_counts.max()} windows "
        f"(up to {n_values.max() * window_counts.max()} pulses per decay curve)."
    )
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "pulses_per_window": xr.DataArray(n_values, attrs={"long_name": "pi pulses per window"}),
        "point": xr.DataArray(range(num_points), attrs={"long_name": "decay curve point"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        n_windows = declare(int)
        j = [declare(int) for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                # One window is unrolled per N; only the number of windows is swept in real time
                for i_n in range(len(n_values)):
                    with for_each_(n_windows, window_counts.tolist()):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            reset_frame(qubit.xy.name)
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        # Qubit manipulation
                        for i, qubit in multiplexed_qubits.items():
                            sequence.play(
                                qubit,
                                n_windows,
                                j[i],
                                n_pulses=int(n_values[i_n]),
                                free_cc=int(schedule["free_cc"][qubit.name][i_n]),
                                strict=node.parameters.sequence.use_strict_timing,
                            )
                        align()
                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(num_points).buffer(len(n_values)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(num_points).buffer(len(n_values)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(num_points).buffer(len(n_values)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
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
                node.parameters.sweep.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    node.results["ds_raw"] = assign_schedule_coords(dataset, node.namespace["schedule"])


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
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
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
    """Plot the decay curves, T2 vs pulses per window, the error per window and the noise spectrum, in figures shaped
    by qubit.grid_location."""
    qubits = node.namespace["qubits"]
    ds_fit = node.results["ds_fit"]
    figures = {
        "decay": plot_decay_curves(ds_fit, qubits),
        "t2_vs_pulses": plot_t2_vs_pulses(ds_fit, qubits),
        "error_per_round": plot_error_per_round(ds_fit, qubits),
    }
    if node.parameters.visualization.show_noise_spectrum:
        figures["noise_spectrum"] = plot_noise_spectrum(ds_fit, qubits)
    plt.show()
    # Store the generated figures
    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            fit_result = node.results["fit_results"][q.name]
            prefix = f"dd_{fit_result['sequence']}"
            q.extras[f"{prefix}_pulses_per_window"] = fit_result["pulses_per_window"]
            q.extras[f"{prefix}_tau_opt"] = fit_result["tau_opt"]
            q.extras[f"{prefix}_T2"] = fit_result["T2"]
            q.extras[f"{prefix}_alpha"] = fit_result["alpha"]
            q.extras[f"{prefix}_error_per_round"] = fit_result["error_per_round"]
            q.extras[f"{prefix}_window"] = fit_result["window"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

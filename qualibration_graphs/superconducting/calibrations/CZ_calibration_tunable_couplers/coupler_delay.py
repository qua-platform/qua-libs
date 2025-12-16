# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.coupler_delay import (
    FitResults,
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_oscillation_data,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs, get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Initialisation}
description = """
    ...
"""


# class Parameters():


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="delay_coupler",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/cz_conditional_phase/parameters.py
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubit_pairs = ["q1-q2"]
    node.parameters.qubits = ["qB1"]
    node.parameters.coupler = "qB1-B2"
    node.parameters.delay_range = 100
    node.parameters.num_averages = 100
    node.parameters.use_state_discrimination = True
    node.parameters.reset_type = "thermal"
    node.parameters.coupler_pulse_amp = 0.3
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_averages
    delays = np.arange(-node.parameters.delay_range / 4 + 4, node.parameters.delay_range / 4, 1, dtype=int)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "e_or_g": xr.DataArray(["e", "g"]),
        "delays": xr.DataArray(delays*4, attrs={"long_name": "delays", "units": "ns"}),
    }

    coupler = node.machine.qubit_pairs[node.parameters.coupler].coupler
    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        relative_delay = declare(int)  # QUA variable for the flux pulse segment index

        # --- Batch over qubits (allows time-multiplexed execution respecting hardware constraints)
        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()  # Ensure all elements start aligned before the averaging loop

            # --- Averaging loop
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)  # Save current average index for live progress

                # --- Initial state preparation loop: prepare |e> (via x180) and |g> (idle) for contrast
                for init_state in ["e", "g"]:
                    # --- Relative delay scan loop (index over baked XY+Z aligned segments)
                    with for_(*from_array(relative_delay, delays)):

                        # 1. Reset qubits to ground state
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()

                            # 2. State preparation: excited (x180) or ground (wait same duration)
                            if init_state == "e":
                                qubit.xy.play("x180")
                            elif init_state == "g":
                                qubit.xy.wait(qubit.xy.operations["x180"].length // 4)

                            # 4. Apply baked XY+Z segment with specific relative shift
                            coupler.wait(qubit.xy.operations["x180"].length // 4)
                            coupler.wait(node.parameters.delay_range // 4)
                            coupler.play(
                                "const",
                                amplitude_scale=node.parameters.coupler_pulse_amp
                                / coupler.operations["const"].amplitude,
                                duration=qubit.xy.operations["x180"].length // 4,
                            )
                            qubit.xy.wait(node.parameters.delay_range // 4 + relative_delay)
                            qubit.xy.play("x180")

                            align()
                            # 5. Measurement (state discrimination or I/Q acquisition)
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

            align()  # Final alignment after batch completes

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(delays)).buffer(2).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(delays)).buffer(2).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(delays)).buffer(2).average().save(f"Q{i + 1}")


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
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset."""
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
                data_fetcher["n"],
                node.parameters.num_averages,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %%

fig, ax = plt.subplots()
node.results["ds_raw"].state.sel(qubit=node.parameters.qubits[0]).plot.line(x="delays", linestyle="--", marker=".", ax=ax)

node.results["figure"] = fig

node.save()


# %% {Load_data}
# @node.run_action(skip_if=node.parameters.load_data_id is None)
# def load_data(node: QualibrationNode[Parameters, Quam]):
#     """Load a previously acquired dataset."""
#     load_data_id = node.parameters.load_data_id
#     # Load the specified dataset
#     node.load_from_id(node.parameters.load_data_id)
#     node.parameters.load_data_id = load_data_id
#     # Get the active qubit pairs from the loaded node parameters
#     node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# # %% {Analyse_data}
# @node.run_action(skip_if=node.parameters.simulate)
# def analyse_data(node: QualibrationNode[Parameters, Quam]):
#     """Analysis the raw data and store the fitted data in another xarray dataset and the fitted results."""
#     node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
#     node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)

#     # Store fit results in the format expected by the rest of the node
#     node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

#     # Log the relevant information extracted from the data analysis
#     log_fitted_results(fit_results, log_callable=node.log)

#     # Set outcomes based on fit success
#     node.outcomes = {
#         qubit_pair_name: ("successful" if fit_result.success else "failed")
#         for qubit_pair_name, fit_result in fit_results.items()
#     }


# # %% {Plot_data}
# @node.run_action(skip_if=node.parameters.simulate)
# def plot_data(node: QualibrationNode[Parameters, Quam]):
#     """Plot the JAZZ_ZZ coupling analysis and oscillation traces."""
#     qubit_pairs = node.namespace["qubit_pairs"]

#     # Plot coupling vs flux bias
#     fig_coupling = plot_raw_data_with_fit(node.results["ds_fit"], qubit_pairs, node)

#     # Plot oscillation traces for verification
#     fig_oscillations = plot_oscillation_data(node.results["ds_raw"], qubit_pairs)

#     node.results["coupling_figure"] = fig_coupling
#     node.results["oscillation_figure"] = fig_oscillations


# # %% {Update_state}
# @node.run_action(skip_if=node.parameters.simulate)
# def update_state(node: QualibrationNode[Parameters, Quam]):
#     """Update the relevant parameters if the qubit pair data analysis was successful."""

#     with node.record_state_updates():
#         fit_results = node.results["fit_results"]
#         for qp in node.namespace["qubit_pairs"]:
#             if node.outcomes[qp.name] == "failed":
#                 continue
#             node.machine.qubit_pairs[qp.name].coupler.decouple_offset += fit_results[qp.name]["optimal_amplitude"]


# # %% {Save_results}
# @node.run_action()
# def save_results(node: QualibrationNode[Parameters, Quam]):
#     node.save()


# # %%

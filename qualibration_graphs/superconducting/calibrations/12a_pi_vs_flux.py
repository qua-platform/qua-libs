# %% {Imports}
import dataclasses
from dataclasses import asdict
from datetime import datetime
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.pi_flux import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from scipy.optimize import curve_fit

from quam.components.pulses import DragGaussianPulse, GaussianPulse

# %% {Node_parameters}
description = """
Pi pulse vs flux calibration experiment.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="12a_pi_vs_flux",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["qD4", "qD3"]
    node.parameters.num_shots = 400
    node.parameters.update_lo = False
    node.parameters.frequency_span_in_mhz = 200
    node.parameters.frequency_step_in_mhz = 2
    node.parameters.operation_amplitude_factor = 1.0
    node.parameters.duration_in_ns = 9000
    node.parameters.reset_type = "thermal"
    node.parameters.multiplexed = True
    node.parameters.qubit_detuning_in_mhz = 300
    # node.parameters.load_data_id = 1875


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    node.namespace["qubits"] = qubits = get_qubits(node)
    n_avg = node.parameters.num_shots  # The number of averages
    operation = node.parameters.operation  # The qubit operation to play

    for qubit in qubits:
        # Check if the qubit has the required operations
        if hasattr(qubit.xy.operations, operation):
            continue
        else:
            x180 = qubit.xy.operations["x180"]
            qubit.xy.operations[operation] = DragGaussianPulse(
                length=int(x180.length),
                amplitude=float(x180.amplitude),
                sigma=int(x180.length / 4),
                alpha=0.0,
                anharmonicity=200e6,
                axis_angle=0.0,
            )
    # Modify the lo frequency to allow for maximum detuning
    tracked_qubits = []
    if node.parameters.update_lo:
        for q in qubits:
            with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
                q.xy.opx_output.upconverter_frequency -= 300e6
                # if q.xy.upconverter_frequency < 4.5e9:
                #     q.xy.opx_output.band = 1
                q.xy.RF_frequency -= 400e6
                tracked_qubits.append(q)

    node.namespace["tracked_qubits"] = tracked_qubits
    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None

    operation_amp = node.parameters.operation_amplitude_factor

    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, span // 2, step, dtype=np.int32)
    times = np.arange(4, node.parameters.duration_in_ns // 4, 60, dtype=np.int32)

    detunings = [node.parameters.qubit_detuning_in_mhz * u.MHz for q in qubits]

    flux_amplitudes = [
        float(np.sqrt(-node.parameters.qubit_detuning_in_mhz * u.MHz / q.freq_vs_flux_01_quad_term)) for q in qubits
    ]

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "time": xr.DataArray(4 * times, attrs={"long_name": "Flux pulse duration", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        df = declare(int)  # QUA variable for the qubit frequency
        t_delay = declare(int)


        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    with for_each_(t_delay, times):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency - detunings[i])
                            # Bring the qubit to the desired point during the saturation pulse
                            qubit.align()
                            qubit.z.play(
                                "const",
                                amplitude_scale=flux_amplitudes[i] / qubit.z.operations["const"].amplitude,
                                duration=t_delay + 200,
                            )
                            qubit.xy.wait(t_delay)
                            qubit.xy.play(operation, amplitude_scale=operation_amp)
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
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
            for i, qubit in enumerate(qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"Q{i + 1}")


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
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
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
    """Analyse the raw data and store the fitted data and results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result.success else "failed") for qubit_name, fit_result in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
# @node.run_action(skip_if=node.parameters.simulate)
# def update_state(node: QualibrationNode[Parameters, Quam]):
#     """Update the relevant parameters if the qubit data analysis was successful."""
#     with node.record_state_updates():
#         for q in node.namespace["qubits"]:
#             if node.outcomes[q.name] == "failed":
#                 continue

#             operation = q.xy.operations[node.parameters.operation]
#             operation.amplitude = node.results["fit_results"][q.name]["opt_amp"]
#             if node.parameters.operation == "x180":
#                 q.xy.operations["x90"].amplitude = node.results["fit_results"][q.name]["opt_amp"] / 2


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    for qubit in node.namespace["tracked_qubits"]:
        qubit.revert_changes()
    node.save()

# %%

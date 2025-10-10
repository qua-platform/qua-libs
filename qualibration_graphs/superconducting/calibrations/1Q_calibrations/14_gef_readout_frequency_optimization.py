# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.readout_gef_frequency_optimization import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_distances_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Description}
description = """
        G-E-F READOUT FREQUENCY OPTIMIZATION
This sequence sweeps the readout resonator intermediate frequency around the current operating point while preparing
the qubit successively in |g>, |e>, and |f> states. For every tested detuning, three IQ blobs (g, e, f) are acquired.
The distances between the three centroids are computed and fitted to identify the optimal frequency shift that
maximizes simultaneous separation (e.g. maximizes the minimum of {d_ge, d_ef, d_gf}). The resulting optimal detuning
is then added to the stored `GEF_frequency_shift` parameter.

Purpose:
    - Optimize a single readout frequency for high-fidelity three-level (g/e/f) state discrimination
        (including leakage monitoring).
    - Improve discrimination robustness against slow frequency drifts or residual mis-calibration.

Measurement flow:
    1. For each qubit, loop over the readout frequency detuning values.
    2. For every detuning value acquire ground state response (idle), excited state response (after x180), and second
       excited state response (after x180, frequency hop to f transition, EF_x180, hop back).
    3. Average IQ samples over all shots for each state & detuning.
    4. Compute centroid distances vs detuning and fit to extract the optimal detuning.

Prerequisites:
    - Resonator frequency & power calibrated (nodes 02a, 08a, 08b as relevant).
    - Qubit ge and ef pi pulses calibrated.
    - EF transition pulse `EF_x180` defined & roughly calibrated (requires anharmonicity knowledge).
    - Proper reset / thermalization parameters set (qubit.thermalization_time, reset_type).
    - An initial (possibly zero) value of `qubit.resonator.GEF_frequency_shift` present in the state.

State update:
    - Adds the fitted optimal detuning to `qubit.resonator.GEF_frequency_shift` for each qubit whose fit is successful.

Notes:
    - A fit can fail if the sweep span is too small or SNR too low; such qubits are flagged as 'failed' and not updated.
    - Increase span, number of shots, or readout power if separation is insufficient.
    - Ensure EF_x180 pulse and anharmonicity are accurate enough to reliably populate |f>.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="14_gef_frequency_optimization",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """
    Allow the user to locally set the node parameters for debugging purposes, or
    execution in the Python IDE.
    """
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Create the sweep axes and generate the QUA program from the pulse sequence and the
    node parameters.
    """
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_runs = node.parameters.num_shots  # Number of runs
    operation = node.parameters.operation

    # Frequency sweep in MHz
    frequencies = np.arange(
        -node.parameters.frequency_span_in_mhz * u.MHz / 2,
        node.parameters.frequency_span_in_mhz * u.MHz / 2,
        node.parameters.frequency_step_in_mhz * u.MHz,
    )
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "frequency": xr.DataArray(frequencies, attrs={"long_name": "readout frequency shift in MHz"}),
    }

    with program() as node.namespace["qua_program"]:
        I_g, I_g_st, Q_g, Q_g_st, n, n_st = node.machine.declare_qua_variables()
        I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.declare_qua_variables()
        I_f, I_f_st, Q_f, Q_f_st, _, _ = node.machine.declare_qua_variables()
        df = declare(int)

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)

                with for_(*from_array(df, frequencies)):
                    if qubit.resonator.GEF_frequency_shift is None:
                        qubit.resonator.GEF_frequency_shift = 0
                    qubit.resonator.update_frequency(
                        qubit.resonator.intermediate_frequency + qubit.resonator.GEF_frequency_shift + df
                    )
                    # Ground state iq blobs for all qubits
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.wait(2 * qubit.thermalization_time * u.ns)
                    align()
                    # Qubit readout
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure(operation, qua_vars=(I_g[i], Q_g[i]))
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        # save data
                        save(I_g[i], I_g_st[i])
                        save(Q_g[i], Q_g_st[i])
                    align()

                    # Excited state iq blobs for all qubits
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.wait(3 * qubit.thermalization_time * u.ns)
                    align()

                    # Qubit readout
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")
                        qubit.resonator.measure(operation, qua_vars=(I_e[i], Q_e[i]))
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        # save data
                        save(I_e[i], I_e_st[i])
                        save(Q_e[i], Q_e_st[i])

                    # Second excited state iq blobs for all qubits
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        if node.parameters.reset_type == "thermal":
                            qubit.wait(3 * qubit.thermalization_time * u.ns)
                    align()

                    # Qubit readout
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")
                        update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency - qubit.anharmonicity)
                        qubit.xy.play("EF_x180")
                        update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                        qubit.resonator.measure(operation, qua_vars=(I_f[i], Q_f[i]))
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        # save data
                        save(I_f[i], I_f_st[i])
                        save(Q_f[i], Q_f_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_g_st[i].buffer(len(frequencies)).average().save(f"Ig{i + 1}")
                Q_g_st[i].buffer(len(frequencies)).average().save(f"Qg{i + 1}")
                I_e_st[i].buffer(len(frequencies)).average().save(f"Ie{i + 1}")
                Q_e_st[i].buffer(len(frequencies)).average().save(f"Qe{i + 1}")
                I_f_st[i].buffer(len(frequencies)).average().save(f"If{i + 1}")
                Q_f_st[i].buffer(len(frequencies)).average().save(f"Qf{i + 1}")


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
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw".
    """
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
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)


# %%
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
    """
    Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary.
    """
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
    """
    Plot the raw and fitted data in specific figures whose shape is given by
    qubit.grid_location.
    """
    fig = plot_distances_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
    )
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "fitted_distances": fig,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            node.machine.qubits[q.name].resonator.GEF_frequency_shift += node.results["fit_results"][q.name][
                "optimal_detuning"
            ]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%

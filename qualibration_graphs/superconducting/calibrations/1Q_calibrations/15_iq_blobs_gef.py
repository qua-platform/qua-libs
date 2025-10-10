# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.iq_blobs_ef import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_confusion_matrices,
    plot_iq_blobs,
    process_raw_dataset,
)
from qm.qua import *
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
        IQ BLOBS GEF
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit in
the |g> state), then after applying a x180 (pi) pulse to the qubit (bringing the qubit to the |e> state) and finally
after applying a x180 (pi) pulse plus an EF_180 pulse (bringing the qubit to the |f> state).
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The centers of the |g>, |e> and |f> state IQ blobs.
    - The readout confusion matrix, which is also influenced by the x180 and EF_180 pulses fidelities.

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters.
    - Having calibrated the qubit EF_180 pulse parameters.

State update:
    - The gef centers positions: qubit.resonator.gef_centers
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="15_iq_blobs_gef",  # Name should be unique
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
    node.parameters.qubits = ["qD1", "qD2"]
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
    if node.parameters.reset_type != "thermal":
        raise ValueError("Only 'thermal' reset is supported")
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_runs = node.parameters.num_shots  # Number of runs
    operation = node.parameters.operation
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}),
    }

    with program() as node.namespace["qua_program"]:
        I_g, I_g_st, Q_g, Q_g_st, n, n_st = node.machine.declare_qua_variables()
        I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.declare_qua_variables()
        I_f, I_f_st, Q_f, Q_f_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            for i, qubit in multiplexed_qubits.items():
                shift = qubit.resonator.GEF_frequency_shift if qubit.resonator.GEF_frequency_shift is not None else 0
                qubit.resonator.update_frequency(
                    qubit.resonator.intermediate_frequency + shift
                )  # resonator frequency shift for GEF

            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)

                # Ground state iq blobs for all qubits
                # Qubit initialization
                for i, qubit in multiplexed_qubits.items():
                    qubit.wait(2 * qubit.thermalization_time * u.ns)  # longer wait for |f> thermalization
                align()
                # |g> state readout
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
                    qubit.wait(2 * qubit.thermalization_time * u.ns)  # longer wait for |f> thermalization
                align()
                # |e> state readout
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    qubit.resonator.measure(operation, qua_vars=(I_e[i], Q_e[i]))
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    # save data
                    save(I_e[i], I_e_st[i])
                    save(Q_e[i], Q_e_st[i])

                # Second excited state iq blobs for all qubits
                # Qubit reset
                for i, qubit in multiplexed_qubits.items():
                    qubit.wait(2 * qubit.thermalization_time * u.ns)  # longer wait for |f> thermalization
                align()
                # |f> state readout
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
                I_g_st[i].buffer(n_runs).save(f"Ig{i + 1}")
                Q_g_st[i].buffer(n_runs).save(f"Qg{i + 1}")
                I_e_st[i].buffer(n_runs).save(f"Ie{i + 1}")
                Q_e_st[i].buffer(n_runs).save(f"Qe{i + 1}")
                I_f_st[i].buffer(n_runs).save(f"If{i + 1}")
                Q_f_st[i].buffer(n_runs).save(f"Qf{i + 1}")


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


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
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
    # Keep a dict version for persistence, but use the original dataclass objects for logging
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log using the dataclass objects (they have attribute access expected by log_fitted_results)
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result_dict["success"] else "failed")
        for qubit_name, fit_result_dict in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """
    Plot the raw and fitted data in specific figures whose shape is given by
    qubit.grid_location.
    """
    fig_iq = plot_iq_blobs(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_confusion = plot_confusion_matrices(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "iq_blobs": fig_iq,
        "confusion_matrix": fig_confusion,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            operation = q.resonator.operations[node.parameters.operation]
            node.machine.qubits[q.name].resonator.gef_centers = (
                node.results["ds_fit"].sel(qubit=q.name).center_matrix.data * operation.length / 2**12
            ).tolist()  # convert to raw adc units


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%

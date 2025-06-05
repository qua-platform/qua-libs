# %% {Imports}
from dataclasses import asdict
import matplotlib.pyplot as plt
import xarray as xr
from qm import QuantumMachinesManager

from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from calibration_utils.power_rabi import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from configuration.configuration_with_lf_fem_and_mw_fem import *

description = """
        A template for a node that contains a QUA program.
"""


node = QualibrationNode[Parameters, None](name="hello_qua_test", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, None]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.num_shots = 10
    node.parameters.simulate = False
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, None]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Get the active qubits from the node and organize them by batches
    # The qubit operation to play
    operation = node.parameters.operation
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(['q1', 'q2']),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
    }
    # define the qubits in the program
    qubits = ['q1', 'q2']
    resonators = ['q1_resonator', 'q2_resonator']
    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        n = declare(int) # QUA variable for the number of averages
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        I = [declare(fixed) for _ in range(len(qubits))]  # QUA variable for the measured 'I' quadrature
        Q = [declare(fixed) for _ in range(len(qubits))]  # QUA variable for the measured 'Q' quadrature
        I_st = [declare_stream for _ in range(len(qubits))] # Stream for the 'I' quadrature
        Q_st = [declare_stream for _ in range(len(qubits))]  # Stream for the 'Q' quadrature
        n_st = declare_stream()
        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)
            with for_(*from_array(a, amps)):
                for i, (qubit, resonator) in enumerate(zip(qubits, resonators)):
                    play(operation * amp(a), qubit)
                    wait(250 * u.ns, qubit)
                    measure(
                        "readout",
                        resonator,
                        None,
                        dual_demod.full("rotated_cos", "rotated_sin", I[i]),
                        dual_demod.full("rotated_minus_sin", "rotated_cos", Q[i]),
                    )
                    wait(100, resonator)
                    # Save the 'I' & 'Q' quadratures to their respective streams
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(len(qubits)):
                I_st[i].buffer(len(amps)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amps)).average().save(f"Q{i + 1}")

# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, None]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, None]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    # with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    qm = qmm.open_qm(config)
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
    print(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset

# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, None]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters



# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, None]):
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
def plot_data(node: QualibrationNode[Parameters, None]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, None]):
    node.save()



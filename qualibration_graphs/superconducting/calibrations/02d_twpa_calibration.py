# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict
import warnings

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.twpa_calibration import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
)

from calibration_utils.twpa_calibration.plotting import plot_raw_data_with_fit

from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        TWPA calibration
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the state.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

State update:
    - Update the relevant flux biases in the state.
    - Save the current state
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="twpa_calibration",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1"]
    pass


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
    # Get the TWPAs from the node
    node.namespace["twpas"] = twpas = node.machine.twpas
    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    # The frequency sweep around the resonator resonance frequency
    pump_frequency_span = node.parameters.pump_frequency_span_in_mhz * u.MHz
    pump_frequency_step = node.parameters.pump_frequency_step_in_mhz * u.MHz
    max_amp_factor = node.parameters.max_amp_factor
    min_amp_factor = node.parameters.min_amp_factor
    amp_factor_step = node.parameters.amp_factor_step
    resonator_frequency_span = node.parameters.frequency_span_in_mhz * u.MHz
    resonator_frequency_step = node.parameters.frequency_step_in_mhz * u.MHz
    amps = np.arange(min_amp_factor, max_amp_factor, amp_factor_step)
    amps = np.insert(amps, 0, 0)
    dfs = np.arange(-resonator_frequency_span / 2, +resonator_frequency_span / 2, resonator_frequency_step)
    dfps = np.arange(-pump_frequency_span / 2, +pump_frequency_span / 2, pump_frequency_step)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
        "pump_amp": xr.DataArray(amps, attrs={"long_name": "TWPA pump amplitude", "units": "V"}),
        "pump_frequency": xr.DataArray(dfps, attrs={"long_name": "TWPA pump frequency", "units": "Hz"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
        # For instance, here 'I' is a python list containing two QUA fixed variables.
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        dfp = declare(int)  # QUA variable for the pump frequency
        a = declare(fixed)  # QUA variable for the pump amplitude
        df = declare(int)  # QUA variable for the readout frequency
        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
                save(n, n_st)
                with for_(*from_array(dfp, dfps)):  # QUA for_ loop for sweeping the pump frequency
                    for i, qubit in multiplexed_qubits.items():
                        rr = qubit.resonator
                        # Update the pump frequency
                        rr.twpa.pump.update_frequency(dfp + rr.twpa.pump.intermediate_frequency)
                        # QUA for_ loop for sweeping the pump amplitude
                        with for_each_(a, amps):
                            # play the pump pulse
                            rr.twpa.pump.play("const", amplitude_scale=a)
                            # wait 1us for pump to settle
                            # wait(250)
                            with for_(*from_array(df, dfs)):
                                # Update the resonator frequencies for all resonators
                                rr.update_frequency(df + rr.intermediate_frequency)
                                # Measure the resonator
                                rr.measure("readout", qua_vars=(I[i], Q[i]))
                                # wait for the resonator to deplete
                                rr.wait(rr.depletion_time * u.ns)
                                # save data
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfps)).buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfps)).buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i + 1}")


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
    node.results["fit_results"] = fit_results

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {"successful" if fit_results.success else "fail"}


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
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    # Revert the change done at the beginning of the node
    for tracked_qubit in node.namespace.get("tracked_qubits", []):
        tracked_qubit.revert_changes()

    # Update the state
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            # Update the Pump frequency and power
        q.resonator.pump.freqeuncy = node.results["fit_results"][q.name]
        # q.resonator.pump.power =


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.twpa_gain_curve import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_gain,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

description = """
TWPA GAIN CURVE AT OPTIMAL PUMPING POINT

do spectroscopy around the bandwidth ~(7GHz ~ 7.6GHz : our usual readout bandwidth)
for various readout power ~(-120dBm~-95dBm)
with the pump off and on(optimal pumping condition is given through node 001) and get the Gain.
For each signal frequency, get the Gain as a function of readout power
and get the input power at which 1dB deviation from linear gain emerges(P1dB)

Prerequisites:
    - Having calibrated the optimal twpa pumping point (nodes 001). All the gain compression 
      is measured under the given pumping point which is obatained through node001

* Gain is defined as the increase in the signal level.
    - twpa pump off : measure the signal response within 600MHz around the readout bandwidth
      singal_off= signal[dB] 
    - twpa pump on :  measure signal response within 600MHz around the readoutbandwidth
      singal_on= signal[dB]
    => gain=signal_on-signal_off
* P1dB : measure the gain as a function of readout amplitude and find the point where
        gain drops by 1dB relative to the small-signal gain (linear gain)
        input signal power at which the amplifier's gain experiences a 1dB reduction
* pump_ : need to use non sticky pump(pump_) for twpa calibration
  pump  : sticky pump is for general twpa usage not for calibration
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="22_twpa_gain_curve",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1", "q2"]
    # node.parameters.simulate = True
    node.parameters.twpas = ["twpaA"]
    node.parameters.num_shots = 30
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
    # Get the TWPAs
    node.namespace["twpas"] = twpas = [node.machine.twpas[t] for t in node.parameters.twpas]
    num_twpas = len(twpas)
    # Get the mapping between qubits and TWPAs
    node.namespace["qubit_to_twpa"] = {q: twpa for twpa in node.machine.twpas for q in node.machine.twpas[twpa].qubits}
    # Update the readout central frequency, this change will be reverted at the end of the node.
    node.namespace["tracked_resonators"] = []

    for i, twpa in enumerate(twpas):
        for q_name in twpa.qubits:
            q = node.machine.qubits[q_name]
            with tracked_updates(q.resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
                resonator.opx_output.upconverter_frequency = node.parameters.frequency_center_in_mhz * u.MHz
                resonator.RF_frequency = node.parameters.frequency_center_in_mhz * u.MHz
                node.namespace["tracked_resonators"].append(resonator)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    # The frequency sweep around the resonator resonance frequency
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "twpa": xr.DataArray(node.parameters.twpas),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        Ioff_st = [declare_stream() for _ in range(num_qubits)]
        Qoff_st = [declare_stream() for _ in range(num_qubits)]
        df = declare(int)  # QUA variable for the readout frequency
        twpa_on = declare(bool)

        with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
            save(n, n_st)
            with for_each_(twpa_on, [False, True]):
                for i, twpa in enumerate(twpas):
                    rr = node.machine.qubits[twpa.qubits[0]].resonator
                    with if_(twpa_on):
                        twpa.pump.play("pump")
                        twpa.pump.wait(twpa.settling_time // 4)
                        twpa.isolation.play("pump")
                        twpa.isolation.wait(twpa.settling_time // 4)
                        align()

                    with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
                        # Update the resonator frequencies for all resonators
                        rr.update_frequency(df + rr.intermediate_frequency)
                        # readout the resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        # save data
                        with if_(twpa_on):
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                        with else_():
                            save(I[i], Ioff_st[i])
                            save(Q[i], Qoff_st[i])
                        # wait for the resonator to deplete
                        rr.wait(rr.depletion_time // 4)
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_twpas):
                I_st[i].buffer(len(dfs)).average().save(f"Ion{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Qon{i + 1}")
                Ioff_st[i].buffer(len(dfs)).average().save(f"Ioff{i + 1}")
                Qoff_st[i].buffer(len(dfs)).average().save(f"Qoff{i + 1}")


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
        data_fetcher = XarrayDataFetcher(node.namespace["job"], node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
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
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_gain = plot_gain(node.results["ds_fit"], node.namespace["twpas"], node)
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "gain": fig_gain,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    # Revert the change done at the beginning of the node
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    # Update the state
    with node.record_state_updates():
        for q in node.namespace["qubit_to_twpa"].keys():
            if node.outcomes[q] == "failed":
                continue


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import json
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.twpa_calibration import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_gain,
    plot_snr,
    plot_iqblobs,
)
from quam_builder.tools.power_tools import calculate_voltage_scaling_factor
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

description = """
TWPA CALIBRATION FOR OPTIMAL PUMPING POINT

Sweep pump frequency and amplitude to find the optimal pump frequency and pump amplitude for the TWPA.
For each pumping point, calculate Gain, SNR improvement.
Optimize pumping point for the worst SNR Qubit so that the multiplexed readout could be done faster without losing SNR.
* Gain is defined as the increase in the signal level.
    - twpa pump off : measure the signal response 4MHz around the readout resonator when the pump is off
      signal_off= signal[dB] 
    - twpa pump on :  measure signal response 4MHz around the readout resonator when the pump is on
      signal_on= signal[dB]
    => gain=signal_on-signal_off
* SNR improvement is defined by the difference in the signal to noise ratio between pump on and pump off.
    - twpa pump off : measure the signal response 4 MHz around the readout resonator twice with 
      measure(amp=0) for noise level
      measure(amp=from state file) for signal level
      snr_off= signal[dB]-noise[dB]
    - twpa pump on :  measure the signal response 4 MHz around the readout resonator twice with
      measure(amp=0) for noise level
      measure(amp=from state file) for signal level
    => dsnr=snr_on-snr_off
Prerequisites:
    - Need to know in which frequency dispersive feature of TWPA RPM resonator appears
    - Having calibrated the resonator frequency (nodes 02a, 02b and/or 02c).
    - Having calibrated the worst SNR Qubit 
How to use optimizers : 
    - average optimized pumping point: define mingain and mindsnr, then the function will return the optimized pump frequency and pump amplitude
      which maximizes the average dSNR among the pumping points which satisfies the minimum gain and minimum dSNR conditions for all qubits
    - multiplexed readout optimized pumping point: define mingain, mindsnr and poorqubit index, then the function will return the optimized pump frequency and pump amplitude
      which maximizes the dSNR of the poor qubit among the pumping points which satisfies the minimum gain and minimum dSNR conditions for all qubits
Before proceeding to the next node:
    - Updates the optimal pump frequency and pump amplitude for the TWPA
    (average optimal point & multiplexed readout optimal point) in the state
    - Save the current state
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="01_twpa_calibration",  # Name should be unique
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
    # node.parameters.num_shots = 30
    # node.parameters.frequency_span_in_mhz_p = 1
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program_off(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Extract the sweep parameters and axes from the node parameters
    n_shots = node.parameters.num_shots
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes_off"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "shots": xr.DataArray(np.linspace(1, n_shots, n_shots), attrs={"long_name": "shot number"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program_off"]:
        # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the readout frequency

        for multiplexed_qubits in qubits.batch():
            with for_(n, 0, n < n_shots, n + 1):  # QUA for_ loop for averaging
                save(n, n_st)
                for i, qubit in multiplexed_qubits.items():
                    rr = qubit.resonator
                    # Update the resonator frequencies for all resonators
                    rr.update_frequency(df + rr.intermediate_frequency)
                    # readout the resonator
                    rr.measure("readout", qua_vars=(I[i], Q[i]))
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    # wait for the resonator to deplete
                    rr.wait(rr.depletion_time // 4)
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(n_shots).average().save(f"Ioff{i + 1}")
                Q_st[i].buffer(n_shots).average().save(f"Qoff{i + 1}")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program_on(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Get the TWPAs
    twpas = [node.machine.twpas[t] for t in node.parameters.twpas]
    # Get the mapping between qubits and TWPAs
    node.namespace["qubit_to_twpa"] = {q.name: twpa for twpa in node.machine.twpas for q in node.namespace["qubits"]}
    # Update the twpa pump/isolation powers to match the desired range, this change will be reverted at the end of the node.
    node.namespace["tracked_twpas"] = []
    for i, twpa in enumerate(twpas):
        with tracked_updates(twpa, auto_revert=False, dont_assign_to_none=True) as twpa:
            twpa.pump.set_output_power(
                power_in_dbm=node.parameters.max_power_dbm_p, max_amplitude=node.parameters.max_amp_p, operation="pump"
            )
            twpa.pump_.set_output_power(
                power_in_dbm=node.parameters.max_power_dbm_p, max_amplitude=node.parameters.max_amp_p, operation="pump"
            )
            node.namespace["tracked_twpas"].append(twpa)
    # Extract the sweep parameters and axes from the node parameters
    n_shots = node.parameters.num_shots
    # The twpa pump frequency sweep around the optimal frequency
    span_p = node.parameters.frequency_span_in_mhz_p * u.MHz
    step_p = node.parameters.frequency_step_in_mhz_p * u.MHz
    dfs_p = np.arange(-span_p / 2, +span_p / 2, step_p)
    if len(dfs_p) == 0:
        dfs_p = [0]
    # The twpa pump amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
    amp_min_p = calculate_voltage_scaling_factor(node.parameters.max_power_dbm_p, node.parameters.min_power_dbm_p)
    amps_p = np.geomspace(amp_min_p, 1, node.parameters.num_power_points_p)
    power_dbm_p = np.linspace(
        node.parameters.min_power_dbm_p,
        node.parameters.max_power_dbm_p,
        node.parameters.num_power_points_p,
    )
    data_size = len(power_dbm_p) * len(dfs_p) * num_qubits * 2 * 4
    assert data_size < 100e6, f"The maximum data size is 100e6, you ask for {data_size:.2e} data points."

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes_on"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning_p": xr.DataArray(dfs_p, attrs={"long_name": "twpa pump frequency", "units": "Hz"}),
        "twpa_power_p": xr.DataArray(power_dbm_p, attrs={"long_name": "TWPA pump power", "units": "dBm"}),
        "shots": xr.DataArray(np.linspace(1, n_shots, n_shots), attrs={"long_name": "shot number"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program_on"]:
        # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        a_p = declare(fixed)  # QUA variable for the twpa pump amplitude pre-factor
        df_p = declare(int)  # QUA variable for the twpa pump frequency

        for multiplexed_qubits in qubits.batch():
            twpa = twpas[0]
            with for_(*from_array(df_p, dfs_p)):  # QUA for_ loop for sweeping the frequency
                twpa.pump.update_frequency(df_p + twpa.pump.intermediate_frequency)
                with for_each_(a_p, amps_p):  # QUA for_ loop for sweeping the twpa pump amplitude
                    twpa.pump.play("pump", amplitude_scale=a_p)
                    twpa.pump.wait(twpa.settling_time // 4)
                    align()
                    with for_(n, 0, n < n_shots, n + 1):  # QUA for_ loop for averaging
                        for i, qubit in multiplexed_qubits.items():
                            rr = qubit.resonator
                            # readout the resonator
                            rr.measure("readout", qua_vars=(I[i], Q[i]))
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                            # wait for the resonator to deplete
                            rr.wait(rr.depletion_time // 4)
                        align()
                    ramp_to_zero(twpa.pump.name)

        with stream_processing():
            for i in range(num_qubits):
                I_st[i].buffer(n_shots).buffer(len(amps_p)).buffer(len(dfs_p)).save(f"Ion{i + 1}")
                Q_st[i].buffer(n_shots).buffer(len(amps_p)).buffer(len(dfs_p)).save(f"Qon{i + 1}")


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
        node.namespace["job_off"] = job = qm.execute(node.namespace["qua_program_off"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(node.namespace["job_off"], node.namespace["sweep_axes_off"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw_off"] = dataset


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
        node.namespace["job_on"] = job = qm.execute(node.namespace["qua_program_on"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(node.namespace["job_on"], node.namespace["sweep_axes_on"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw_on"] = dataset


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
    # Load the best TWPA parameters as a dict
    node.results["ds_fit"].attrs["coords_best"] = json.loads(node.results["ds_fit"].attrs["coords_best"])
    node.results["ds_raw"].attrs["coords_best"] = json.loads(node.results["ds_raw"].attrs["coords_best"])


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_gain = plot_gain(node.results["ds_raw"], node.results["ds_fit"].qubit.values, node.results["ds_fit"], node)
    fig_snr = plot_snr(node.results["ds_raw"], node.results["ds_fit"].qubit.values, node.results["ds_fit"], node)
    fig_iqblobs = plot_iqblobs(
        node.results["ds_raw"], node.results["ds_fit"].qubit.values, node.results["ds_fit"], node
    )
    plt.show()
    # Store the generated figures
    node.results["figures"] = {"gain": fig_gain, "snr": fig_snr, "iq_blobs": fig_iqblobs}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    # Revert the change done at the beginning of the node
    for tracked_twpas in node.namespace.get("tracked_twpas", []):
        tracked_twpas.revert_changes()

    # Update the state
    with node.record_state_updates():
        for twpa_n in node.parameters.twpas:
            twpa = node.machine.twpas[twpa_n]
            q = twpa.qubits[0]
            if node.outcomes[twpa.qubits[0]] == "failed":
                continue

            twpa.pump.set_output_power(
                power_in_dbm=node.results["fit_results"][q]["twpa_power_p"],
                max_amplitude=node.parameters.max_amp_p,
                operation="pump",
            )
            twpa.pump_.set_output_power(
                power_in_dbm=node.results["fit_results"][q]["twpa_power_p"],
                max_amplitude=node.parameters.max_amp_p,
                operation="pump",
            )
            # Update the readout frequency for the given flux point
            twpa.pump_frequency = node.results["fit_results"][q]["twpa_frequency_p"]
            twpa.pump.RF_frequency = node.results["fit_results"][q]["twpa_frequency_p"]
            twpa.pump_.RF_frequency = node.results["fit_results"][q]["twpa_frequency_p"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    # Make the coords_best attribute serializable
    if type(node.results["ds_fit"].attrs["coords_best"]) is dict:
        coords_serializable = {k: v for k, v in node.results["ds_fit"].coords_best.items()}
        node.results["ds_fit"].attrs["coords_best"] = json.dumps(coords_serializable)
        node.results["ds_raw"].attrs["coords_best"] = json.dumps(coords_serializable)
    node.save()

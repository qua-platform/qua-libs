# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.hello_qua import Parameters
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.common_utils.experiment import get_dots, get_sensors

description = """
        Basic script to play with the QUA program and test the QOP connectivity.
"""


node = QualibrationNode[Parameters, Quam](name="00_hello_qua", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.multiplexed = True
    # node.parameters.num_shots = 2
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    node.namespace["sensor_dots"] = sensors = get_sensors(node)
    num_sensors = len(sensors)

    n_points = node.parameters.num_points

    f_step = node.parameters.frequency_step_in_mhz * u.MHz
    f_span = node.parameters.frequency_span_in_mhz * u.MHz
    f_array = np.arange(-f_span / 2, f_span / 2, f_step)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors),
        "frequency": xr.DataArray(f_array, attrs={"long_name": "frequency", "units": "Hz"}),
    }

    # node.namespace["sweep"]
    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
        df = declare(int)  # QUA variable for the readout frequency
        for multiplexed_sensors in sensors.batch():
            align()
            with for_(n, 0, n < n_points, n + 1):
                save(n, n_st)
                with for_(*from_array(df, f_array)):
                    align()
                    for i, sensor in multiplexed_sensors.items():
                        # Select the resonator tied to the sensor
                        rr = sensor.readout_resonator
                        # Update the frequency
                        rr.update_frequency(df + rr.intermediate_frequency)
                        # Measure using said resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        # Post-measurement wait (Optional)
                        rr.wait(500)

                        # Save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            # This example doesn't save I/Q, adjust if needed
            # I_st[0].buffer(len(voltages)).average().save("I1")
            # Q_st[0].buffer(len(voltages)).average().save("Q1")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    # qmm = node.machine.connect()
    # # Get the config from the machine
    # config = node.machine.generate_config()
    # # Simulate the QUA program, generate the waveform report and plot the simulated samples
    # samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # # Store the figure, waveform report and simulated samples
    # node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    # qmm = node.machine.connect()
    # # Get the config from the machine
    # config = node.machine.generate_config()
    # # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    # with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    #     # The job is stored in the node namespace to be reused in the fetching_data run_action
    #     node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
    #     # Display the progress bar
    #     data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
    #     for dataset in data_fetcher:
    #         progress_counter(
    #             data_fetcher.get("n", 0),
    #             node.parameters.num_shots,
    #             start_time=data_fetcher.t_start,
    #         )
    #     # Display the execution report to expose possible runtime errors
    #     print(job.execution_report())
    # # Register the raw dataset
    # node.results["ds_raw"] = dataset


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    # node.save()

# %% {Imports}
import matplotlib.pyplot as plt
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

    node.namespace["quantum_dots"] = quantum_dots = get_dots(node)
    node.namespace["sensors"] = sensors = get_sensors(node)

    num_sensors = len(sensors)

    # Ensure that all components are part of the same VirtualGateSet, for compensation reasons
    virtual_gate_set_ids = [qd.voltage_sequence.gate_set.id for qd in quantum_dots] + [
        sensor.voltage_sequence.gate_set.id for sensor in sensors
    ]
    if len(set(virtual_gate_set_ids)) > 1:
        raise ValueError(
            f"Quantum dots and sensors must be part of the same VirtualGateSet. VirtualGateSet IDs: {virtual_gate_set_ids}"
        )
    vgs_id = virtual_gate_set_ids[0]

    v_center = node.parameters.v_center
    v_span = node.parameters.v_span
    n_points = node.parameters.num_points

    if node.parameters.dc_control:
        node.machine.connect_to_external_source(external_qdac=True)
        node.machine.virtual_dc_sets[vgs_id].set_voltages({qd.name: v_center for qd in quantum_dots})
        voltages = np.linspace(-v_span / 2, +v_span / 2, n_points)
    else:
        voltages = np.linspace(v_center - v_span / 2, v_center + v_span / 2, n_points)
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "voltage": xr.DataArray(voltages, attrs={"long_name": "voltage", "units": ""}),
    }
    # node.namespace["sweep"]
    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        seq = node.machine.virtual_gate_sets[vgs_id].new_sequence()

        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
        v = declare(fixed)

        # Average on outermost
        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)
            # Step each QD first
            with for_(*from_array(v, voltages)):
                for multiplexed_dots in quantum_dots.batch():
                    with seq.simultaneous():
                        for j, qd in multiplexed_dots.items():
                            qd.go_to_voltages({qd.name: v}, duration=node.parameters.dwell_time)
                    align()
                    # Measure each batch, multiplexed by sensors
                    for multiplexed_sensors in sensors.batch():
                        for i, sensor in multiplexed_sensors.items():
                            # Select the resonator tied to the sensor
                            rr = sensor.readout_resonator
                            # Measure using said resonator
                            rr.measure("readout", qua_vars=(I[i], Q[i]))
                            # Post-measurement wait (Optional)
                            rr.wait(500)

                            # Save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
                I_st[i].buffer(len(voltages)).average().save(f"I{i}")
                Q_st[i].buffer(len(voltages)).average().save(f"Q{i}")


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
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        print(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["quantum_dots"] = get_dots(node)
    node.namespace["sensors"] = get_sensors(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Compute amplitude and phase from the raw I/Q data."""
    ds_raw = node.results["ds_raw"]
    ds_raw["amplitude"] = np.sqrt(ds_raw["I"] ** 2 + ds_raw["Q"] ** 2)
    phase = np.arctan2(ds_raw["Q"], ds_raw["I"])
    ds_raw["phase"] = phase.copy(data=np.unwrap(phase.values))
    node.results["ds_raw"] = ds_raw


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot I and Q vs voltage for each quantum dot, showing all sensor responses."""
    ds_raw = node.results["ds_raw"]
    quantum_dots = node.namespace["quantum_dots"]
    sensors = node.namespace["sensors"]
    voltages = node.namespace["sweep_axes"]["voltage"].values

    node.results["figures"] = {}
    for qd in quantum_dots:
        fig, axes = plt.subplots(1, len(sensors), figsize=(5 * len(sensors), 4), squeeze=False)
        fig.suptitle(qd.name)
        for i, sensor in enumerate(sensors):
            sensor_data = ds_raw.sel(sensors=sensor.name)
            ax = axes[0, i]
            ax.plot(voltages, sensor_data["I"].values, label="I")
            ax.plot(voltages, sensor_data["Q"].values, label="Q")
            ax.set_title(sensor.name)
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Signal (a.u.)")
            ax.legend()
        fig.tight_layout()
        node.results["figures"][qd.name] = fig
    plt.show()


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()

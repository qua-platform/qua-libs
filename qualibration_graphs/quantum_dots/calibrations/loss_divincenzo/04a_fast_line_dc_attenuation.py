# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict
import time

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from quam.components import pulses

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.fast_line_dc_attenuation import Parameters
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.fast_line_dc_attenuation import validate_and_add_square_wave


# %% {Node initialisation}
description = """
        FAST LINE DC ATTENUATION CALIBRATION - Calibrate the attenuation difference between the fast lines and DC lines
This node aims to calibrate the attenuation difference between the fast line of plunger/barrier gates and the associated DC lines. This is
done by sending a square wave to the fast line, while varying the DC voltage. That way, the fast movement of transition peaks can
be compared to the slowly varying DC, which gives us the ratio and therefore the attenuation value. The DC is assumed to be via a QDAC for this node.

This node takes place before the calibration of the virtual matrices. This may lower the SNR of the sensor dot, but will not change the relative
location of the transition from AC to DC.

Future nodes will all aim to be run in the external voltage space. The VoltageSequence will automatically scale up the OPX output to match the
required voltage scale.

Prerequisites:
    - Having calibrated the sensor operating point.
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having set up the external DC settings in your Quam state.

State update:
    - The attenuation value of the VoltageGate elements, so that the VoltageSequence can scale the outputs.
"""


node = QualibrationNode[Parameters, Quam](
    name="04a_fast_line_dc_attenuation", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.quantum_dot_pair_names = ["virtual_dot_1_virtual_dot_2_pair"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):

    dc_array = np.arange(
        -node.parameters.dc_sweep_span / 2, node.parameters.dc_sweep_span / 2, node.parameters.dc_sweep_step
    )

    node.namespace["sensor_names"] = sensors = [node.machine.sensor_dots[s] for s in node.parameters.sensor_names]
    node.namespace["components"] = components = node.parameters.components  # This gives list of strings
    machine_channel_list = [node.machine.get_component(name) for name in components]  # This gives the actual objects
    node.namespace["original_offsets"] = dc_offsets = {}
    node.namespace["tracked_operations"] = []

    # For each chosen component, add a square wave operation (tracked) and save the current offset, so that it can be returned later
    for ch in machine_channel_list:
        validate_and_add_square_wave(node, ch, node.namespace["tracked_operations"])
        channel_offset = ch.physical_channel.offset_parameter()
        dc_offsets[ch.physical_channel.name] = channel_offset

    # A dict of the arrays to be used in the measurement. Accessed below in execute
    node.namespace["dc_list_values"] = dc_list_values = {
        channel.physical_channel.name: dc_offsets[channel.physical_channel.name] + dc_array
        for channel in machine.channel_list
    }

    node.namespace["sweep_axes"] = {
        "components": xr.DataArray(components),
        "dc_values": xr.DataArray(
            [dc_list_values[c.physical_channel.name] for c in machine_channel_list],
            attrs={"long_name": "voltage", "units": "V"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()

        i_dc = declare(int)

        I = {s.name: {comp_name: declare(fixed) for comp_name in components} for s in sensors}
        Q = {s.name: {comp_name: declare(fixed) for comp_name in components} for s in sensors}
        I_st = {s.name: {comp_name: declare_stream() for comp_name in components} for s in sensors}
        Q_st = {s.name: {comp_name: declare_stream() for comp_name in components} for s in sensors}

        for ch in machine_channel_list:  # These are NOT strings. These are the objects.
            chan = ch.physical_channel
            with for_(i_dc, 0, i_dc < len(dc_array), i_dc + 1):
                pause()  # Step the external instrument to the correct location
                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    align()
                    chan.play("square_wave")
                    for s in sensors:
                        I[s.name][ch.name], Q[s.name][ch.name] = s.measure("readout")
                        save(I[s.name][ch.name], I_st[s.name][ch.name])
                        save(Q[s.name][ch.name], Q_st[s.name][ch.name])

        with stream_processing():
            n_st.save("n")
            for s in sensors:
                for comp_name in components:
                    I_st[s.name][comp_name].average().buffer(len(dc_array)).save(f"I_{comp_name}_sensor_{s.name}")
                    Q_st[s.name][comp_name].average().buffer(len(dc_array)).save(f"Q_{comp_name}_sensor_{s.name}")


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
    # node.machine.connect() should already connect to the external source and wire the offset_parameters based on the wiring.json
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).

    node.namespace["gate_set_names"] = gate_set_names = {
        node.machine.get_component(k).physical_channel.name: node.machine.get_component(k).voltage_sequence.gate_set
        for k in node.namespace["components"]
    }

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])

        for physical_channel_name, dc_values in node.namespace["dc_list_values"].items():
            for val in dc_values:
                while not job.is_paused():
                    time.sleep(0.01)  # Wait until next pause

                node.machine.virtual_dc_sets[gate_set_names[physical_channel_name]].set_voltages(
                    {physical_channel_name: val}
                )
                job.resume()

        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
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
    pass


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor data analysis was successful."""
    # Re-set the DC offsets to the previous state
    for phys_name, val in node.namespace["original_offsets"].items():
        gate_set_name = node.namespace["gate_set_names"][phys_name]
        node.machine.virtual_dc_sets[gate_set_name].set_voltages({phys_name: val})

    with node.record_state_updates():
        # This is a characterization measurement and typically does not update state parameters.
        # If needed in the future, the identified PSB region coordinates and optimal detuning could be stored.
        # Example of potential state update (commented out):
        # for qubit_pair in node.namespace["qubit_pairs"]:
        #     if not node.results["fit_results"][qubit_pair.name]["success"]:
        #         continue
        #     # Update PSB region coordinates and optimal detuning if needed
        # TODO: how to update the PSB region coordinates and optimal detuning for a given qd pair?
        pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

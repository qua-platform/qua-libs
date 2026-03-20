# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from quam.components import pulses

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.fast_line_dc_attenuation import Parameters, SquareWave
from calibration_utils.charge_stability.parameters import prepare_dc_lists
from qualang_tools.loops import from_array
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Node initialisation}
description = """
        FAST LINE DC ATTENUATION CALIBRATION QDAC - Calibrate the attenuation difference between the fast lines and DC lines
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
    - Having calibrated the "empty" and "initialization" voltage points, and having defined the detuning axis.
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

    node.machine.connect_to_external_source(
        external_qdac="qdac_ip" in node.machine.network or "qdac_usb_port" in node.machine.network
    )

    node.namespace["sensor_names"] = sensors = [node.machine.sensor_dots[s] for s in node.parameters.sensor_names]

    if node.parameters.components is None:
        components = (
            list(node.machine.quantum_dots.keys())
            + list(node.machine.sensor_dots.keys())
            + list(node.machine.barrier_gates.keys())
        )
        # + list(node.machine.global_gates.keys()),

    else:
        components = node.parameters.components

    node.namespace["components"] = components  # List of names
    machine_channel_list = [node.machine.get_component(name) for name in components]

    dc_array = np.arange(
        -node.parameters.dc_sweep_span / 2, node.parameters.dc_sweep_span / 2, node.parameters.dc_sweep_step
    )

    dc_list_values = {}
    for ch in machine_channel_list:
        if (
            not type(ch.physical_channel).__name__ == "VoltageGate"
        ):  # Not isinstance, to avoid import. Can change if necessary
            raise ValueError(
                f"Channel {ch.name}'s physical_channel is not a VoltageGate instance, but is {type(ch.physical_channel).__name__}."
            )
        # elif ch.physical_channel.offset_parameter is None:
        #     raise ValueError(f"Channel {ch.name}'s physical_channel does not have an offset_parameter")
        elif ch.physical_channel.qdac_spec is None:
            raise ValueError(f"Channel {ch.name}'s physical_channel has no Qdac Spec.")

        # Temporarily add a square wave to the physical channel elements, with an IF
        ch.physical_channel.operations["square_wave"] = SquareWave(
            length=max([s.readout_resonator.operations["readout"].length for s in sensors]),
            amplitude=node.parameters.square_wave_amplitude,
            frequency_hz=node.parameters.square_wave_frequency,
        )

        channel_offset = ch.physical_channel.offset_parameter()
        dc_list_values[ch.physical_channel.name] = channel_offset + dc_array

    node.namespace["dc_list_values"] = dc_list_values

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

        for ch in components:  # These are strings
            chan = node.machine.get_component(ch).physical_channel

            pause()  # Loads the DC list into the QDAC
            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)

                with for_(i_dc, 0, i_dc < len(dc_array), i_dc + 1):
                    # Trigger the DC list
                    chan.qdac_spec.opx_trigger_out.play("trigger")

                    wait(chan.settling_time or 200_000)  # Wait for QDAC to settle

                    align()
                    chan.play("square_wave")

                    for s in sensors:
                        I[s.name][ch], Q[s.name][ch] = s.measure("readout")
                        save(I[s.name][ch], I_st[s.name][ch])
                        save(Q[s.name][ch], Q_st[s.name][ch])
                        wait(500)

        with stream_processing():
            n_st.save("n")
            for s in sensors:
                for comp_name in components:
                    I_st[s.name][comp_name].buffer(len(dc_array)).average().save(f"I_{comp_name}_sensor_{s.name}")
                    Q_st[s.name][comp_name].buffer(len(dc_array)).average().save(f"Q_{comp_name}_sensor_{s.name}")


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
        for ch_name in node.namespace["components"]:
            ch = node.machine.get_component(ch_name)
            vgs_id = ch.voltage_sequence.gate_set.id
            dc_list_values = (
                np.arange(
                    -node.parameters.dc_sweep_span / 2, node.parameters.dc_sweep_span / 2, node.parameters.dc_sweep_step
                )
                + ch.physical_channel.offset_parameter()
            )  # Measure the actual physical output at the moment
            prepare_dc_lists(
                node=node,
                virtual_dc_set_id=vgs_id,
                axis_name=ch.physical_channel.name,  # Avoid virtualisation, work purely in the physical space for this calibration.
                axis_values=dc_list_values,
                trigger=ch.physical_channel.qdac_spec.qdac_trigger_in,
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

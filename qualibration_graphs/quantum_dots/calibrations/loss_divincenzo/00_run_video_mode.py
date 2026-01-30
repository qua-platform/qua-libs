# %%
from quam_config import Quam
from qm import QuantumMachinesManager
from qualibrate import QualibrationNode, NodeParameters
from calibration_utils.run_video_mode import create_video_mode, Parameters


description = """
        RUN VIDEO MODE.
"""

node = QualibrationNode[Parameters, Quam](
    name="00_run_video_mode", description=description, parameters=Parameters()
)

node.machine = Quam.load("/Users/kalidu_laptop/.qualibrate/quam_state")

# %%
@node.run_action()
def run_video_mode(node: QualibrationNode[Parameters, Quam]):
    machine = node.machine
    readout_pulses = [sensor.readout_resonator.operations["readout"] for sensor in [node.machine.sensor_dots[sensor] for sensor in node.parameters.sensor_names]]

    x_axis_name = node.parameters.x_axis_name
    x_mode = node.parameters.x_axis_mode
    y_axis_name = node.parameters.y_axis_name
    y_mode = node.parameters.y_axis_mode
    num_software_averages = node.parameters.num_shots
    create_video_mode(
        machine = machine, 
        log = node.log, 
        x_axis_name = x_axis_name, 
        y_axis_name = y_axis_name, 
        x_mode = x_mode, 
        y_mode = y_mode,
        num_software_averages = num_software_averages,
        virtual_gate_id = node.parameters.virtual_gate_set_id, 
        dc_control = node.parameters.dc_control, 
        readout_pulses = readout_pulses, 
        save_path = "/Users/User/.qualibrate/quam_state"
    )


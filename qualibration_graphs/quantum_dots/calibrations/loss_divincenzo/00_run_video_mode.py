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

    create_video_mode(
        machine = node.machine, 
        x_axis_name = node.parameters.x_axis_name, 
        y_axis_name = node.parameters.y_axis_name, 
        virtual_gate_id = "main_qpu", 
        dc_control = node.parameters.dc_control, 
        readout_pulses = [node.machine.sensor_dots[name].readout_resonator.operations["readout"] for name in node.parameters.sensor_names], 
        save_path = "/Users/kalidu_laptop/.qualibrate/user_storage"
    )

# %% {Imports}
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.charge_stability_qdac import Parameters, get_voltage_arrays

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.common_utils.experiment import get_dots, get_sensors, _make_batchable_list_from_multiplexed

description = """
            OPX & QDAC 2D CHARGE STABILITY MAP
This script involves a simple 2D voltage map, done by stepping the X and Y Quantum Dots 
to their corresponding voltages, sending a readout pulse, and demodulating the 'I' and 'Q'
quadratures. In this node, you may perform the 2D map using either OPX outputs or QDAC 
voltage source outputs, which are triggered by the OPX. 

Note: Currently the external v external 2D map has a large number of pause() functions, which 
    increases the runtime drastically. Use with caution. 

Prerequisites: 
    - Having calibrated the IQ mixer/Octave connected to the readout line (node 01a_mixer_calibration.py).
    - Having calibrated the time of flight, offsets, and gains (node 01a_time_of_flight.py).
    - Having calibrated the resonators coupled to the SensorDot components (nodes 02a_resonator_spectroscopy.py, 02b_resonator_spectroscopy_vs_power.py).
    - Having initialized the QUAM state parameters for the readout pulse amplitude and duration.
    - Having registered the QuantumDot elements and your SensorDot elements in your QUAM state. 
    - Having configured the QdacSpec on each of the VoltageGate objects.
    - Having configured the VirtualDCSet in your machine.
"""


node = QualibrationNode[Parameters, Quam](name="04b_charge_stability_qdac", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.multiplexed = True
    # node.parameters.num_shots = 2
    pass


# # Instantiate the QUAM class from the state file
# node.machine = Quam.load("/Users/kalidu_laptop/.qualibrate/quam_state")

# # %% {Create_QUA_program}
# @node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.run_in_video_mode)
# def create_qua_program(node: QualibrationNode[Parameters, Quam]):
#     """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
#     # Class containing tools to help handle units and conversions.
#     u = unit(coerce_to_integer=True)

#     virtual_gate_set = node.machine.virtual_gate_sets[node.parameters.virtual_gate_set_id]
#     x_obj, y_obj = node.machine.get_component(node.parameters.x_axis_name), node.machine.get_component(node.parameters.y_axis_name)
#     x_volts, y_volts = get_voltage_arrays(node)
#     quantum_dot_pair = node.machine.find_quantum_dot_pair(node.parameters.x_axis_name, node.parameters.y_axis_name)
#     dwell_time = node.parameters.points_duration

#     sensor_dot_list = node.machine.quantum_dot_pairs[quantum_dot_pair].sensor_dots
#     node.namespace["sensors"] = sensors = _make_batchable_list_from_multiplexed(
#         sensor_dot_list, 
#         False
#     )

#     # Connect machine to QDAC
#     node.machine.connect_to_external_source(external_qdac=True)
#     # Set up the DC lists 
#     dc_list_x = node.machine.qdac.channel(x_obj.physical_channel.qdac_channel).dc_list(
#             voltages = x_volts, 
#             dwell_s = dwell_time*2, 
#             stepped = True
#     )
#     dc_list_x.start_on_external(trigger = 1)

#     dc_list_y = node.machine.qdac.channel(y_obj.physical_channel.qdac_channel).dc_list(
#             voltages = y_volts, 
#             dwell_s = dwell_time*2, 
#             stepped = True
#     )
#     dc_list_y.start_on_external(trigger = 2)



#     num_sensors = len(sensors)

#     # Register the sweep axes to be added to the dataset when fetching data
#     node.namespace["sweep_axes"] = {
#         "sensors": xr.DataArray(sensors.get_names()),
#         "x_volts": xr.DataArray(x_volts, attrs={"long_name": "voltage", "units": "V"}),
#         "y_volts": xr.DataArray(y_volts, attrs={"long_name": "voltage", "units": "V"}),
#     }
#     x_external, y_external = node.parameters.x_from_qdac, node.parameters.y_from_qdac

#     # node.namespace["sweep"]
#     # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions

#     # Case 1: Both axes OPX voltages
#     if not x_external and not y_external: 
#         with program() as node.namespace["qua_program"]: 
#             seq = virtual_gate_set.new_sequence()

#             I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs = num_sensors)
#             x = declare(fixed)
#             y = declare(fixed)

#             for multiplexed_sensors in sensors.batch():
#                 align() 
#                 with for_(n, 0, n<node.parameters.num_shots, n+1):
#                     save(n, n_st)
#                     with for_(*from_array(x, x_volts)): 
#                         with for_(*from_array(y, y_volts)):
#                             with seq.simultaneous():
#                                 x_obj.go_to_voltages({x_obj.id: x}, duration = dwell_time)
#                                 y_obj.go_to_voltages({y_obj.id: y}, duration = dwell_time)
#                             align()
#                             for i, sensor in multiplexed_sensors.items():
#                                 # Select the resonator tied to the sensor
#                                 rr = sensor.readout_resonator
#                                 # Measure using said resonator
#                                 rr.measure("readout", qua_vars = (I[i], Q[i]))
#                                 # Post-measurement wait (Optional)
#                                 rr.wait(500)

#                                 # Save data
#                                 save(I[i], I_st[i])
#                                 save(Q[i], Q_st[i])
#             with stream_processing():
#                 n_st.save("n")
#                 for i in range(num_sensors):
#                     I_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"I{i}")
#                     Q_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"Q{i}")
        
#     # Case 2: X external and Y OPX
#     elif x_external and not y_external: 
#         with program() as node.namespace["qua_program"]: 
#             seq = virtual_gate_set.new_sequence()

#             I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs = num_sensors)
#             x = declare(fixed)
#             y = declare(fixed)

#             for multiplexed_sensors in sensors.batch():
#                 align()
#                 # We know that the X is the slow axis. Order it so that the X axis comes first
#                 with for_(n, 0, n<node.parameters.num_shots, n+1): 
#                     save(n, n_st)
#                     with for_(*from_array(x, x_volts)): 
#                         x_obj.physical_channel.qdac_trigger.play("trigger")
#                         with for_(*from_array(y, y_volts)):
#                             y_obj.go_to_voltages({y_obj.id: y}, duration = dwell_time)
#                             align()
#                             for i, sensor in multiplexed_sensors.items():
#                                 # Select the resonator tied to the sensor
#                                 rr = sensor.readout_resonator
#                                 # Measure using said resonator
#                                 rr.measure("readout", qua_vars = (I[i], Q[i]))
#                                 # Post-measurement wait (Optional)
#                                 rr.wait(500)

#                                 # Save data
#                                 save(I[i], I_st[i])
#                                 save(Q[i], Q_st[i])
#             with stream_processing():
#                 n_st.save("n")
#                 for i in range(num_sensors):
#                     I_st[i].buffer(len(y_volts)).average().buffer(len(x_volts)).save(f"I{i}")
#                     Q_st[i].buffer(len(y_volts)).average().buffer(len(x_volts)).save(f"Q{i}")

#     # Case 3: X OPX and Y external 
#     elif not x_external and y_external: 
#         # Transpose so that the slow (Y) is on the outer loop
#         node.namespace["sweep_axes"] = {
#             "sensors": xr.DataArray(sensors.get_names()),
#             "y_volts": xr.DataArray(y_volts, attrs={"long_name": "voltage", "units": "V"}),
#             "x_volts": xr.DataArray(x_volts, attrs={"long_name": "voltage", "units": "V"}),
#         }
#         with program() as node.namespace["qua_program"]: 
#             seq = virtual_gate_set.new_sequence()

#             I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs = num_sensors)
#             x = declare(fixed)
#             y = declare(fixed)

#             for multiplexed_sensors in sensors.batch():
#                 align()
#                 # We know that the Y is the slow axis. Order it so that the Y axis comes first
#                 with for_(n, 0, n<node.parameters.num_shots, n+1): 
#                     save(n, n_st)
#                     with for_(*from_array(y, y_volts)): 
#                         y_obj.physical_channel.qdac_trigger.play("trigger")
#                         with for_(*from_array(x, x_volts)):
#                             x_obj.go_to_voltages({x_obj.id: x}, duration = dwell_time)
#                             align()
#                             for i, sensor in multiplexed_sensors.items():
#                                 # Select the resonator tied to the sensor
#                                 rr = sensor.readout_resonator
#                                 # Measure using said resonator
#                                 rr.measure("readout", qua_vars = (I[i], Q[i]))
#                                 # Post-measurement wait (Optional)
#                                 rr.wait(500)

#                                 # Save data
#                                 save(I[i], I_st[i])
#                                 save(Q[i], Q_st[i])
#             with stream_processing():
#                 n_st.save("n")
#                 for i in range(num_sensors):
#                     I_st[i].buffer(len(x_volts)).average().buffer(len(y_volts)).save(f"I{i}")
#                     Q_st[i].buffer(len(x_volts)).average().buffer(len(y_volts)).save(f"Q{i}")

#     # Case 4: Both external 
#     elif x_external and y_external: 
#         with program() as node.namespace["qua_program"]: 
#             seq = virtual_gate_set.new_sequence()

#             I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs = num_sensors)
#             x = declare(fixed)
#             y = declare(fixed)

#             for multiplexed_sensors in sensors.batch():
#                 align()
#                 # We know that the Y is the slow axis. Order it so that the Y axis comes first
#                 with for_(n, 0, n<node.parameters.num_shots, n+1): 
#                     save(n, n_st)
#                     with for_(*from_array(x, x_volts)): 
#                         x_obj.physical_channel.qdac_trigger.play("trigger")
#                         with for_(*from_array(y, y_volts)):
#                             y_obj.physical_channel.qdac_trigger.play("trigger")
#                             for i, sensor in multiplexed_sensors.items():
#                                 # Select the resonator tied to the sensor
#                                 rr = sensor.readout_resonator
#                                 # Measure using said resonator
#                                 rr.measure("readout", qua_vars = (I[i], Q[i]))
#                                 # Post-measurement wait (Optional)
#                                 rr.wait(500)

#                                 # Save data
#                                 save(I[i], I_st[i])
#                                 save(Q[i], Q_st[i])
#             with stream_processing():
#                 n_st.save("n")
#                 for i in range(num_sensors):
#                     I_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"I{i}")
#                     Q_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"Q{i}")


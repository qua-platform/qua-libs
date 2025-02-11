import numpy as np
from quam_libs.components import Transmon
from qm.qua import declare, for_, switch_, case_
from quam_libs.macros import active_reset

def reset_qubits(node, control: Transmon, target: Transmon, thermalization_time: float | None = None):
    if node.parameters.reset_type == "active":
        active_reset(control, "readout")
        active_reset(target, "readout")
    else:
        control.resonator.wait(thermalization_time)

# def gates_to_name_list(qc: QuantumCircuit) -> list[str]:
    
#     qc_str = []
#     for instruction in qc:
#         qc_str.append(instruction.name)

# def play_gate(gate: int, element: Transmon, angle: float, int_to_gate_map: dict):
    
#     with switch_(gate, unsafe=False):
#         with case_(0): # I
#             pass
        
#         with case_(1): # Rz
#             qubit.xy.frame_rotation(angle)
        
#         with case_(2): # X180
#             qubit.xy.play("x180")
                                            
#         with case_(3): # x90
#             qubit.xy.play("x90")
            
#         with case_(4): # cz
#             qubit.gate.play("-x90")
            
#         with case_(14): # y90
#             qubit.xy.frame_rotation(-np.pi/2)
#             qubit.xy.play("x90")
#             qubit.xy.frame_rotation(np.pi/2)
                            
#         with case_(15): # -y90
#             qubit.xy.frame_rotation(np.pi/2)
#             qubit.xy.play("x90")
#             qubit.xy.frame_rotation(-np.pi/2)
                            
#         with case_(16): # Z90
#             qubit.xy.frame_rotation(np.pi/2)
            
#         with case_(17): # -Z90
#             qubit.xy.frame_rotation(-np.pi/2)
            
#         with case_(18): # x180 y90
#             qubit.xy.play("x90")
#             qubit.xy.frame_rotation(-np.pi/2)
#             qubit.xy.play("x90")
            
#         with case_(19):  # x180 -y90
#             qubit.xy.play("x90")
#             qubit.xy.frame_rotation(np.pi/2)
#             qubit.xy.play("x90")
            
#         with case_(20): # Y180 X90
#             qubit.xy.frame_rotation(-np.pi/2)
#             qubit.xy.play("x90")
#             qubit.xy.frame_rotation(np.pi/2)
#             qubit.xy.play("x90")
#             qubit.xy.frame_rotation(np.pi/2)
                            
#         with case_(21): # Y180 -X90
#             qubit.xy.frame_rotation(-np.pi/2)
#             qubit.xy.play("x90")
#             qubit.xy.frame_rotation(np.pi/2)
#             qubit.xy.play("-x90")
#             qubit.xy.frame_rotation(-np.pi/2)
            
#         with case_(22): # x90 Y90 X90
#             qubit.xy.play("x180")
#             qubit.xy.frame_rotation(np.pi/2)
            
#         with case_(23): # -x90 Y90 -X90
#             qubit.xy.play("x180")
#             qubit.xy.frame_rotation(-np.pi/2)

# def play_sequence(sequence_list, start: int, length: int, qubit: Transmon):
#     i = declare(int)
#     with for_(i, start, i < start + length, i + 1):
#         play_gate(sequence_list[i], qubit)
        
            
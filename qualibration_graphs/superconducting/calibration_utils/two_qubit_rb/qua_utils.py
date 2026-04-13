from typing import Literal
from more_itertools import flatten
import numpy as np
from quam_builder.architecture.superconducting.qubit_pair.flux_tunable_transmon_pair import FluxTunableTransmonPair as TransmonPair
from qm.qua import * 
from qm.qua._expressions import QuaVariable
from qualibrate import QualibrationNode
from qualang_tools.units import unit

def play_gate(gate: QuaVariable, qubit_pair: dict[int, TransmonPair], state: QuaVariable, state_control: QuaVariable, state_target: QuaVariable, state_st: "_ResultSource", reset_type: Literal["thermal", "active"], simulate: bool = False):
    with switch_(gate, unsafe=True):
                               
        with case_(0):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.qubit_target.xy.play("x90")
                qp.align()
        with case_(1):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.qubit_target.xy.play("x180")
                qp.align()
        with case_(2):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.qubit_target.xy.frame_rotation(np.pi/2)
                qp.align()
        with case_(3):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.qubit_target.xy.frame_rotation(np.pi)
                qp.align()
        with case_(4):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.qubit_target.xy.frame_rotation(3*np.pi/2)
                qp.align()
        with case_(5):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.align()
        with case_(6):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.qubit_target.xy.play("x90")
                qp.align()
        with case_(7):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.qubit_target.xy.play("x180")
                qp.align()
        with case_(8):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.qubit_target.xy.frame_rotation(np.pi/2)
                qp.align()
        with case_(9):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.qubit_target.xy.frame_rotation(np.pi)
                qp.align()
        with case_(10):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.qubit_target.xy.frame_rotation(3*np.pi/2)
                qp.align()
        with case_(11):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.align()
        with case_(12):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi/2)
                qp.qubit_target.xy.play("x90")
                qp.align()
        with case_(13):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi/2)
                qp.qubit_target.xy.play("x180")
                qp.align()
        with case_(14):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi/2)
                qp.qubit_target.xy.frame_rotation(np.pi/2)
                qp.align()
        with case_(15):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi/2)
                qp.qubit_target.xy.frame_rotation(np.pi)
                qp.align()
        with case_(16):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi/2)
                qp.qubit_target.xy.frame_rotation(3*np.pi/2)
                qp.align()
        with case_(17):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi/2)
                qp.align()
        with case_(18):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi)
                qp.qubit_target.xy.play("x90")
                qp.align()
        with case_(19):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi)
                qp.qubit_target.xy.play("x180")
                qp.align()
        with case_(20):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi)
                qp.qubit_target.xy.frame_rotation(np.pi/2)
                qp.align()
        with case_(21):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi)
                qp.qubit_target.xy.frame_rotation(np.pi)
                qp.align()
        with case_(22):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi)
                qp.qubit_target.xy.frame_rotation(3*np.pi/2)
                qp.align()
        with case_(23):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(np.pi)
                qp.align()
        with case_(24):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(3*np.pi/2)
                qp.qubit_target.xy.play("x90")
                qp.align()
        with case_(25):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(3*np.pi/2)
                qp.qubit_target.xy.play("x180")
                qp.align()
        with case_(26):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(3*np.pi/2)
                qp.qubit_target.xy.frame_rotation(np.pi/2)
                qp.align()
        with case_(27):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(3*np.pi/2)
                qp.qubit_target.xy.frame_rotation(np.pi)
                qp.align()
        with case_(28):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(3*np.pi/2)
                qp.qubit_target.xy.frame_rotation(3*np.pi/2)
                qp.align()
        with case_(29):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.frame_rotation(3*np.pi/2)
                qp.align()
        with case_(30):
            for qp in qubit_pair.values():
                qp.qubit_target.xy.play("x90")
                qp.align()
        with case_(31):
            for qp in qubit_pair.values():
                qp.qubit_target.xy.play("x180")
                qp.align()
        with case_(32):
            for qp in qubit_pair.values():
                qp.qubit_target.xy.frame_rotation(np.pi/2)
                qp.align()
        with case_(33):
            for qp in qubit_pair.values():
                qp.qubit_target.xy.frame_rotation(np.pi)
                qp.align()
        with case_(34):
            for qp in qubit_pair.values():
                qp.qubit_target.xy.frame_rotation(3*np.pi/2)
                qp.align()
        with case_(35): # idle gate
            for qp in qubit_pair.values():
                qp.qubit_control.wait(4)
                qp.qubit_target.wait(4)
                qp.align()
        with case_(36): #CZ
            for qp in qubit_pair.values():
                qp.macros['cz'].apply()
                qp.align()
        with case_(37): # idle_2q
            for qp in qubit_pair.values():
                # qp.qubit_control.wait(int(1e9*(qp.qubit_control.T1/1000)) // 4)
                # qp.qubit_target.wait(int(1e9*(qp.qubit_target.T1/1000)) // 4)
                qp.qubit_control.wait(4)
                qp.qubit_target.wait(4)
                qp.align()
        
        with case_(38): # readout and thermalization
            
            align()
            
            for i, qp in qubit_pair.items():
                # qp.qubit_control.xy.align(qp.qubit_target.xy.name, qp.qubit_control.resonator.name, qp.qubit_target.resonator.name)
               
                
                wait(8)
                
                qp.qubit_control.readout_state(state_control)
                qp.qubit_target.readout_state(state_target)
                assign(state, state_control*2 + state_target)
                save(state, state_st[i])
            
            align()   
             
            for qp in qubit_pair.values():
                
                # reset the qubits
                qp.qubit_control.reset(reset_type, simulate)
                qp.qubit_target.reset(reset_type, simulate)
                
            align()
                
                # # Reset the frame of the qubits in order not to accumulate rotations
                # reset_frame(qp.qubit_control.xy.name, qp.qubit_target.xy.name)
                
                # align()
                # qp.qubit_control.xy.align(qp.qubit_target.xy.name, qp.qubit_control.resonator.name, qp.qubit_target.resonator.name)


class QuaProgramHandler:
    def __init__(self, node: QualibrationNode, circuits_as_ints: list, qubit_pairs: list):
        self.u = unit(coerce_to_integer=True)
        self.node = node
        self.circuits_as_ints = circuits_as_ints
        self.qubit_pairs = qubit_pairs
        self.num_pairs = len(qubit_pairs)

    def _build_qua_program(self):
        
        job_sequence = list(flatten(self.circuits_as_ints))
        sequence_length = len(job_sequence)
        
        
        with program() as rb:
    
            n_st = declare_stream()
            job_sequence_qua = declare(int, value=job_sequence)
            # The relevant streams
            state_st = [declare_stream() for _ in range(self.num_pairs)]

            # Initialize the flux on qubits
            self.node.machine.initialize_qpu(target=self.qubit_pairs[0].qubit_control)
            
            for multiplexed_qubit_pairs in self.qubit_pairs.batch():
                
                n = declare(int)
                state_control = declare(int)
                state_target = declare(int)
                state = declare(int)
                i = declare(int)

                # reset
                for qp in multiplexed_qubit_pairs.values():
                    qp.qubit_control.reset(self.node.parameters.reset_type, self.node.parameters.simulate)
                    qp.qubit_target.reset(self.node.parameters.reset_type, self.node.parameters.simulate)
                
                align()
                
                # play sequences
                with for_(n, 0, n < self.node.parameters.num_averages, n + 1):      
                        with for_(i, 0, i < sequence_length, i + 1):
                                play_gate(job_sequence_qua[i], multiplexed_qubit_pairs, state, state_control, state_target, 
                                          state_st, self.node.parameters.reset_type, 
                                          self.node.parameters.simulate)    
                        save(n, n_st)
                
                align() 
            

            with stream_processing():
                n_st.save("iteration")
                for i in range(len(self.qubit_pairs)):
                    state_st[i].buffer(self.node.parameters.num_circuits_per_length).buffer(len(self.node.parameters.circuit_lengths)).buffer(self.node.parameters.num_averages).save(
                        f"state{i + 1}"
                    )
        return rb

    def get_qua_program(self):
        return self._build_qua_program()
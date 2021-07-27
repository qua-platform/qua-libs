from QuaGST import QuaGST
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig
from pygsti.modelpacks import smq1Q_XYI
from example_configuration import *

qmm = QuantumMachinesManager()


def x_pi2():
    play("x_pi/2", "qe1")


def y_pi2():
    frame_rotation_2pi(1 / 4, "qe1")
    play("x_pi/2", "qe1")


def id_gate():
    wait(1000, "qe1")


def post_circuit(out_st):
    I = declare(fixed)
    measure("readoutOp", "qe1", None, I)
    save(I, out_st)


GST_sequence_file = 'Circuits_before_results.txt'
gate_macros = {"xpi2:0": x_pi2, "ypi2:0": y_pi2, "[]": id_gate}
gst = QuaGST(GST_sequence_file, model=smq1Q_XYI, basic_gates_macros=gate_macros, N_shots=1, post_circuit=post_circuit,
             config=config, quantum_machines_manager=qmm, simulate=SimulationConfig(int(1e4)))
gst.run(300)
# gst.run_IO()

results = gst.results
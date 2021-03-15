from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import QuantumMachine
from qm import LoopbackInterface
from qm import SimulationConfig
from bakary import *
from qm.qua import *
from test_configuration import *

with baking(config=config, padding_method="symmetric_r") as b:
    const_Op = [0.3, 0.3, 0.3, 0.3, 0.3]
    const_Op2 = [0.2, 0.2, 0.2, 0.3, 0.4]
    b.add_Op("const_Op_b", "fluxline", const_Op) #Add marker name as optional
    b.add_Op("const_Op2", "qe1", [const_Op, const_Op2])
    Op3 = [1., 1., 1.]
    Op4 = [2., 2., 2.]
    b.add_Op("Op3", "qe1", [Op3, Op4])
    b.play("const_Op2", "qe1")
    print("wait")
    b.play_at("Op3", "qe1", t=-2)
    #b.wait(-3, "qe1")
    #b.play("Op3", "qe1")
    #b.play("Op3", "qe1")

    #b.ramp(0.2, 6, "fluxline")
    #b.align("qe1", "fluxline")

# with baking(config=config, padding_method="symmetric_r") as b2:
#     # gaussianOp = gauss(100, 0.4, 3, 1, 8)
#     # derGaussOp = gauss_der(100, 0.4, 3, 1, 8)
#     # b.add_Op('Gauss_Op_b', "qe1", [gaussianOp, derGaussOp])
#     const_Op = [0.3, 0.3, 0.3, 0.3, 0.3]
#     const_Op2 = [0.2, 0.2, 0.2, 0.3, 0.4]
#     b2.add_Op("const_Op_b", "fluxline", const_Op) #Add marker name as optional
#     b2.add_Op("const_Op2", "qe1", [const_Op, const_Op2])
#     b2.play("const_Op2", "qe1")
#     b2.ramp(0.2, 6, "fluxline")
#     b2.align("qe1", "fluxline")
# with baking(config=config, padding_method="symmetric_r") as b3:
#     # gaussianOp = gauss(100, 0.4, 3, 1, 8)
#     # derGaussOp = gauss_der(100, 0.4, 3, 1, 8)
#     # b.add_Op('Gauss_Op_b', "qe1", [gaussianOp, derGaussOp])
#     const_Op = [0.3, 0.3, 0.3, 0.3, 0.3]
#     const_Op2 = [0.2, 0.2, 0.2, 0.3, 0.4]
#     b3.add_Op("const_Op_b", "fluxline", const_Op) #Add marker name as optional
#     b3.add_Op("const_Op2", "qe1", [const_Op, const_Op2])
#     b3.play("const_Op2", "qe1")
#     b3.ramp(0.2, 6, "fluxline")
#     b3.align("qe1", "fluxline")
# b_list = [b, b2, b3]
#
# qmm = QuantumMachinesManager("3.122.60.129")
# QM = qmm.open_qm(config)
#
# with program() as prog:
#     for b in b_list:
#         b.run()
#
# job = qmm.simulate(config, prog,
#                    SimulationConfig(int(1000), simulation_interface=LoopbackInterface(
#                        [("con1", 1, "con1", 1)])))  # Use LoopbackInterface to simulate the response of the qubit
#
# samples = job.get_simulated_samples()
#
# samples.con1.plot()
#


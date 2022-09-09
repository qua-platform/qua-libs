"""
A script used for playing with QUA
"""
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *


###################
# The QUA program #
###################

with program() as hello_qua:

    # a = declare(fixed, value=1)
    # b = declare(fixed, value=0.3)
    # c = declare(fixed)

    play("const", "ensemble")
    # play('const'*amp(0.5), 'ensemble', duration=200)
    #
    # wait(100)
    #
    # update_frequency('ensemble', int(10e6))
    # play('const', 'ensemble')
    #
    # update_frequency('ensemble', int(20e6))
    # play('const', 'ensemble', duration=300)
    #
    # play('const', 'ensemble')
    # frame_rotation_2pi(0.25, 'ensemble')
    # play('const', 'ensemble')
    #
    # reset_frame('ensemble')
    #
    # assign(c, a - b)
    # play('const'*amp(c), 'ensemble')
    #
    # play('const', 'ensemble', chirp=(100000, 'Hz/nsec'))
    #
    # align()
    #
    # wait(100)
    #
    # play('X', 'ensemble')
    # frame_rotation_2pi(0.25, 'ensemble')
    # play('X', 'ensemble')
    # play('X'*amp(c), 'ensemble')
    #
    # reset_frame('ensemble')

################################
# Open quantum machine manager #
################################

# qmm = QuantumMachinesManager(host=qop_ip, port="80")
qmm = QuantumMachinesManager(qop_ip)

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=2000,
        simulation_interface=LoopbackInterface(([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]), latency=180),
    )
    job = qmm.simulate(config, hello_qua, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    qm = qmm.open_qm(config)

    job = qm.execute(hello_qua)  # execute QUA program
